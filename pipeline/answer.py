"""答疑决策链（方案 §4.2）的建库自检版。

和将来在线服务用的是同一套逻辑，目的是让「阶段 0 验收」可以现在就做：
从提问一路走到「回答 + 正确页码引用」，并且资料没覆盖时老老实实转人工。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .llm import LLMError, OpenAICompatClient
from .retrieval import RetrievalResult, build_context, looks_like_complex_question, retrieve
from .settings import Settings
from .store import Hit, Store
from .textutils import (
    extract_citation_pages,
    has_any_citation,
    mentions_missing_coverage,
    validate_citations,
)
from .websearch import WebSearchUnavailable, search as web_search

MODE_KB = "kb"
MODE_WEB = "web"
MODE_HANDOFF = "handoff"


@dataclass
class Answer:
    question: str
    mode: str
    text: str
    citations: list[int] = field(default_factory=list)
    citation_valid: bool = True
    hits: list[Hit] = field(default_factory=list)
    top_score: float = 0.0
    model: str = ""
    latency_ms: int = 0
    note: str = ""
    #: 是否因模型调用失败而降级。降级时不能把「资料未覆盖」当成真的结论——
    #: 那会把「服务故障」说成「课件没有」，对学生是误导。
    failed: bool = False

    @property
    def page_labels(self) -> list[str]:
        return [hit.page_label for hit in self.hits]


def _handoff_text(settings: Settings) -> str:
    return settings.prompt("handoff_message")


def answer_question(
    settings: Settings,
    store: Store,
    embed_client: OpenAICompatClient,
    generate_client: OpenAICompatClient,
    question: str,
    *,
    course_id: str | None = None,
    log: bool = True,
    rerank_client: OpenAICompatClient | None = None,
    session_id: int | None = None,
) -> Answer:
    started = time.perf_counter()
    course = course_id or settings.default_course_id
    model_name = generate_client.config.name

    def finish(answer: Answer) -> Answer:
        answer.latency_ms = int((time.perf_counter() - started) * 1000)
        if log:
            store.log_qa(
                {
                    "course_id": course,
                    "session_id": session_id,
                    "question": question,
                    "retrieved_chunk_ids": [hit.chunk_id for hit in answer.hits if hit.chunk_id],
                    "answer": answer.text,
                    "mode": answer.mode,
                    "citations": answer.citations or None,
                    "citation_valid": answer.citation_valid,
                    "top_score": answer.top_score,
                    "model": model_name,
                    "latency_ms": answer.latency_ms,
                }
            )
        return answer

    # ---- 复杂综合题：不进生成环节，直接转人工（方案 §4.2 的额外防线）----
    if looks_like_complex_question(settings, question):
        store.record_gap(course, question)
        return finish(
            Answer(
                question=question,
                mode=MODE_HANDOFF,
                text=_handoff_text(settings),
                model=model_name,
                note="命中复杂题拦截规则",
            )
        )

    # ---- ① 检索 + ② 依据判定 ----
    try:
        result: RetrievalResult = retrieve(
            settings,
            store,
            embed_client,
            question,
            course_id=course,
            rerank_client=rerank_client,
        )
    except LLMError as exc:
        return finish(
            Answer(
                question=question,
                mode=MODE_HANDOFF,
                text=_handoff_text(settings),
                model=model_name,
                note=f"向量化失败：{exc}",
                failed=True,
            )
        )

    if not result.has_evidence:
        store.record_gap(course, question)
        if settings.retrieval.get("allow_web"):
            try:
                web = web_search(question)
            except WebSearchUnavailable as exc:
                return finish(
                    Answer(
                        question=question,
                        mode=MODE_HANDOFF,
                        text=_handoff_text(settings),
                        hits=result.hits,
                        top_score=result.top_score,
                        model=model_name,
                        note=f"无依据且联网不可用：{exc}",
                    )
                )
            return finish(
                Answer(
                    question=question,
                    mode=MODE_WEB,
                    text=web.text,
                    hits=result.hits,
                    top_score=result.top_score,
                    citation_valid=False,
                    model=model_name,
                    note=f"联网兜底（{result.reason}），来源：{', '.join(web.sources[:3])}",
                )
            )

        return finish(
            Answer(
                question=question,
                mode=MODE_HANDOFF,
                text=_handoff_text(settings),
                hits=result.hits,
                top_score=result.top_score,
                model=model_name,
                note=result.reason,
            )
        )

    # ---- ③ 生成 ----
    context = build_context(settings, result.hits)
    system = settings.prompt("qa_answer", "system").format(
        course_name=settings.course_name(course)
    )
    user = settings.prompt("qa_answer", "user").format(context=context, question=question)
    allowed = result.allowed_pages

    text = ""
    invalid: list[int] = []
    for attempt in range(2):
        prompt_user = user
        if attempt == 1:
            if invalid:
                bad = "、".join(f"第 {page} 页" for page in invalid)
                good = "、".join(f"第 {page} 页" for page in sorted(allowed))
                prompt_user += (
                    f"\n\n注意：上一次回答里引用了参考资料中不存在的页码（{bad}）。"
                    f"参考资料里真实存在的页码只有（{good}），请只用这些页码重新作答。"
                )
            else:
                # 完全没有页码引用是另一种情况：不能拼出「不存在的页码（）」那种空话，
                # 要直接说清要求。实测模型会写出一段很好的回答却漏掉页码。
                good = "、".join(f"第 {page} 页" for page in sorted(allowed))
                prompt_user += (
                    f"\n\n注意：上一次回答里一个页码都没有标注。"
                    f"参考资料里的页码有（{good}）。"
                    f"请重新作答，并在每个关键结论后面用（第 X 页）标出它出自哪一页。"
                )
        try:
            text = generate_client.chat(system=system, user=prompt_user)
        except LLMError as exc:
            return finish(
                Answer(
                    question=question,
                    mode=MODE_HANDOFF,
                    text=_handoff_text(settings),
                    hits=result.hits,
                    top_score=result.top_score,
                    model=model_name,
                    note=f"生成失败：{exc}",
                    failed=True,
                )
            )

        ok, invalid = validate_citations(text, allowed)
        if ok and has_any_citation(text):
            break
        if ok and not has_any_citation(text) and not settings.retrieval.get("require_citation", True):
            break
    else:
        store.record_gap(course, question)
        return finish(
            Answer(
                question=question,
                mode=MODE_HANDOFF,
                text=_handoff_text(settings),
                hits=result.hits,
                top_score=result.top_score,
                citation_valid=False,
                model=model_name,
                note=f"引用校验未通过（越界页码 {invalid} 或没有任何页码引用）",
            )
        )

    ok, invalid = validate_citations(text, allowed)
    # 回答了、也带了页码，但明确说了「资料没有覆盖」——这仍然是知识缺口，
    # 必须记下来，否则这个缺口永远进不了待补资料清单。
    gap_note = ""
    if mentions_missing_coverage(text):
        store.record_gap(course, question)
        gap_note = "回答里明确说了资料未覆盖，已记入知识缺口清单"

    return finish(
        Answer(
            question=question,
            mode=MODE_KB,
            text=text,
            citations=extract_citation_pages(text),
            citation_valid=ok,
            hits=result.hits,
            top_score=result.top_score,
            model=model_name,
            note=gap_note,
        )
    )
