"""答疑决策链：课件优先，其次联网，最后通用知识。

产品取向（2026-10 调整）
------------------------
学生问的问题如果课件没覆盖，**不能直接拒答**。按依据强弱分三档作答，
每一档都在回答开头明确标出依据来源：

* ``kb``      课件里有可用依据，回答带页码引用；
* ``web``     课件没有，但联网搜索到了，回答带来源链接；
* ``general`` 课件和网络都没有，用模型自己的知识作答；
* ``handoff`` 只在模型故障、或者确实生成不出任何回答时才用。

「防胡说」靠的是**标注**，不是拒答：学生一眼能看出这句话是老师课件里写的，
还是 AI 自己补充的，从而决定要不要拿它当复习依据。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Sequence

from .llm import LLMError, OpenAICompatClient
from .retrieval import (
    RetrievalResult,
    build_context,
    looks_like_complex_question,
    retrieve,
)
from .settings import Settings
from .store import Hit, Store
from .textutils import (
    extract_citation_pages,
    has_any_citation,
    has_external_supplement,
    mentions_missing_coverage,
    validate_citations,
)
from .websearch import WebSearchUnavailable, search as web_search

MODE_KB = "kb"
MODE_WEB = "web"
MODE_GENERAL = "general"
MODE_HANDOFF = "handoff"

#: 前端和报告里显示的依据标签。措辞刻意直白，学生不用理解「RAG」是什么。
SOURCE_LABELS = {
    MODE_KB: "依据课件",
    MODE_WEB: "网络搜索·未审核",
    MODE_GENERAL: "课件未覆盖·通用知识",
    MODE_HANDOFF: "需人工确认",
}

#: 外部作答（联网/通用知识）的默认抬头。可在 config/retrieval.yaml 的 external 段改。
DEFAULT_WEB_NOTE = "【网络搜索结果·未经审核，请与课件核对】"
DEFAULT_GENERAL_NOTE = "【课件未覆盖·以下为通用知识解答，仅供参考】"

#: 这些词说明当前这句是追问，脱离上一轮就说不清楚。
#: 命中时把上一轮学生问题拼进检索查询，避免「不是，是那种……」被当成新问题。
FOLLOWUP_MARKERS = (
    "不是",
    "那种",
    "这个",
    "那个",
    "它",
    "它们",
    "刚才",
    "上面",
    "前面",
    "还有",
    "另外",
    "接着",
    "再说",
    "换个",
    "举例",
)

#: 拼进检索查询的历史上限（字），避免把整段对话塞进向量模型
RETRIEVAL_HISTORY_CHARS = 200


def build_retrieval_query(
    question: str, history: Sequence[dict] | None = None
) -> str:
    """把追问和上一轮问题拼成一个自包含的检索查询。

    只在这种句子明显依赖上文时才拼：否则会把无关的上一题带进来，
    反而降低检索准确率。
    """
    query = (question or "").strip()
    previous = [
        str(item.get("content") or "").strip()
        for item in (history or [])
        if item.get("role") == "user" and str(item.get("content") or "").strip()
    ]
    if not previous:
        return query
    if not any(marker in query for marker in FOLLOWUP_MARKERS):
        return query
    context = " ".join(previous[-2:])[-RETRIEVAL_HISTORY_CHARS:]
    return f"{context} {query}".strip()


def _history_block(history: Sequence[dict] | None, *, limit: int = 4) -> str:
    """给生成模型的对话上下文。只保留最近几轮，且每条截断。"""
    turns: list[str] = []
    for item in list(history or [])[-limit:]:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        role = "学生" if item.get("role") == "user" else "助教"
        turns.append(f"{role}：{content[:400]}")
    if not turns:
        return ""
    return "之前的对话（只用来理解当前问题，不要重复回答）：\n" + "\n".join(turns) + "\n\n"


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
    #: 联网作答时的来源链接，前端直接展示，学生可以点开核对
    sources: list[str] = field(default_factory=list)
    #: 课件作答时引用的原页图片：(资料编号, 页码)。前端据此展示课件原图
    image_pages: list[tuple[str, int]] = field(default_factory=list)
    #: 是否因模型调用失败而降级。降级时不能把「资料未覆盖」当成真的结论——
    #: 那会把「服务故障」说成「课件没有」，对学生是误导。
    failed: bool = False

    @property
    def source_level(self) -> str:
        return self.mode

    @property
    def source_label(self) -> str:
        return SOURCE_LABELS.get(self.mode, "")

    @property
    def page_labels(self) -> list[str]:
        return [hit.page_label for hit in self.hits]


def _handoff_text(settings: Settings) -> str:
    return settings.prompt("handoff_message")


def _external_note(settings: Settings, key: str, default: str) -> str:
    external = settings.retrieval.get("external") or {}
    value = external.get(key) if isinstance(external, dict) else None
    return str(value or default).strip()


def _ensure_note(text: str, note: str) -> str:
    """保证回答第一行是依据标签。

    模型偶尔会忘掉这条格式要求，或者把它加粗。这里做一次兜底拼接，
    而不是因为格式不合规就把整段回答扔掉——内容本身通常是对的。
    """
    body = (text or "").strip()
    if not body:
        return note
    head = "\n".join(body.splitlines()[:2])
    if note in head:
        return body
    # 容忍模型把【】换成其他括号、或者加了 ** 加粗
    plain = note.strip("【】")
    if plain and plain in head:
        return body
    return f"{note}\n\n{body}"


def _labeled_block(title: str, content: str) -> str:
    body = (content or "").strip()
    if not body:
        return ""
    return f"{title}\n\n{body}\n\n"


def collect_image_pages(
    hits: list[Hit], citations: list[int], *, limit: int = 3
) -> list[tuple[str, int]]:
    """把回答引用的页码映射回「资料编号 + 页码」，用于取课件原图。

    同一页码可能同时被多份资料的检索结果命中（各资料有自己的第 25 页），
    这里按检索命中的块去对应，既不会拿错图，也不会把别份资料的图误附上去。
    """
    found: list[tuple[str, int]] = []
    for page in citations:
        for hit in hits:
            if hit.page_no is None:
                continue
            end = hit.page_end or hit.page_no
            if not (hit.page_no <= page <= end):
                continue
            doc_key = (hit.chunk_key or "").split("#", 1)[0]
            if not doc_key:
                continue
            item = (doc_key, page)
            if item not in found:
                found.append(item)
            if len(found) >= limit:
                return found
    return found


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
    history: Sequence[dict] | None = None,
) -> Answer:
    started = time.perf_counter()
    course = course_id or settings.default_course_id
    course_name = settings.course_name(course)
    model_name = generate_client.config.name
    retrieval_query = build_retrieval_query(question, history)
    history_block = _history_block(history)

    def finish(answer: Answer) -> Answer:
        answer.latency_ms = int((time.perf_counter() - started) * 1000)
        if log:
            try:
                store.log_qa(
                    {
                        "course_id": course,
                        "session_id": session_id,
                        "question": question,
                        "retrieved_chunk_ids": [
                            hit.chunk_id for hit in answer.hits if hit.chunk_id
                        ],
                        "answer": answer.text,
                        "mode": answer.mode,
                        "citations": answer.citations or None,
                        "citation_valid": answer.citation_valid,
                        "top_score": answer.top_score,
                        "model": model_name,
                        "latency_ms": answer.latency_ms,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - 日志失败不能拖垮回答
                print(f"[警告] 问答日志写入失败：{exc}", file=sys.stderr)
        return answer

    def handoff(note: str, *, hits=None, score: float = 0.0, failed: bool = False) -> Answer:
        return finish(
            Answer(
                question=question,
                mode=MODE_HANDOFF,
                text=_handoff_text(settings),
                hits=list(hits or []),
                top_score=score,
                model=model_name,
                note=note,
                failed=failed,
            )
        )

    def answer_external(
        mode: str,
        note: str,
        context_block: str,
        *,
        hits=None,
        score: float = 0.0,
        sources=None,
        note_extra: str = "",
    ) -> Answer:
        system = settings.prompt("qa_answer_external", "system").format(
            course_name=course_name, mode_note=note
        )
        user = settings.prompt("qa_answer_external", "user").format(
            context_block=context_block, question=question
        )
        user = history_block + user
        try:
            text = generate_client.chat(system=system, user=user)
        except LLMError as exc:
            return handoff(
                f"外部作答失败：{exc}",
                hits=hits,
                score=score,
                failed=True,
            )

        text = _ensure_note(text, note)
        cited = extract_citation_pages(text)
        note_parts = [note_extra] if note_extra else []
        if cited:
            # 外部作答里冒出来的页码没有依据，不能透给学生
            note_parts.append(f"模型仍写了页码 {cited}，已忽略这些引用")
        return finish(
            Answer(
                question=question,
                mode=mode,
                text=text,
                hits=list(hits or []),
                top_score=score,
                citation_valid=not cited,
                model=model_name,
                note="；".join(part for part in note_parts if part),
                sources=list(sources or []),
            )
        )

    # ---- 复杂综合题：配置里的硬拦截（默认已清空，保留机制备用）----
    if looks_like_complex_question(settings, question):
        store.record_gap(course, question)
        return handoff("命中复杂题拦截规则")

    # ---- ① 检索 ----
    result: RetrievalResult | None = None
    retrieval_error = ""
    try:
        result = retrieve(
            settings,
            store,
            embed_client,
            retrieval_query,
            course_id=course,
            rerank_client=rerank_client,
        )
    except LLMError as exc:
        # 检索挂了不代表不能答：先记下原因，后面用联网/通用知识兜底
        retrieval_error = f"向量检索失败：{exc}"

    fallback_hits = result.hits if result is not None else []
    fallback_score = result.top_score if result is not None else 0.0
    fallback_reason = retrieval_error

    # ---- ② 课件里有依据：正常作答，要求带页码 ----
    if result is not None and result.has_evidence:
        context = build_context(settings, result.hits)
        system = settings.prompt("qa_answer", "system").format(course_name=course_name)
        user = settings.prompt("qa_answer", "user").format(
            context=context, question=question
        )
        user = history_block + user
        allowed = result.allowed_pages

        text = ""
        invalid: list[int] = []
        passed = False
        for attempt in range(2):
            prompt_user = user
            if attempt == 1:
                good = "、".join(f"第 {page} 页" for page in sorted(allowed))
                if invalid:
                    bad = "、".join(f"第 {page} 页" for page in invalid)
                    prompt_user += (
                        f"\n\n注意：上一次回答里引用了参考资料中不存在的页码（{bad}）。"
                        f"参考资料里真实存在的页码只有（{good}），请只用这些页码重新作答。"
                    )
                else:
                    # 完全没有页码引用是另一种情况：不能拼出「不存在的页码（）」那种空话，
                    # 要直接说清要求。实测模型会写出一段很好的回答却漏掉页码。
                    prompt_user += (
                        f"\n\n注意：上一次回答里一个页码都没有标注。"
                        f"参考资料里的页码有（{good}）。"
                        f"请重新作答，并在每个关键结论后面用（第 X 页）标出它出自哪一页。"
                    )
            try:
                text = generate_client.chat(system=system, user=prompt_user)
            except LLMError as exc:
                return handoff(
                    f"生成失败：{exc}",
                    hits=result.hits,
                    score=result.top_score,
                    failed=True,
                )

            ok, invalid = validate_citations(text, allowed)
            if ok and has_any_citation(text):
                passed = True
                break
            if (
                ok
                and not has_any_citation(text)
                and not settings.retrieval.get("require_citation", True)
            ):
                passed = True
                break

        if passed:
            gap_note = ""
            if mentions_missing_coverage(text) or has_external_supplement(text):
                store.record_gap(course, question)
                gap_note = "回答里明确说了资料未完全覆盖，已记入知识缺口清单"
            cited_pages = extract_citation_pages(text)
            return finish(
                Answer(
                    question=question,
                    mode=MODE_KB,
                    text=text,
                    citations=cited_pages,
                    citation_valid=True,
                    hits=result.hits,
                    top_score=result.top_score,
                    image_pages=collect_image_pages(result.hits, cited_pages),
                    model=model_name,
                    note=gap_note,
                )
            )

        # 检索到了、但回答始终过不了页码校验：不要转人工，换成通用知识作答
        store.record_gap(course, question)
        fallback_reason = (
            f"课件里有相关内容，但回答没能通过页码校验（越界页码 {invalid} 或没有引用）"
        )
    elif result is not None:
        # 知识库里确实没有可用依据——这是内容缺口，记下来给补资料用
        store.record_gap(course, question)
        fallback_reason = result.reason or "课件里没有找到能回答这个问题的内容"

    # ---- ③ 联网兜底 ----
    web_error = ""
    if settings.retrieval.get("allow_web", False):
        try:
            web = web_search(question)
        except WebSearchUnavailable as exc:
            web_error = str(exc)
        else:
            return answer_external(
                MODE_WEB,
                _external_note(settings, "web_note", DEFAULT_WEB_NOTE),
                _labeled_block(
                    "以下是从网络检索到的资料（未经审核，只作参考）：", web.context
                ),
                hits=fallback_hits,
                score=fallback_score,
                sources=web.sources,
                note_extra=f"联网兜底（{web.provider}）；{fallback_reason}",
            )
    else:
        web_error = "联网未开启"

    # ---- ④ 通用知识兜底 ----
    if settings.retrieval.get("allow_general", True):
        context_block = ""
        if fallback_hits:
            background = build_context(settings, fallback_hits[:3])
            context_block = _labeled_block(
                "以下是课件里与问题可能相关的片段。它们相关度不高，只作背景参考，"
                "**不要当成答案依据，也不要标注它们的页码**：",
                background,
            )
        note_extra = fallback_reason
        if web_error:
            note_extra = f"{note_extra}；{web_error}" if note_extra else web_error
        return answer_external(
            MODE_GENERAL,
            _external_note(settings, "general_note", DEFAULT_GENERAL_NOTE),
            context_block,
            hits=fallback_hits,
            score=fallback_score,
            note_extra=note_extra,
        )

    # ---- ⑤ 确实无法作答 ----
    return handoff(
        f"{fallback_reason}；{web_error}".strip("；"),
        hits=fallback_hits,
        score=fallback_score,
    )
