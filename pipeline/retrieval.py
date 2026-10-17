"""步骤①②：向量召回 + 「有没有依据」的判定。

判定口径只在这里定义一次，在线服务与建库自检共用，
避免「自检时命中、学生用时兜底」这种两套标准的问题。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .llm import OpenAICompatClient
from .settings import Settings
from .store import Hit, Store


@dataclass
class RetrievalResult:
    hits: list[Hit] = field(default_factory=list)
    top_score: float = 0.0
    has_evidence: bool = False
    reason: str = ""

    @property
    def allowed_pages(self) -> set[int]:
        pages: set[int] = set()
        for hit in self.hits:
            if hit.page_no is None:
                continue
            start = hit.page_no
            end = hit.page_end or hit.page_no
            pages.update(range(start, end + 1))
        return pages


def looks_like_complex_question(settings: Settings, question: str) -> bool:
    patterns = settings.retrieval.get("handoff_patterns") or []
    for pattern in patterns:
        if re.search(str(pattern), question):
            return True
    return False


def rerank_hits(
    settings: Settings,
    client: OpenAICompatClient | None,
    question: str,
    hits: Sequence[Hit],
) -> list[Hit]:
    """用交叉编码器重排向量召回的结果。

    为什么必须做这一步：向量检索只衡量「语义接近」，不衡量「能不能回答」。
    实测一道问二端口网络 Z 参数的题，向量最高分 0.715（判定为有依据），
    但召回的全是电容串并联的内容——模型于是用自己的知识答了一遍，
    还挂上了那些真实存在却内容无关的页码。这是最危险的一类错误：
    引用校验只查页码存不存在，查不出内容不对。
    """
    if client is None or not hits:
        return list(hits)

    top_n = int(settings.retrieval.get("rerank_top_n", 8))
    candidates = list(hits[:top_n])
    scored = client.rerank(
        query=question, documents=[hit.content for hit in candidates]
    )

    reranked: list[Hit] = []
    for index, score in scored:
        if 0 <= index < len(candidates):
            hit = candidates[index]
            reranked.append(
                Hit(
                    chunk_key=hit.chunk_key,
                    page_no=hit.page_no,
                    page_end=hit.page_end,
                    chapter=hit.chapter,
                    section_title=hit.section_title,
                    content=hit.content,
                    trust_level=hit.trust_level,
                    score=score,
                    document_id=hit.document_id,
                    chunk_id=hit.chunk_id,
                    vector_score=hit.vector_score or hit.score,
                )
            )
    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked


def retrieve(
    settings: Settings,
    store: Store,
    client: OpenAICompatClient,
    question: str,
    *,
    course_id: str | None = None,
    top_k: int | None = None,
    rerank: bool = True,
    rerank_client: OpenAICompatClient | None = None,
) -> RetrievalResult:
    course = course_id or settings.default_course_id
    limit = top_k or int(settings.retrieval.get("retrieve_top_k", 20))
    max_trust = int(settings.retrieval.get("max_trust_level", 2))

    vectors = client.embed([question])
    hits = store.search(course, vectors[0], limit=limit, max_trust=max_trust)

    if rerank and rerank_client is not None:
        hits = rerank_hits(settings, rerank_client, question, hits)
        threshold = float(settings.retrieval.get("min_rerank_score", 0.5))
        metric = "重排"
    else:
        threshold = float(settings.retrieval.get("min_score", 0.35))
        metric = "向量"

    top_score = hits[0].score if hits else 0.0
    if not hits:
        return RetrievalResult(
            hits=[], top_score=0.0, has_evidence=False, reason="知识库里没有任何内容块"
        )
    if top_score < threshold:
        return RetrievalResult(
            hits=hits,
            top_score=top_score,
            has_evidence=False,
            reason=f"最高{metric}相关度 {top_score:.3f} 低于阈值 {threshold:.2f}",
        )
    return RetrievalResult(
        hits=hits,
        top_score=top_score,
        has_evidence=True,
        reason=f"最高{metric}相关度 {top_score:.3f}",
    )


def build_context(settings: Settings, hits: Sequence[Hit], rerank_top_k: int | None = None) -> str:
    """把命中的内容块拼成给模型看的「参考资料」，每条都带页码标签。"""
    budget = int(settings.retrieval.get("max_context_chars", 6000))
    top_n = rerank_top_k or int(settings.retrieval.get("rerank_top_k", 5))

    blocks: list[str] = []
    used = 0
    for hit in hits[:top_n]:
        section = f"｜{hit.section_title}" if hit.section_title else ""
        block = f"【{hit.page_label}{section}】\n{hit.content}"
        if used + len(block) > budget and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)
