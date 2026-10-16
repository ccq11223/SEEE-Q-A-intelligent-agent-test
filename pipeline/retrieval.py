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


def retrieve(
    settings: Settings,
    store: Store,
    client: OpenAICompatClient,
    question: str,
    *,
    course_id: str | None = None,
    top_k: int | None = None,
) -> RetrievalResult:
    course = course_id or settings.default_course_id
    limit = top_k or int(settings.retrieval.get("retrieve_top_k", 20))
    max_trust = int(settings.retrieval.get("max_trust_level", 2))
    min_score = float(settings.retrieval.get("min_score", 0.35))

    vectors = client.embed([question])
    hits = store.search(course, vectors[0], limit=limit, max_trust=max_trust)

    top_score = hits[0].score if hits else 0.0
    if not hits:
        return RetrievalResult(
            hits=[], top_score=0.0, has_evidence=False, reason="知识库里没有任何内容块"
        )
    if top_score < min_score:
        return RetrievalResult(
            hits=hits,
            top_score=top_score,
            has_evidence=False,
            reason=f"最高相关度 {top_score:.3f} 低于阈值 {min_score:.2f}",
        )
    return RetrievalResult(hits=hits, top_score=top_score, has_evidence=True, reason="")


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
