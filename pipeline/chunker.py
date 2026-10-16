"""步骤 5：清洗与切块。

切块原则（方案 §4.1）：
  - 按章节标题的层级切块，一个块尽量落在一个小节里；
  - 单块 300-800 字，太短检索噪声大，太长引用不精确；
  - 每块必须带上：来源文件、页码、章节、可信级别——缺一个都不能入库。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .textutils import PAGE_MARKER_RE, estimate_tokens, split_sentences

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class Chunk:
    chunk_key: str
    page_no: int | None
    page_end: int | None
    chapter: str
    section_title: str
    content: str
    char_count: int
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_key": self.chunk_key,
            "page_no": self.page_no,
            "page_end": self.page_end,
            "chapter": self.chapter,
            "section_title": self.section_title,
            "content": self.content,
            "char_count": self.char_count,
            "token_count": self.token_count,
        }


@dataclass
class _Block:
    text: str
    page: int | None
    level: int  # 0 = 正文段落，1-6 = 标题级别
    path: tuple[str, ...] = ()  # 该块所在位置的标题路径


@dataclass
class ChunkConfig:
    target_chars: int = 550
    max_chars: int = 800
    min_chars: int = 200
    overlap_chars: int = 80
    flush_on_section_change: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ChunkConfig":
        return cls(
            target_chars=int(raw.get("target_chars", 550)),
            max_chars=int(raw.get("max_chars", 800)),
            min_chars=int(raw.get("min_chars", 200)),
            overlap_chars=int(raw.get("overlap_chars", 80)),
            flush_on_section_change=bool(raw.get("flush_on_section_change", True)),
        )


def _to_blocks(markdown: str) -> list[_Block]:
    blocks: list[_Block] = []
    heading_stack: list[str] = []
    page: int | None = None
    buffer: list[str] = []

    def flush_paragraph() -> None:
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        buffer.clear()
        if text:
            blocks.append(_Block(text=text, page=page, level=0, path=tuple(heading_stack)))

    for line in markdown.splitlines():
        marker = PAGE_MARKER_RE.match(line)
        if marker:
            flush_paragraph()
            page = int(marker.group(1))
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            del heading_stack[level - 1 :]
            heading_stack.append(title)
            blocks.append(
                _Block(
                    text=f"{'#' * level} {title}",
                    page=page,
                    level=level,
                    path=tuple(heading_stack),
                )
            )
            continue

        if not line.strip():
            flush_paragraph()
            continue

        buffer.append(line)

    flush_paragraph()
    return blocks


def _section_key(path: tuple[str, ...]) -> tuple[str, ...]:
    """用于判断「是否换了小节」：只看前两级标题，避免三级标题频繁切块。"""
    return path[:2]


def _split_long_text(
    text: str, *, max_chars: int, overlap_chars: int
) -> list[str]:
    """把超长段落按句子边界切开，避免单块远超 max_chars。"""
    sentences = split_sentences(text) or [text]
    pieces: list[str] = []
    current = ""
    carry = ""

    for sentence in sentences:
        candidate = f"{current}{sentence}"
        if current and len(candidate) > max_chars:
            pieces.append((carry + current).strip())
            carry = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = sentence
        else:
            current = candidate

    if current:
        pieces.append((carry + current).strip())
    return [piece for piece in pieces if piece]


def chunk_document(
    markdown: str,
    *,
    doc_key: str,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    cfg = config or ChunkConfig()
    blocks = _to_blocks(markdown)

    chunks: list[Chunk] = []
    current: list[_Block] = []
    carry: list[_Block] = []
    counter = 0

    def current_chars(items: list[_Block]) -> int:
        return sum(len(item.text) + 2 for item in items)

    def emit(items: list[_Block]) -> None:
        nonlocal counter, carry
        if not items:
            return

        body = [item for item in items if item.level == 0]
        if not body:
            # 只有标题、没有正文：不单独成块，留给下一块当开头
            carry = items
            return

        text = "\n\n".join(item.text for item in items).strip()
        pages = [item.page for item in items if item.page is not None]
        deepest = max((item.path for item in items), key=len, default=())

        counter += 1
        chunks.append(
            Chunk(
                chunk_key=f"{doc_key}#{counter:04d}",
                page_no=min(pages) if pages else None,
                page_end=max(pages) if pages and max(pages) != min(pages) else None,
                chapter=deepest[0] if deepest else "",
                section_title=" > ".join(deepest),
                content=text,
                char_count=len(text),
                token_count=estimate_tokens(text),
            )
        )

        if cfg.overlap_chars > 0 and body:
            tail_text = body[-1].text
            tail = tail_text[-cfg.overlap_chars :]
            carry = [_Block(text=tail, page=body[-1].page, level=0, path=body[-1].path)]
        else:
            carry = []

    def flush() -> None:
        nonlocal current, carry
        items = carry + current
        carry = []
        current = []
        emit(items)

    for block in blocks:
        if not block.text.strip():
            continue

        # 超长正文：先结束当前块，再按句子切开单独成块
        if block.level == 0 and len(block.text) > cfg.max_chars:
            flush()
            for piece in _split_long_text(
                block.text, max_chars=cfg.max_chars, overlap_chars=cfg.overlap_chars
            ):
                emit([_Block(text=piece, page=block.page, level=0, path=block.path)])
            continue

        if current or carry:
            existing = carry + current
            existing_chars = current_chars(existing)
            changed_section = _section_key(block.path) != _section_key(existing[0].path)
            if (
                existing_chars >= cfg.min_chars
                and changed_section
                and cfg.flush_on_section_change
            ) or (existing_chars + len(block.text) > cfg.max_chars and existing_chars >= cfg.min_chars):
                flush()

        current.append(block)

    flush()
    return chunks
