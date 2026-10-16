"""Markdown 清洗、页码标记、引用校验等纯函数。

这些函数不碰网络也不碰数据库，是单元测试覆盖的重点——
「提问有没有编造」最终靠的就是引用校验这一小段逻辑。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 页码标记：整篇 Markdown 里用 <!-- page: N --> 记录每个自然页的起点
# ---------------------------------------------------------------------------

PAGE_MARKER_RE = re.compile(r"^\s*<!--\s*page\s*[:：]\s*(\d+)\s*-->\s*$", re.IGNORECASE)


def page_marker(page_no: int) -> str:
    return f"<!-- page: {page_no} -->"


def strip_page_markers(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not PAGE_MARKER_RE.match(line)
    ).strip()


FIGURE_PLACEHOLDER_RE = re.compile(r"\[\s*图\s*第?\s*(\d+)\s*页?\s*-\s*(\d+)\s*[:：][^\]]*\]")
UNCLEAR_MARKER_RE = re.compile(r"\[此处原图不清晰\]")


# ---------------------------------------------------------------------------
# 清洗
# ---------------------------------------------------------------------------

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)


def strip_code_fences(text: str) -> str:
    """模型有时会把整页内容包进 ```markdown 围栏，去掉最外层围栏。"""
    stripped = text.strip()
    match = re.match(r"^```[a-zA-Z]*\s*\n(.*)\n```$", stripped, re.DOTALL)
    return match.group(1).strip() if match else stripped


def normalize_markdown(text: str) -> str:
    """统一换行、去掉零宽字符与多余空行，顺手修正常见 LaTeX 转义漂移。"""
    text = strip_code_fences(text)
    text = text.translate(_ZERO_WIDTH)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 模型输出 \( ... \) / \[ ... \] 时统一成 $ / $$，便于后续切块与检索
    text = text.replace("\\(", "$").replace("\\)", "$")
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_probably_empty(text: str) -> bool:
    """去掉标记后几乎没有实质内容——空白页或纯装饰页。"""
    plain = strip_page_markers(text)
    plain = UNCLEAR_MARKER_RE.sub("", plain)
    plain = re.sub(r"[#*`>\-\s]", "", plain)
    return len(plain) < 5


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def estimate_tokens(text: str) -> int:
    """粗估 token 数：中文按字计，拉丁字母数字按词计。

    只用于填 chunks.token_count 做观测，不参与计费精确核算，
    因此不引入 tiktoken 之类的分词器依赖。
    """
    return len(_CJK_RE.findall(text)) + len(_LATIN_TOKEN_RE.findall(text))


def has_formula(text: str) -> bool:
    if "$" in text:
        return True
    return bool(re.search(r"\\[a-zA-Z]+", text))


def count_figure_placeholders(text: str) -> int:
    return len(FIGURE_PLACEHOLDER_RE.findall(text))


def count_unclear_marks(text: str) -> int:
    return len(UNCLEAR_MARKER_RE.findall(text))


# ---------------------------------------------------------------------------
# 句子切分（切块时用来在不破坏语义的前提下截断超长段落）
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(r"(?<=[。！？；!?;])")


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_END_RE.split(text)]
    return [part for part in parts if part]


# ---------------------------------------------------------------------------
# 引用校验：回答里出现的页码必须真实存在于检索结果中
# ---------------------------------------------------------------------------

_CITATION_PATTERNS = (
    re.compile(r"第\s*(\d{1,4})\s*页"),
    re.compile(r"页码\s*[:：]\s*(\d{1,4})"),
    re.compile(r"(?<![\w.])(?:[pP]\.)\s*(\d{1,4})(?!\d)"),
)


def extract_citation_pages(text: str) -> list[int]:
    """按出现顺序抽出回答中声明的页码，去重。"""
    seen: list[int] = []
    for pattern in _CITATION_PATTERNS:
        for match in pattern.finditer(text):
            page = int(match.group(1))
            if page not in seen:
                seen.append(page)
    return seen


def validate_citations(answer: str, allowed_pages: set[int]) -> tuple[bool, list[int]]:
    """校验回答里的页码是否都在检索结果里。

    返回 (是否通过, 越界页码列表)。
    """
    cited = set(extract_citation_pages(answer))
    invalid = sorted(page for page in cited if page not in allowed_pages)
    return (not invalid), invalid


def has_any_citation(answer: str) -> bool:
    return bool(extract_citation_pages(answer))
