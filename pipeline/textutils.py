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

#: 从图上照抄回来的文字（提示词规则 4b 要求的前缀）
FIGURE_TEXT_MARKER = "[图上文字]"
FIGURE_TEXT_RE = re.compile(r"\[\s*图上文字\s*\]")

#: 版面上当项目符号用的装饰符号。只删行首，正文里出现的一律不动。
BULLET_GLYPHS = "◆◇●○■□▪▫★☆▲▼△▽※·•"


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
    text = _strip_bullet_glyphs(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_bullet_glyphs(text: str) -> str:
    """去掉行首当装饰用的 ◆□● 这类符号。

    课件里的项目符号常常被当成正文一起转写，比如「## □ 实际电路器件种类繁多」。
    这些符号会跟着标题进入切块元数据、再出现在引用里，属于纯噪声，
    所以在源头清掉。**只动行首**：正文中间出现的符号可能是有意义的。
    """
    glyphs = re.escape(BULLET_GLYPHS)
    heading_re = re.compile(rf"^(#{{1,6}}\s+)[{glyphs}]+\s*")
    list_re = re.compile(rf"^(\s*(?:[-*+]|\d+\.)\s+)[{glyphs}]+\s*")

    cleaned: list[str] = []
    for line in text.splitlines():
        line = heading_re.sub(r"\1", line)
        line = list_re.sub(r"\1", line)
        cleaned.append(line)
    return "\n".join(cleaned)


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


def count_figure_text_marks(text: str) -> int:
    """统计「[图上文字]」出现了几次。

    这个数字是给人工抽查用的：它代表有多少内容是从图上看出来的、
    而不是文字层里本来就有的，也就是抽查最该盯的地方。
    """
    return len(FIGURE_TEXT_RE.findall(text))


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

#: 引用页码的几种写法。**必须支持合并写法**，否则会漏判：
#: 实测模型经常写「（第25、26页）」「（第25-26页）」，而早期只认「第25页」，
#: 结果明明引用了正确页码却被判成「一个页码都没写」→ 假转人工。
_CITATION_BLOCK_RE = re.compile(
    # 分隔符要覆盖模型实际会写的各种横线：- — – − ~ 至
    r"第\s*(\d{1,4}(?:\s*[、,，/~\-—–−至]\s*\d{1,4})*)\s*页"
)
_CITATION_RANGE_RE = re.compile(r"(\d{1,4})\s*[~\-—–−至]\s*(\d{1,4})")
_CITATION_NUMBER_RE = re.compile(r"\d{1,4}")

#: 区间展开的上限：写「第25-9999页」这种明显是笔误，不要展开成几千个页码
_MAX_RANGE_SPAN = 80

_OTHER_CITATION_PATTERNS = (
    re.compile(r"页码\s*[:：]\s*(\d{1,4})"),
    re.compile(r"(?<![\w.])(?:[pP]\.)\s*(\d{1,4})(?!\d)"),
)


def extract_citation_pages(text: str) -> list[int]:
    """按出现顺序抽出回答中声明的页码，去重。

    支持四种写法：「第25页」「第25、26页」「第25-26页」「第25至28页」。
    合并写法很常见，解析漏了会直接造成假转人工。
    """
    seen: list[int] = []

    def record(page: int) -> None:
        if page > 0 and page not in seen:
            seen.append(page)

    for block in _CITATION_BLOCK_RE.finditer(text):
        body = block.group(1)
        covered = [False] * len(body)
        for span in _CITATION_RANGE_RE.finditer(body):
            start, end = int(span.group(1)), int(span.group(2))
            if start <= end and end - start <= _MAX_RANGE_SPAN:
                for page in range(start, end + 1):
                    record(page)
            else:
                record(start)
                record(end)
            for index in range(span.start(), span.end()):
                covered[index] = True
        for number in _CITATION_NUMBER_RE.finditer(body):
            if not covered[number.start()]:
                record(int(number.group(0)))

    for pattern in _OTHER_CITATION_PATTERNS:
        for match in pattern.finditer(text):
            record(int(match.group(1)))

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


#: 模型自己声明「资料没覆盖」时的说法。命中它意味着这确实是个知识缺口，
#: 即使回答里给了相关的其他内容、走了 kb 分支，也必须记进缺口清单。
#:
#: 刻意只认**明确指向整体覆盖**的说法，不认「资料里没有提到某个干扰项」——
#: 实测一道选择题答对了 B，但答案里说了「资料里没有提到 C 选项那种情况」，
#: 宽泛的匹配把它误记成了知识缺口。
_NO_COVERAGE_PATTERNS = (
    "现有资料没有覆盖",
    "现有资料未覆盖",
    "资料里没有覆盖",
    "资料没有覆盖",
    "资料中未覆盖",
    "未覆盖",
    "资料里没有出现",
    "资料中没有出现",
    "资料里没有讲到",
    "资料里没有提及",
    "资料中没有提及",
    "现有资料不包含",
    "没有覆盖",
)

#: 模型按提示词要求写的显式标记。比猜措辞可靠得多，出现在哪儿都认。
COVERAGE_MARKER = "【资料未覆盖】"

#: 课件只覆盖了一部分时，模型用来标出「这一段是课外知识」的标题。
#: 它和「资料未覆盖」是两种不同的话：前者是「课件讲了大部分，我补一点」，
#: 后者是「课件根本没讲」。两者都说明资料不够完整，都要记进缺口清单。
EXTERNAL_SUPPLEMENT_MARKER = "【课件外补充"


def has_external_supplement(answer: str) -> bool:
    """回答里是否出现了「课件外补充」小节。"""
    return EXTERNAL_SUPPLEMENT_MARKER in answer


#: 没有标记时的兜底：只看开头这么多字。「资料没覆盖」这种结论一定写在开头，
#: 写在后面的通常只是针对某个细节或某个干扰项的说明。
COVERAGE_HEAD_CHARS = 120


def mentions_missing_coverage(answer: str) -> bool:
    """回答里是否出现了「这件事资料里没有」的声明。

    为什么需要它：实测有过这种情况——学生问「回转器怎么用在有源滤波器设计里」，
    模型老实回答「现有资料没有覆盖回转器」，然后列出资料里确实有的滤波器内容
    并逐条标了页码。这个回答是诚实且有用的，但它走的是「有依据」分支，
    于是这个真实的缺口**没被记进知识缺口清单**，后续就没人去补资料。
    """
    if COVERAGE_MARKER in answer:
        return True
    head = answer[:COVERAGE_HEAD_CHARS]
    return any(pattern in head for pattern in _NO_COVERAGE_PATTERNS)
