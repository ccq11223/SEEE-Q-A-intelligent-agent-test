"""步骤 3：结构化解析 —— 用视觉大模型把每一页课件转成结构化 Markdown。

这是整条链路质量的天花板（方案 §7 阶段 0 的结论），所以：
  - 默认逐页送图（vision），公式与电路图靠模型看，不靠文本层猜；
  - 每页结果落盘缓存，重跑只补失败页，不重复花钱；
  - 单页失败不中断整份资料，失败页单独记账，最后一起报给你。
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .convert import PageAsset, RenderedDoc
from .llm import LLMError, OpenAICompatClient
from .manifest import ManifestEntry
from .settings import Settings
from .textutils import (
    count_figure_placeholders,
    count_figure_text_marks,
    count_unclear_marks,
    has_formula,
    is_probably_empty,
    normalize_markdown,
    page_marker,
)


@dataclass
class PageParse:
    page_no: int
    markdown: str = ""
    source: str = "vision"  # vision / text / skipped
    cached: bool = False
    error: str | None = None
    has_formula: bool = False
    figure_count: int = 0
    figure_text_count: int = 0
    unclear_count: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ParseReport:
    pages: list[PageParse] = field(default_factory=list)

    @property
    def failures(self) -> list[PageParse]:
        return [page for page in self.pages if page.error]

    @property
    def cached_count(self) -> int:
        return sum(1 for page in self.pages if page.cached)

    def summary(self) -> str:
        total = len(self.pages)
        formula = sum(1 for page in self.pages if page.has_formula)
        figures = sum(page.figure_count for page in self.pages)
        figure_texts = sum(page.figure_text_count for page in self.pages)
        unclear = sum(page.unclear_count for page in self.pages)
        return (
            f"共 {total} 页｜缓存命中 {self.cached_count} 页｜含公式 {formula} 页｜"
            f"图片占位 {figures} 个｜图上文字 {figure_texts} 处｜"
            f"不清晰标记 {unclear} 处｜失败 {len(self.failures)} 页"
        )


def _resolve_mode(settings: Settings, asset: PageAsset) -> str:
    mode = str(settings.parsing.get("mode", "vision")).lower()
    if asset.image_path is None:
        return "text"
    if mode != "hybrid":
        return "vision"

    threshold = int(settings.parsing.get("hybrid_min_native_chars", 80))
    rich_text = len(asset.native_text) >= threshold
    if rich_text and not asset.has_embedded_image:
        return "text"
    return "vision"


def _cache_key(
    settings: Settings,
    entry: ManifestEntry,
    asset: PageAsset,
    mode: str,
    model_name: str,
) -> str:
    digest = hashlib.sha256()
    if asset.image_path is not None and asset.image_path.is_file():
        digest.update(asset.image_path.read_bytes())
    digest.update(asset.native_text.encode("utf-8", errors="replace"))
    digest.update(
        f"|{mode}|{model_name}|{settings.prompts_version}|{entry.doc_key}|{asset.page_no}".encode(
            "utf-8"
        )
    )
    return digest.hexdigest()[:40]


def _load_cache(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    content = payload.get("markdown")
    return content if isinstance(content, str) else None


def _save_cache(path: Path, markdown: str, meta: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"markdown": markdown, "meta": meta}, ensure_ascii=False),
        encoding="utf-8",
    )


def _text_user_prompt(entry: ManifestEntry, settings: Settings, asset: PageAsset, total: int) -> str:
    course_name = settings.course_name(entry.course_id)
    return (
        f"课程：{course_name}\n"
        f"资料：《{entry.title}》\n"
        f"这是第 {asset.page_no} 页，共 {total} 页。\n"
        f"本页没有可用图片，下面是自动提取的原始文本（公式可能已经错乱或被拆散）。\n"
        f"请整理成结构化 Markdown：忠实于原文，不要补充、不要解释；\n"
        f"能判断出的公式用 LaTeX 重写，判断不出的保留原样并加 [公式无法还原]。\n\n"
        f"----- 原始文本开始 -----\n{asset.native_text}\n----- 原始文本结束 -----"
    )


def _parse_one(
    entry: ManifestEntry,
    settings: Settings,
    asset: PageAsset,
    total: int,
    client: OpenAICompatClient,
    *,
    use_cache: bool,
) -> PageParse:
    mode = _resolve_mode(settings, asset)
    model_name = client.config.name
    cache_path = settings.paths.vision_cache() / (
        _cache_key(settings, entry, asset, mode, model_name) + ".json"
    )

    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            result = PageParse(page_no=asset.page_no, markdown=cached, source=mode, cached=True)
            annotate_page(result)
            return result

    system = settings.prompt("page_parse", "system")
    try:
        if mode == "vision" and asset.image_path is not None:
            user = settings.prompt("page_parse", "user").format(
                course_name=settings.course_name(entry.course_id),
                doc_title=entry.title,
                page_no=asset.page_no,
                page_count=total,
            )
            raw = client.chat_with_images(
                system=system, user=user, image_paths=[asset.image_path]
            )
        else:
            raw = client.chat(
                system=system, user=_text_user_prompt(entry, settings, asset, total)
            )
    except LLMError as exc:
        return PageParse(page_no=asset.page_no, source=mode, error=str(exc))

    markdown = normalize_markdown(raw)
    if is_probably_empty(markdown):
        markdown = ""  # 空白页/纯装饰页，不占用知识库

    if markdown:
        _save_cache(
            cache_path,
            markdown,
            {
                "doc_key": entry.doc_key,
                "page_no": asset.page_no,
                "mode": mode,
                "model": model_name,
                "prompts_version": settings.prompts_version,
            },
        )

    result = PageParse(page_no=asset.page_no, markdown=markdown, source=mode)
    annotate_page(result)
    return result


def annotate_page(result: PageParse) -> None:
    """补齐一页的统计字段（缓存命中与失败重建时都要用）。"""
    result.has_formula = has_formula(result.markdown)
    result.figure_count = count_figure_placeholders(result.markdown)
    result.figure_text_count = count_figure_text_marks(result.markdown)
    result.unclear_count = count_unclear_marks(result.markdown)


def parse_document(
    rendered: RenderedDoc,
    entry: ManifestEntry,
    settings: Settings,
    client: OpenAICompatClient,
    *,
    pages: Sequence[int] | None = None,
    use_cache: bool = True,
    progress: Callable[[PageParse, int, int], None] | None = None,
) -> tuple[ParseReport, str]:
    """解析一份资料，返回（解析报告, 合并后的整篇 Markdown）。

    合并 Markdown 里每个自然页前都有 <!-- page: N --> 标记，
    这是后面切块时「页码不丢」的唯一依据。
    """
    wanted = set(pages) if pages else None
    targets = [
        asset for asset in rendered.pages if wanted is None or asset.page_no in wanted
    ]
    total = rendered.page_count

    report = ParseReport()
    if not targets:
        return report, ""

    concurrency = max(1, int(settings.parsing.get("concurrency", 3)))
    retry = max(1, int(settings.parsing.get("retry", 3)))
    ordered: dict[int, PageParse] = {}
    done = 0

    def worker(asset: PageAsset) -> PageParse:
        attempt = 0
        result = _parse_one(
            entry, settings, asset, total, client, use_cache=use_cache
        )
        while result.error and attempt < retry:
            attempt += 1
            result = _parse_one(
                entry, settings, asset, total, client, use_cache=False
            )
        return result

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(worker, asset): asset for asset in targets}
        for future in as_completed(futures):
            asset = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # 兜底：worker 内部异常不能让整份资料失败
                result = PageParse(page_no=asset.page_no, error=f"未预期的错误：{exc}")
            ordered[asset.page_no] = result
            done += 1
            if progress:
                progress(result, done, len(targets))

    pages_dir = settings.paths.doc_pages(entry.doc_key)
    pages_dir.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    for page_no in sorted(ordered):
        page = ordered[page_no]
        report.pages.append(page)
        body = page.markdown
        if page.error:
            body = f"[本页解析失败：{page.error.splitlines()[0][:300]}]"
        (pages_dir / f"{entry.doc_key}_page_{page_no:04d}.md").write_text(
            body, encoding="utf-8"
        )
        parts.append(page_marker(page_no))
        if body:
            parts.append(body)

    # 失败详情另存一份完整版：页文件里的标记会被截断，排查时不方便
    errors_path = settings.paths.doc_work(entry.doc_key) / "parse_errors.json"
    errors_path.write_text(
        json.dumps(
            [
                {"page_no": page.page_no, "source": page.source, "error": page.error}
                for page in report.failures
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return report, "\n\n".join(parts).strip() + "\n"


def build_client(settings: Settings) -> OpenAICompatClient:
    from .llm import ModelConfig

    cfg = ModelConfig.from_dict("vision", settings.model_config("vision"))
    return OpenAICompatClient(
        cfg, retry=max(1, int(settings.parsing.get("retry", 3)))
    )


# ---------------------------------------------------------------------------
# 补漏校对
# ---------------------------------------------------------------------------

RECALL_EMPTY = "[无遗漏]"
RECALL_MARKER = "<!-- 以下内容由补漏校对补上 -->"

#: 做「这行是不是已经有了」比较时，先抹掉这些不承载语义的符号：
#: 空白、LaTeX 标记、Markdown 标记、全角半角标点、等号加号。
#: 抹掉它们的原因：同一段内容在已有结果和补漏结果里写法可能不同——
#: 逐字符比较会把它当成新内容重复加一遍（实测发生过）。
_NORMALIZE_RE = re.compile(r"[\s\$\{\}\\_\^~\*#`，,：:；;。.、()（）\[\]【】=\-—+&|]+")

#: 太短的片段不参与去重判断，否则「电压」这种两字词会把新内容误判成重复
_MIN_DEDUP_LENGTH = 2

#: 补漏模型偶尔会「跑偏」，把思考过程当成结果写出来
#: （实测出现过「等等，以上其实已在已有结果中。重新严格比对图片与已有转写：……」）。
#: 这些短语只可能出现在它的自我分析里，不可能出现在课件正文中，直接丢掉。
RECALL_META_PHRASES = (
    "已在已有结果",
    "已有结果中",
    "已有转写",
    "重新比对",
    "重新严格比对",
    "可视为已覆盖",
    "等等，",
)

#: 单独成行的这些符号没有信息量，是模型输出的空占位
_EMPTY_LINE_TOKENS = {"$$", "$", "---", "***", "```", "|"}


def _match_key(text: str) -> str:
    # 统一小写：同一段内容在「已有结果」和「补漏结果」里大小写不一致是常事
    # （实测 \text{cases} 里的 U_c 与 U_C、以及 &= 与 & 的对齐差异），
    # 逐字符比较会把它当成新内容重复加一遍。去重只看「有没有」，不看大小写。
    return _NORMALIZE_RE.sub("", text).translate(_VARIANT_CHARS).lower()


#: 只覆盖实际遇到过的繁简/异体字，不追求完整转换表：
#: 课件 PDF 的文字层偶尔是繁体字形（实测「暫态分量」「矩陣」），
#: 而模型读图写的是简体，逐字符比对会把同一句话当成新内容重复加一遍。
_VARIANT_CHARS = str.maketrans("暫陣與為說點電壓電號稱", "暂阵与为说点电压电号称")


def filter_new_lines(
    existing: str, added: str, *, page_no: int | None = None
) -> str:
    """丢掉「其实已经有了」的行，避免补漏把同一段内容写成两份。

    宁可漏掉少数真重复，也不要把新内容误删：所以只在整行归一化后
    **长度足够、且在已有结果里能整段找到** 时才判定为重复。
    """
    existing_key = _match_key(existing)
    kept: list[str] = []
    for raw in added.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in _EMPTY_LINE_TOKENS:
            continue
        key = _match_key(line)
        if len(key) < _MIN_DEDUP_LENGTH:
            continue
        # 孤立的纯数字：要么是页码，要么是编号，单独一行没有意义
        if key.isdigit() and (len(key) <= 2 or key == str(page_no)):
            continue
        if len(key) >= _MIN_DEDUP_LENGTH and key in existing_key:
            continue
        kept.append(line)
    return "\n".join(kept)


@dataclass
class RecallResult:
    page_no: int
    added: str = ""
    error: str | None = None
    #: 补漏结果异常膨胀时置位：这时**不要自动写入**，交给人看
    suspicious: bool = False
    reason: str = ""


#: 补漏结果超过「已有内容的这个比例」就认为跑偏了。
#: 实测教训：某页已有 1500 字，补漏返回 3000 字，里面大半是模型的自言自语
#: 和空占位；这种结果一旦自动追加进页文件，人工审核反而更费劲。
RECALL_MAX_RATIO = 0.6
RECALL_MAX_LINES = 40


def judge_recall(existing: str, added: str) -> tuple[bool, str]:
    """判断补漏结果是否异常膨胀。返回（是否可疑, 原因）。"""
    if not added.strip():
        return False, ""
    lines = [line for line in added.splitlines() if line.strip()]
    if len(lines) > RECALL_MAX_LINES:
        return True, f"补漏返回了 {len(lines)} 行，超过 {RECALL_MAX_LINES} 行的上限"
    existing_len = len(existing.strip())
    if existing_len and len(added) > existing_len * RECALL_MAX_RATIO:
        return (
            True,
            f"补漏内容 {len(added)} 字，已超过已有内容 {existing_len} 字的 "
            f"{RECALL_MAX_RATIO:.0%}",
        )
    return False, ""


def _clean_recall(text: str) -> str:
    """把「无遗漏」这类空回复识别出来，返回真正需要补充的内容。"""
    cleaned = text.strip()
    if not cleaned:
        return ""
    # 模型有时会在前后加一句「没有遗漏」再给结果，去掉纯声明行
    lines = [
        line
        for line in cleaned.splitlines()
        if line.strip()
        and "无遗漏" not in line
        and "没有遗漏" not in line
        and not any(phrase in line for phrase in RECALL_META_PHRASES)
    ]
    if not lines:
        return ""
    return "\n".join(lines).strip()


def recall_page(
    entry: ManifestEntry,
    settings: Settings,
    asset: PageAsset,
    existing_markdown: str,
    client: OpenAICompatClient,
    hints: Sequence[str] = (),
) -> RecallResult:
    """拿原图和已有结果再问一遍，专门找漏掉的文字。不写缓存（只对少数页跑）。

    ``hints`` 是自检程序怀疑漏掉的具体内容。带上它之后，任务从
    「开放地找找有什么漏的」变成「逐一确认这几处是否在图上」——
    实测开放式提问会漏（同一页两次都报「没有发现遗漏」，
    而原文文字层里明明有 1.5A、1A 这些图内数值）。
    """
    if asset.image_path is None:
        return RecallResult(page_no=asset.page_no, error="这一页没有图片，无法校对")

    if hints:
        listing = "\n".join(f"  - {item}" for item in hints if str(item).strip())
        hints_block = (
            "\n自检程序怀疑以下内容被漏掉了，请逐一确认它们是否出现在图上：\n"
            f"{listing}\n"
            "出现在图上的请原样抄出来；图上确实没有的不要写。\n"
        )
    else:
        hints_block = ""

    system = settings.prompt("page_recall", "system")
    user = settings.prompt("page_recall", "user").format(
        doc_title=entry.title,
        page_no=asset.page_no,
        existing=existing_markdown.strip()[:8000],
        hints_block=hints_block,
    )
    try:
        raw = client.chat_with_images(
            system=system, user=user, image_paths=[asset.image_path]
        )
    except LLMError as exc:
        return RecallResult(page_no=asset.page_no, error=str(exc))

    cleaned = _clean_recall(normalize_markdown(raw))
    filtered = filter_new_lines(
        existing_markdown, cleaned, page_no=asset.page_no
    )
    suspicious, reason = judge_recall(existing_markdown, filtered)
    return RecallResult(
        page_no=asset.page_no,
        added=filtered,
        suspicious=suspicious,
        reason=reason,
    )


def merge_recall(existing: str, added: str) -> str:
    """把补漏内容追加到页文件末尾。

    刻意**不**试图插回原来的位置：定位插回需要判断语义位置，容易插错；
    追加在末尾并加一行标记，既不会破坏已有内容，人工核对时也一眼能看到补了什么。

    重复校对同一页时复用已有标记，不叠加第二个——否则页文件里会出现
    好几个「以下内容由补漏校对补上」，人工看的时候容易混淆哪段是补的。
    """
    addition = added.strip()
    if not addition:
        return existing
    body = existing.rstrip()
    if RECALL_MARKER in body:
        return f"{body}\n{addition}\n"
    return f"{body}\n\n{RECALL_MARKER}\n\n{addition}\n"
