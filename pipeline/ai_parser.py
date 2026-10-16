"""步骤 3：结构化解析 —— 用视觉大模型把每一页课件转成结构化 Markdown。

这是整条链路质量的天花板（方案 §7 阶段 0 的结论），所以：
  - 默认逐页送图（vision），公式与电路图靠模型看，不靠文本层猜；
  - 每页结果落盘缓存，重跑只补失败页，不重复花钱；
  - 单页失败不中断整份资料，失败页单独记账，最后一起报给你。
"""

from __future__ import annotations

import hashlib
import json
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
        unclear = sum(page.unclear_count for page in self.pages)
        return (
            f"共 {total} 页｜缓存命中 {self.cached_count} 页｜含公式 {formula} 页｜"
            f"图片占位 {figures} 个｜不清晰标记 {unclear} 处｜失败 {len(self.failures)} 页"
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
            body = f"[本页解析失败：{page.error.splitlines()[0][:120]}]"
        (pages_dir / f"{entry.doc_key}_page_{page_no:04d}.md").write_text(
            body, encoding="utf-8"
        )
        parts.append(page_marker(page_no))
        if body:
            parts.append(body)

    return report, "\n\n".join(parts).strip() + "\n"


def build_client(settings: Settings) -> OpenAICompatClient:
    from .llm import ModelConfig

    cfg = ModelConfig.from_dict("vision", settings.model_config("vision"))
    return OpenAICompatClient(
        cfg, retry=max(1, int(settings.parsing.get("retry", 3)))
    )
