"""把七个步骤串起来，并让每一步的产物都落盘、可单独重跑。

设计要点：阶段之间通过 `data/work/<doc_key>/` 下的文件传递，
因此「解析到一半断了」也能接着跑，任何时候都可以只重跑其中一步。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .ai_parser import parse_document
from .chunker import Chunk, ChunkConfig, chunk_document
from .convert import PageAsset, RenderedDoc, render_document
from .embedder import embed_chunks, load_cached_embeddings, save_embeddings
from .llm import OpenAICompatClient
from .manifest import ManifestEntry
from .settings import Settings
from .store import ChunkRecord, DocumentRecord, Store
from .textutils import PAGE_MARKER_RE, page_marker


@dataclass
class RunOptions:
    pages: set[int] | None = None
    use_cache: bool = True
    allow_partial: bool = False
    skip_embed: bool = False
    dry_run: bool = False


@dataclass
class RunSummary:
    doc_key: str
    title: str
    converter: str = ""
    page_count: int = 0
    parsed_pages: int = 0
    failed_pages: int = 0
    chunk_count: int = 0
    embedded: int = 0
    loaded: int = 0
    skipped_stages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failure_details: list[tuple[int, str]] = field(default_factory=list)
    sample_report: Path | None = None


# ---------------------------------------------------------------------------
# 步骤 2：渲染
# ---------------------------------------------------------------------------

def save_render(settings: Settings, rendered: RenderedDoc) -> Path:
    path = settings.paths.doc_work(rendered.doc_key) / "render.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "page_no": page.page_no,
            "image_path": str(page.image_path) if page.image_path else None,
            "native_text": page.native_text,
            "has_embedded_image": page.has_embedded_image,
        }
        for page in rendered.pages
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def load_render(settings: Settings, doc_key: str) -> RenderedDoc | None:
    path = settings.paths.doc_work(doc_key) / "render.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    pages = [
        PageAsset(
            page_no=row["page_no"],
            image_path=Path(row["image_path"]) if row.get("image_path") else None,
            native_text=row.get("native_text", ""),
            has_embedded_image=bool(row.get("has_embedded_image")),
        )
        for row in payload
    ]
    return RenderedDoc(doc_key=doc_key, source_path=Path(""), converter="cached", pages=pages)


# ---------------------------------------------------------------------------
# 步骤 5 的输入：把逐页解析结果拼回整篇 Markdown
# ---------------------------------------------------------------------------

def build_combined_markdown(
    settings: Settings, doc_key: str, *, page_count: int | None = None
) -> tuple[str, list[int]]:
    pages_dir = settings.paths.doc_pages(doc_key)
    total = page_count
    if total is None:
        render = load_render(settings, doc_key)
        total = render.page_count if render else 0

    parts: list[str] = []
    missing: list[int] = []
    for page_no in range(1, total + 1):
        path = pages_dir / f"{doc_key}_page_{page_no:04d}.md"
        if not path.is_file():
            missing.append(page_no)
            continue
        parts.append(page_marker(page_no))
        body = path.read_text(encoding="utf-8").strip()
        if body:
            parts.append(body)

    return "\n\n".join(parts).strip() + ("\n" if parts else ""), missing


def write_combined_markdown(settings: Settings, entry: ManifestEntry, page_count: int) -> tuple[Path, list[int]]:
    markdown, missing = build_combined_markdown(settings, entry.doc_key, page_count=page_count)
    path = settings.paths.doc_markdown(entry.doc_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path, missing


# ---------------------------------------------------------------------------
# 步骤 5：切块
# ---------------------------------------------------------------------------

def run_chunking(
    settings: Settings, entry: ManifestEntry, *, allow_missing: bool = False
) -> list[Chunk]:
    """切块。

    **每次都从页文件重建整篇 Markdown**，而不是直接读 doc.md。
    原因：页文件 `pages/<编号>_page_XXXX.md` 才是权威来源——补漏校对和人工
    修补改的都是它，`doc.md` 只是派生产物。曾经踩过：手工改了页文件后跑
    `chunk`，它读的还是旧的 doc.md，于是切出来的是过时内容（表现为云端
    知识库里还留着早已删掉的段落）。
    """
    rendered = load_render(settings, entry.doc_key)
    if rendered is None:
        raise FileNotFoundError(
            f"找不到 {entry.doc_key} 的渲染结果，请先执行 render 与 parse。"
        )
    markdown, missing = write_combined_markdown(settings, entry, rendered.page_count)
    if missing and not allow_missing:
        raise FileNotFoundError(
            f"{entry.doc_key} 还有 {len(missing)} 页没有解析结果"
            f"（前几页：{missing[:5]}），请先补齐再切块。"
        )
    markdown = markdown.read_text(encoding="utf-8")
    chunks = chunk_document(
        markdown,
        doc_key=entry.doc_key,
        config=ChunkConfig.from_dict(settings.chunking),
    )
    out = settings.paths.doc_chunks(entry.doc_key)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks) + "\n",
        encoding="utf-8",
    )
    return chunks


def load_chunks(settings: Settings, doc_key: str) -> list[Chunk]:
    path = settings.paths.doc_chunks(doc_key)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}，请先执行 chunk（切块）。")
    chunks: list[Chunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def count_page_markers(markdown: str) -> int:
    return sum(1 for line in markdown.splitlines() if PAGE_MARKER_RE.match(line))


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------

def run_document(
    entry: ManifestEntry,
    settings: Settings,
    store: Store,
    *,
    vision_client: OpenAICompatClient,
    embed_client: OpenAICompatClient,
    options: RunOptions,
    log: Callable[[str], None] = print,
) -> RunSummary:
    summary = RunSummary(doc_key=entry.doc_key, title=entry.title)

    # ---- 步骤 1+2：登记与渲染 ----
    rendered, warnings = render_document(entry, settings)
    summary.warnings.extend(warnings)
    summary.converter = rendered.converter
    summary.page_count = rendered.page_count
    save_render(settings, rendered)
    log(f"  [2/6] 渲染完成：{rendered.page_count} 页（{rendered.converter}）")

    if not options.dry_run:
        store.upsert_document(
            DocumentRecord.from_entry(
                entry, status="converted", page_count=rendered.page_count
            )
        )

    # ---- 步骤 3：AI 结构化解析 ----
    report, _ = parse_document(
        rendered,
        entry,
        settings,
        vision_client,
        pages=sorted(options.pages) if options.pages else None,
        use_cache=options.use_cache,
        progress=lambda page, done, total: (
            log(
                f"  [3/6] 解析 {done}/{total} 页"
                + (f"｜第 {page.page_no} 页失败" if page.error else "")
            )
            if (done % 5 == 0 or done == total or page.error)
            else None
        ),
    )
    summary.parsed_pages = len(report.pages)
    summary.failed_pages = len(report.failures)
    summary.failure_details = [
        (page.page_no, page.error or "") for page in report.failures
    ]
    log(f"  [3/6] 解析完成：{report.summary()}")

    _, missing = write_combined_markdown(settings, entry, rendered.page_count)
    if missing:
        summary.warnings.append(
            f"还有 {len(missing)} 页没有解析结果：{missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )

    if not options.dry_run:
        store.upsert_document(
            DocumentRecord.from_entry(
                entry,
                status="parsed",
                page_count=rendered.page_count,
                parse_error_count=summary.failed_pages,
            )
        )

    if missing and not options.allow_partial:
        summary.skipped_stages.append("chunk/embed/load")
        summary.warnings.append(
            "存在未解析页，已跳过切块与入库，避免用残缺内容覆盖已有知识库。"
            "确认要按现有页入库，请加 --allow-partial。"
        )
        return summary

    # ---- 质量闸门：解析大面积失败时不许切块入库 ----
    # 这一条拦的是「密钥失效 / 没有额度」这类原因：那种情况下每一页都会失败，
    # 而页文件其实是写出来了的（内容是失败标记），所以上面的「缺页」检查拦不住。
    max_ratio = float(
        (settings.pipeline.get("quality_gate") or {}).get("max_failure_ratio", 0.1)
    )
    if summary.parsed_pages and summary.failed_pages:
        ratio = summary.failed_pages / summary.parsed_pages
        if ratio > max_ratio and not options.allow_partial:
            summary.skipped_stages.append("chunk/embed/load")
            summary.warnings.append(
                f"已解析的 {summary.parsed_pages} 页里有 {summary.failed_pages} 页失败"
                f"（{ratio:.0%}，超过质量闸门 {max_ratio:.0%}），已跳过切块与入库。"
                f"先按上面的失败原因修好，再重跑；确实要带着失败页入库才加 --allow-partial。"
            )
            return summary

    # ---- 步骤 5：切块 ----
    chunks = run_chunking(
        settings, entry, allow_missing=options.allow_partial
    )
    summary.chunk_count = len(chunks)
    log(f"  [5/6] 切块完成：{len(chunks)} 块")
    if not options.dry_run:
        store.upsert_document(
            DocumentRecord.from_entry(
                entry,
                status="chunked",
                page_count=rendered.page_count,
                chunk_count=len(chunks),
                parse_error_count=summary.failed_pages,
            )
        )

    if options.skip_embed:
        summary.skipped_stages.append("embed/load")
        return summary

    # ---- 步骤 6：向量化 ----
    emb_path = settings.paths.doc_embeddings(entry.doc_key)
    vectors, note = load_cached_embeddings(emb_path, chunks)
    if vectors is not None:
        log(f"  [6/6] {note}，跳过向量模型调用")
    else:
        vectors = embed_chunks(
            chunks,
            embed_client,
            batch_size=int(settings.embedding.get("batch_size", 32)),
            expected_dim=int(settings.embedding.get("dim", 0)) or None,
            progress=lambda done, total: (
                log(f"  [6/6] 向量化 {done}/{total} 块")
                if (done % 40 == 0 or done == total)
                else None
            ),
        )
    # 复用或新算都按带指纹的新格式重存，让下次比对能逐条核对内容
    save_embeddings(emb_path, chunks, vectors)
    summary.embedded = len(vectors)
    log(f"  [6/6] 向量化完成：{len(vectors)} 条")

    if options.dry_run:
        summary.skipped_stages.append("load")
        summary.warnings.append("dry-run：已跳过写入知识库。")
        return summary

    # ---- 入库 ----
    records = [
        ChunkRecord.from_chunk(chunk, trust_level=entry.trust_level, embedding=vector)
        for chunk, vector in zip(chunks, vectors)
    ]
    document = DocumentRecord.from_entry(
        entry,
        status="loaded",
        page_count=rendered.page_count,
        chunk_count=len(records),
        parse_error_count=summary.failed_pages,
    )
    summary.loaded = store.replace_chunks(document, records)
    log(f"  [6/6] 入库完成：{summary.loaded} 块 → {store.backend}")

    return summary
