"""步骤 4：人工抽查。

抽查的目的不是「确保 100% 正确」，而是尽早发现系统性错误
——比如某一类公式全部转错，那就该换工具或换参数，而不是逐页去修。

所以抽样策略是「按风险定向 + 少量随机」，并且把要看的图片路径和解析结果并排放好，
让检查的人能几秒钟看一页。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .ai_parser import PageParse, ParseReport
from .manifest import ManifestEntry
from .settings import Settings


@dataclass
class SampleItem:
    page_no: int
    reason: str
    image_path: Path | None
    markdown: str


def build_sample(
    entry: ManifestEntry,
    settings: Settings,
    report: ParseReport,
    *,
    ratio: float = 0.05,
    min_pages: int = 3,
    seed: int = 20261016,
) -> list[SampleItem]:
    pages = list(report.pages)
    if not pages:
        return []

    quota = max(min_pages, math.ceil(len(pages) * ratio))
    quota = min(quota, len(pages))

    image_by_page: dict[int, Path | None] = {}
    render_json = settings.paths.doc_work(entry.doc_key) / "render.json"
    if render_json.is_file():
        for row in json.loads(render_json.read_text(encoding="utf-8")):
            image_by_page[row["page_no"]] = (
                Path(row["image_path"]) if row.get("image_path") else None
            )

    priority: list[tuple[int, str]] = []
    for page in pages:
        if page.error:
            priority.append((page.page_no, "解析失败，必须重跑或人工补录"))
        elif page.unclear_count:
            priority.append((page.page_no, "有 [此处原图不清晰] 标记"))
        elif page.has_formula and page.figure_count:
            priority.append((page.page_no, "同时含公式与图片，最容易出错"))
        elif page.has_formula:
            priority.append((page.page_no, "含公式，重点核对 LaTeX"))
        elif page.figure_count:
            priority.append((page.page_no, "含图片占位，核对是否有关键信息被丢掉"))

    chosen: list[tuple[int, str]] = []
    for page_no, reason in priority:
        if len(chosen) >= quota:
            break
        chosen.append((page_no, reason))

    if len(chosen) < quota:
        rest = [page.page_no for page in pages if page.page_no not in {p for p, _ in chosen}]
        random.Random(seed).shuffle(rest)
        for page_no in rest[: quota - len(chosen)]:
            chosen.append((page_no, "随机抽查"))

    by_page = {page.page_no: page for page in pages}
    items: list[SampleItem] = []
    for page_no, reason in sorted(chosen):
        page: PageParse | None = by_page.get(page_no)
        items.append(
            SampleItem(
                page_no=page_no,
                reason=reason,
                image_path=image_by_page.get(page_no),
                markdown=page.markdown if page else "",
            )
        )
    return items


def write_sample_report(
    settings: Settings, entry: ManifestEntry, items: list[SampleItem], *, summary: str = ""
) -> Path:
    path = settings.paths.doc_work(entry.doc_key) / "sample_report.md"
    lines = [
        f"# 解析质量抽查 · {entry.title}",
        "",
        f"- 资料编号：`{entry.doc_key}`",
        f"- 可信级别：{entry.trust_label}",
        f"- 解析概览：{summary}",
        f"- 抽查页数：{len(items)}",
        "",
        "核对方法：把左列的图片和右列的解析结果对着看，重点看公式是否还原、"
        "图片位置的关键信息有没有被丢掉。发现问题就在下面的「问题登记」里加一行。",
        "",
        "## 抽查清单",
        "",
        "| 页码 | 抽查原因 | 图片 |",
        "| --- | --- | --- |",
    ]
    for item in items:
        image = f"`{item.image_path}`" if item.image_path else "（无图片）"
        lines.append(f"| {item.page_no} | {item.reason} | {image} |")

    lines.extend(["", "## 逐页对照", ""])
    for item in items:
        lines.append(f"### 第 {item.page_no} 页 · {item.reason}")
        lines.append("")
        lines.append("```markdown")
        lines.append(item.markdown or "（无解析内容）")
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "## 问题登记",
            "",
            "发现的问题请写进 `config/` 同级的问题文件或直接记进数据库 `parse_errors` 表，",
            "字段：`page_no, error_type(公式丢失/图片丢失/文字乱码/表格错乱/其他), detail`。",
            "如果同一类错误出现三次以上，说明是系统性问题——**换参数或换工具重跑整个文档**，不要逐页修。",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
