"""步骤 1：资料登记与分级。

台账是 `config/manifest.csv`，用 Excel 就能维护、用 git 就能看改动。
每一行 = 一份资料；`doc_key` 是它在整条流水线里的稳定标识。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .settings import Paths

REQUIRED_COLUMNS = (
    "doc_key",
    "course_id",
    "doc_type",
    "trust_level",
    "filename",
    "title",
    "term",
    "priority",
    "notes",
)

DOC_TYPES = {
    "textbook": "课本重点",
    "slide": "串讲 PPT",
    "exam_scope": "老师划定考点",
    "example": "典型例题",
    "past_exam": "往年题",
    "past_exam_solution": "往年题解析",
    "web": "网络资料",
}

TRUST_LABELS = {
    1: "一级 · 权威（可直接引用）",
    2: "二级 · 可用（引用时须标注）",
    3: "三级 · 待确认（不得直接引用）",
}


class ManifestError(ValueError):
    """台账写错时抛出，消息里必须指出是哪一行、哪一列。"""


@dataclass
class ManifestEntry:
    doc_key: str
    course_id: str
    doc_type: str
    trust_level: int
    filename: str
    title: str
    term: str = ""
    priority: int = 5
    notes: str = ""

    @property
    def doc_type_label(self) -> str:
        return DOC_TYPES.get(self.doc_type, self.doc_type)

    @property
    def trust_label(self) -> str:
        return TRUST_LABELS.get(self.trust_level, str(self.trust_level))

    @property
    def is_remote(self) -> bool:
        return self.filename.startswith(("http://", "https://"))

    @property
    def raw_ext(self) -> str:
        return Path(self.filename).suffix.lower()

    def local_path(self, paths: Paths) -> Path:
        """原始文件在本地 data/raw/ 下的位置。

        同时接受两种摆放方式：
          1. data/raw/<原文件名>            —— 直接丢进去最省事
          2. data/raw/<doc_key><扩展名>     —— 文件名混乱时手工改好名
        """
        direct = paths.raw_dir / self.filename
        if direct.is_file():
            return direct
        if self.raw_ext:
            by_key = paths.raw_dir / f"{self.doc_key}{self.raw_ext}"
            if by_key.is_file():
                return by_key
        return direct


def _clean(row: dict[str, str], column: str) -> str:
    return (row.get(column) or "").strip()


def load_manifest(path: Path) -> list[ManifestEntry]:
    """读取并校验台账。任何一处写错都直接报错，不做静默兜底。"""
    if not path.is_file():
        raise ManifestError(
            f"找不到资料台账：{path}\n"
            f"请把 config/manifest.example.csv 复制为 config/manifest.csv 后按行登记资料。"
        )

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ManifestError(
                f"台账缺少列：{', '.join(missing)}。"
                f"需要的列依次为：{', '.join(REQUIRED_COLUMNS)}"
            )

        entries: list[ManifestEntry] = []
        seen: dict[str, int] = {}

        for line_no, row in enumerate(reader, start=2):  # 第 1 行是表头
            if not _clean(row, "doc_key") and not _clean(row, "filename"):
                continue  # 空行

            doc_key = _clean(row, "doc_key")
            if not doc_key:
                raise ManifestError(f"台账第 {line_no} 行：doc_key 不能为空")
            if doc_key in seen:
                raise ManifestError(
                    f"台账第 {line_no} 行：doc_key {doc_key!r} 与第 {seen[doc_key]} 行重复"
                )
            seen[doc_key] = line_no

            doc_type = _clean(row, "doc_type")
            if doc_type not in DOC_TYPES:
                raise ManifestError(
                    f"台账第 {line_no} 行：doc_type={doc_type!r} 不认识。"
                    f"可选值：{', '.join(sorted(DOC_TYPES))}"
                )

            trust_raw = _clean(row, "trust_level")
            if trust_raw not in {"1", "2", "3"}:
                raise ManifestError(
                    f"台账第 {line_no} 行：trust_level 必须是 1 / 2 / 3，当前是 {trust_raw!r}"
                )
            trust_level = int(trust_raw)

            filename = _clean(row, "filename")
            if not filename:
                raise ManifestError(f"台账第 {line_no} 行：filename 不能为空")

            course_id = _clean(row, "course_id")
            if not course_id:
                raise ManifestError(f"台账第 {line_no} 行：course_id 不能为空")

            priority_raw = _clean(row, "priority") or "5"
            try:
                priority = int(priority_raw)
            except ValueError as exc:
                raise ManifestError(
                    f"台账第 {line_no} 行：priority 必须是整数，当前是 {priority_raw!r}"
                ) from exc

            entries.append(
                ManifestEntry(
                    doc_key=doc_key,
                    course_id=course_id,
                    doc_type=doc_type,
                    trust_level=trust_level,
                    filename=filename,
                    title=_clean(row, "title") or doc_key,
                    term=_clean(row, "term"),
                    priority=priority,
                    notes=_clean(row, "notes"),
                )
            )

    return entries


def select(
    entries: list[ManifestEntry],
    *,
    doc_keys: list[str] | None = None,
    course_id: str | None = None,
    max_trust: int | None = None,
) -> list[ManifestEntry]:
    """按 doc_key / 课程 / 可信级别筛选，并按 priority 升序返回。"""
    chosen = entries
    if doc_keys:
        wanted = set(doc_keys)
        known = {entry.doc_key for entry in entries}
        unknown = sorted(wanted - known)
        if unknown:
            raise ManifestError(
                f"台账里没有这些 doc_key：{', '.join(unknown)}。"
                f"已有：{', '.join(sorted(known)) or '（空）'}"
            )
        chosen = [entry for entry in chosen if entry.doc_key in wanted]
    if course_id:
        chosen = [entry for entry in chosen if entry.course_id == course_id]
    if max_trust is not None:
        chosen = [entry for entry in chosen if entry.trust_level <= max_trust]
    return sorted(chosen, key=lambda entry: (entry.priority, entry.doc_key))
