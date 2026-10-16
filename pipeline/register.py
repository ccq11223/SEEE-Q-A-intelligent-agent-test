"""步骤 1 的辅助：扫描原始资料目录，生成待确认的台账草稿行。

登记这件事的判断（类型、可信级别）只能由人来做，所以这个模块的定位是
**起草而不是定稿**：它把文件名、编号、默认级别填好，你只需要打开
`config/manifest.csv` 逐行确认。生成的行 `notes` 里会标明「自动登记」，
确认完把这句话删掉即可——一眼就能看出哪些行还没人看过。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .manifest import REQUIRED_COLUMNS, ManifestEntry

# 支持的原始资料扩展名
SUPPORTED_SUFFIXES = {
    ".pdf",
    ".pptx",
    ".ppt",
    ".docx",
    ".doc",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".md",
    ".txt",
}

# 文件名关键词 → 资料类型。顺序有意义：越靠前越优先。
KEYWORD_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("解析", "答案"), "past_exam_solution"),
    (("试题", "真题", "试卷", "考题", "往年题"), "past_exam"),
    (("考点", "大纲"), "exam_scope"),
    (("例题", "习题", "练习"), "example"),
    (("教材", "课本", "教科书"), "textbook"),
    (("串讲", "复习", "讲义", "课件"), "slide"),
)

# 扩展名兜底
SUFFIX_FALLBACK = {
    ".pptx": "slide",
    ".ppt": "slide",
    ".pdf": "slide",
    ".docx": "example",
    ".doc": "example",
}

# 类型 → 默认可信级别、默认优先级
TYPE_DEFAULTS = {
    "textbook": (1, 2),
    "slide": (1, 1),
    "exam_scope": (1, 0),
    "example": (2, 3),
    "past_exam": (2, 5),
    "past_exam_solution": (2, 6),
    "web": (3, 9),
}

AUTO_NOTE = "自动登记，待人工确认类型与级别"


@dataclass
class DraftRow:
    doc_key: str
    course_id: str
    doc_type: str
    trust_level: int
    filename: str
    title: str
    term: str
    priority: int
    notes: str
    source_path: Path | None = None

    def as_csv_row(self) -> dict[str, str]:
        return {
            "doc_key": self.doc_key,
            "course_id": self.course_id,
            "doc_type": self.doc_type,
            "trust_level": str(self.trust_level),
            "filename": self.filename,
            "title": self.title,
            "term": self.term,
            "priority": str(self.priority),
            "notes": self.notes,
        }


def guess_doc_type(filename: str) -> tuple[str, bool]:
    """猜资料类型。返回 (类型, 是否为猜测而非命中关键词)。"""
    stem = Path(filename).stem
    for keywords, doc_type in KEYWORD_RULES:
        if any(keyword in stem for keyword in keywords):
            return doc_type, False

    suffix = Path(filename).suffix.lower()
    if suffix in SUFFIX_FALLBACK:
        return SUFFIX_FALLBACK[suffix], True
    return "slide", True


def guess_trust_level(doc_type: str) -> int:
    return TYPE_DEFAULTS.get(doc_type, (2, 5))[0]


def guess_priority(doc_type: str) -> int:
    return TYPE_DEFAULTS.get(doc_type, (2, 5))[1]


def scan_files(directory: Path, *, recursive: bool = False) -> list[Path]:
    if not directory.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    found = [
        path
        for path in sorted(directory.glob(pattern))
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.startswith("~$")
    ]
    return found


def next_doc_key(doc_type: str, taken: set[str]) -> str:
    """生成稳定、纯 ASCII 的编号：slide-01、exam_scope-01……"""
    index = 1
    while True:
        candidate = f"{doc_type.replace('_', '-')}-{index:02d}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate
        index += 1


def build_drafts(
    files: list[Path],
    *,
    course_id: str,
    term: str = "",
    existing: list[ManifestEntry] | None = None,
    doc_type: str | None = None,
    trust_level: int | None = None,
) -> list[DraftRow]:
    """为还没登记的文件生成草稿行。已登记（同名文件或同编号）的自动跳过。"""
    existing = existing or []
    taken = {entry.doc_key for entry in existing}
    known_filenames = {entry.filename for entry in existing}
    known_names = {Path(entry.filename).name for entry in existing}

    drafts: list[DraftRow] = []
    for path in files:
        if path.name in known_filenames or path.name in known_names:
            continue

        guessed_type, is_guess = (
            (doc_type, False) if doc_type else guess_doc_type(path.name)
        )
        level = (
            trust_level if trust_level is not None else guess_trust_level(guessed_type)
        )
        notes = AUTO_NOTE if not doc_type and is_guess else ""

        drafts.append(
            DraftRow(
                doc_key=next_doc_key(guessed_type, taken),
                course_id=course_id,
                doc_type=guessed_type,
                trust_level=level,
                filename=path.name,
                title=path.stem,
                term=term,
                priority=guess_priority(guessed_type),
                notes=notes,
                source_path=path,
            )
        )
    return drafts


def _detect_bom(path: Path) -> bool:
    if not path.is_file():
        return False
    return path.read_bytes()[:3] == b"\xef\xbb\xbf"


def append_rows(path: Path, rows: list[DraftRow]) -> int:
    """把草稿行追加到台账末尾，保留原有内容与换行风格。

    只做追加，不重写已有行——台账是人工维护的资产，程序不该动它。
    """
    if not rows:
        return 0

    bom = _detect_bom(path)
    body = (
        path.read_text(encoding="utf-8-sig") if path.is_file() else ""
    )
    if body and not body.endswith("\n"):
        body += "\n"
    if not body:
        body = ",".join(REQUIRED_COLUMNS) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if bom:
            handle.write("\ufeff")
        handle.write(body)
        writer = csv.DictWriter(
            handle, fieldnames=list(REQUIRED_COLUMNS), lineterminator="\n"
        )
        for row in rows:
            writer.writerow(row.as_csv_row())

    return len(rows)
