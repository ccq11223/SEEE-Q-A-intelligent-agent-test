"""解析完整性自检：把原始 PDF 的文字层和解析结果逐页比对，找「漏了什么」。

为什么需要它：人工抽查 5% 只能发现系统性错误，但「某一页漏掉一整段」这种
局部遗漏很容易被漏过去。这个脚本用两个可自动化的指标兜底：

* **数字覆盖率**：原文里的数值（1000、314、0.5…）在解析结果里还在不在。
  数值是电路课的关键信息，丢了就答错。
* **中文串覆盖率**：原文里连续 3 个以上的汉字，在解析结果里还找不找得到。

比对时只保留汉字、去掉所有 Markdown 与 LaTeX 标记，因此
「**电路模型**」和「电路模型」算一致，不会因为格式差异误报。

用法：
    python scripts/check_coverage.py --doc slide-06
    python scripts/check_coverage.py --doc slide-06 --pages 1-20
    python scripts/check_coverage.py --all --threshold 0.9
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.manifest import load_manifest  # noqa: E402
from pipeline.settings import Settings  # noqa: E402

NUMBER_RE = re.compile(r"[0-9]+(?:[.][0-9]+)?")
MIN_NUMBER_DIGITS = 2   # 单个数字（页码、序号）噪声太大，只查两位以上
MIN_CJK_RUN = 3         # 连续 3 个汉字以上才算一个可比对的片段

#: Unicode 上下标数字 → ASCII。模型常把 U_6 写成 U₆，直接比对会误报缺失。
SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")


def cjk_only(text: str) -> str:
    return "".join(c for c in text if "\u4e00" <= c <= "\u9fff")


def numbers_in(text: str) -> list[str]:
    found = []
    for match in NUMBER_RE.findall(text):
        digits = match.replace(".", "")
        if len(digits) >= MIN_NUMBER_DIGITS:
            found.append(match)
    return found


def cjk_runs(text: str, min_length: int = MIN_CJK_RUN) -> list[str]:
    runs: list[str] = []
    current = ""
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            current += char
        else:
            if len(current) >= min_length:
                runs.append(current)
            current = ""
    if len(current) >= min_length:
        runs.append(current)
    return runs


def compact_numbers(text: str) -> str:
    """把 LaTeX 里可能插入的空白与转义去掉，便于做子串比对。

    顺便把 Unicode 上下标数字还原成 ASCII，否则 U₆ 会被判成「6 丢失」。
    """
    text = text.translate(SUBSCRIPT_DIGITS)
    return re.sub(r"[\s\,\;\~\!\$\{\}\\]+", "", text)


def compare_page(source: str, parsed: str, *, page_no: int | None = None) -> dict[str, object]:
    source_numbers = numbers_in(source)
    parsed_numbers = compact_numbers(parsed)
    # 页脚上的页码会被模型按规则正确丢弃，不算缺失
    ignorable = {str(page_no)} if page_no else set()
    missing_numbers = [
        value
        for value in dict.fromkeys(source_numbers)
        if value not in parsed_numbers and value not in ignorable
    ]

    parsed_cjk = cjk_only(parsed)
    source_runs = list(dict.fromkeys(cjk_runs(source)))
    missing_runs = [run for run in source_runs if run not in parsed_cjk]

    number_total = len(set(source_numbers)) or 1
    run_total = len(source_runs) or 1
    return {
        "numbers_total": len(set(source_numbers)),
        "numbers_missing": missing_numbers,
        "runs_total": len(source_runs),
        "runs_missing": missing_runs,
        "number_coverage": 1 - len(missing_numbers) / number_total,
        "run_coverage": 1 - len(missing_runs) / run_total,
    }


def source_page_text(pdf_path: Path, page_no: int) -> str:
    try:
        import pymupdf
    except ImportError:  # pragma: no cover
        import fitz as pymupdf  # type: ignore

    with pymupdf.open(pdf_path) as document:
        if page_no > document.page_count:
            return ""
        return document[page_no - 1].get_text("text")


def parse_page_range(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    pages: set[int] = set()
    for part in spec.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            pages.update(range(int(start), int(end) + 1))
        else:
            pages.add(int(part))
    return pages or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="解析完整性自检")
    parser.add_argument("--root", help="项目根目录")
    parser.add_argument("--doc", action="append", help="资料编号，可重复")
    parser.add_argument("--all", action="store_true", help="检查全部已解析的资料")
    parser.add_argument("--pages", help="只查这些页，如 1-20")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="中文串覆盖率低于此值就报警，默认 0.9",
    )
    parser.add_argument("--show", type=int, default=6, help="每页最多列出几条缺失片段")
    args = parser.parse_args(argv)

    settings = Settings.load(args.root)
    entries = load_manifest(settings.root / "config" / "manifest.csv")
    wanted = parse_page_range(args.pages)

    if args.doc:
        targets = [entry for entry in entries if entry.doc_key in set(args.doc)]
    elif args.all:
        targets = entries
    else:
        parser.error("请用 --doc <编号> 指定资料，或用 --all")

    bad_pages = 0
    checked_pages = 0

    for entry in targets:
        pages_dir = settings.paths.doc_pages(entry.doc_key)
        parsed_files = sorted(pages_dir.glob(f"{entry.doc_key}_page_*.md"))
        if not parsed_files:
            print(f"{entry.doc_key}：还没有解析结果，跳过")
            continue

        source_path = entry.local_path(settings.paths)
        if not source_path.is_file():
            print(f"{entry.doc_key}：找不到原始文件 {source_path}，跳过")
            continue

        print(f"\n=== {entry.doc_key} · {entry.title} ===")
        print(f"{'页':>5} {'数字':>10} {'中文串':>10}  缺失片段")

        for parsed_file in parsed_files:
            page_no = int(parsed_file.stem.rsplit("_", 1)[1])
            if wanted and page_no not in wanted:
                continue

            parsed_text = parsed_file.read_text(encoding="utf-8")
            if parsed_text.startswith("[本页解析失败"):
                print(f"{page_no:>5} {'—':>10} {'—':>10}  解析失败，跳过")
                continue

            source_text = source_page_text(source_path, page_no)
            if not source_text.strip():
                # 纯图片页没有文字层，无从比对
                continue

            result = compare_page(source_text, parsed_text, page_no=page_no)
            checked_pages += 1
            number_coverage = float(result["number_coverage"])
            run_coverage = float(result["run_coverage"])
            missing = list(result["runs_missing"]) + [
                f"数字{value}" for value in result["numbers_missing"]
            ]

            flag = ""
            if run_coverage < args.threshold or result["numbers_missing"]:
                flag = "  <<< 需要人工确认"
                bad_pages += 1

            preview = "；".join(str(item) for item in missing[: args.show])
            print(
                f"{page_no:>5} {number_coverage:>9.0%} {run_coverage:>9.0%}"
                f"  {preview}{flag}"
            )

    print(f"\n共核对 {checked_pages} 页，其中 {bad_pages} 页有缺失需要人工确认。")
    print("覆盖率说明：数字缺失一条就算问题；中文串覆盖率低于阈值才报警。")
    return 1 if bad_pages else 0


if __name__ == "__main__":
    raise SystemExit(main())
