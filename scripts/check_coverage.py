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
from pipeline.cli import configure_stdio  # noqa: E402

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

    这里踩过三个坑，都会把「信息其实还在」误判成「数字丢失」：

    1. Unicode 上下标：模型把 U_6 写成 U₆，字符不同信息相同；
    2. 上标被 PDF 文字层压平：原文「ω＝10⁶ rad/s」在文字层里是「106」，
       模型正确写成 ``10^6``，字面上却匹配不上 106；
    3. LaTeX 里的 \\, \\; 等空白。

    所以统一把上下标字符还原成 ASCII，并把 ``^`` ``_`` 也一起去掉。
    """
    text = text.translate(SUBSCRIPT_DIGITS)
    return re.sub(r"[\s\,\;\~\!\$\{\}\\^_]+", "", text)


def _gram_size(run: str) -> int:
    # 短片段用 2-gram（否则切不出足够的块），长句子用 4-gram
    if len(run) <= 5:
        return 2
    if len(run) <= 12:
        return 3
    return 4


def gram_coverage(run: str, haystack: str) -> float:
    """这段原文有多大比例能在这页的解析结果里找到。

    返回 0 到 1。用 n-gram 而不是整串匹配，是为了容纳「改了一个字」
    和「繁体写成简体」这类无害差异。
    """
    if not run:
        return 1.0
    size = _gram_size(run)
    if len(run) < size:
        return 1.0 if run in haystack else 0.0
    grams = [run[index : index + size] for index in range(len(run) - size + 1)]
    hits = sum(1 for gram in grams if gram in haystack)
    return hits / len(grams)


def best_snippet(run: str, haystack: str, *, pad: int = 6) -> str:
    """在解析结果里找出与这段原文最像的一小段，供人工一眼判断。"""
    size = _gram_size(run)
    for index in range(len(run) - size + 1):
        gram = run[index : index + size]
        position = haystack.find(gram)
        if position >= 0:
            start = max(0, position - index - pad)
            return haystack[start : start + len(run) + 2 * pad]
    return ""


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

    # 用 n-gram 重合度代替精确子串匹配。原因：模型会顺手改掉一个字的笔误
    # （「缩短无电阻支路」写成「缩短短无电阻支路」）、把繁体字形写成简体
    # （暫态分量 → 暂态分量）。精确匹配会把这些判成「整段丢失」，
    # 20 页样本里一半的报警都是这类误报，人工审核时间全浪费在这上面。
    triaged: list[dict[str, object]] = []
    for run in source_runs:
        coverage = gram_coverage(run, parsed_cjk)
        if coverage >= 0.75:
            continue  # 只是润色或繁简差异，忽略
        triaged.append(
            {
                "run": run,
                "coverage": coverage,
                "hint": best_snippet(run, parsed_cjk),
                "level": "疑似缺失" if coverage < 0.3 else "待抽查",
            }
        )

    run_total = len(source_runs) or 1
    number_total = len(set(source_numbers)) or 1
    return {
        "numbers_total": len(set(source_numbers)),
        "numbers_missing": missing_numbers,
        "runs_total": len(source_runs),
        "suspicious": triaged,
        "number_coverage": 1 - len(missing_numbers) / number_total,
        # 只把「疑似缺失」计入覆盖率，「待抽查」不算扣分
        "run_coverage": 1 - sum(
            1 for item in triaged if item["level"] == "疑似缺失"
        )
        / run_total,
    }


def source_page_count(source_path: Path) -> int:
    """原始文件有多少页。PPTX 必须用 python-pptx 数，不能用 PyMuPDF。"""
    suffix = source_path.suffix.lower()
    if suffix == ".pptx":
        from pptx import Presentation

        return len(Presentation(str(source_path)).slides)
    if suffix == ".pdf":
        try:
            import pymupdf
        except ImportError:  # pragma: no cover
            import fitz as pymupdf  # type: ignore

        with pymupdf.open(source_path) as document:
            return document.page_count
    return 0


def source_page_text(source_path: Path, page_no: int) -> str:
    """取原始文件某一页的文字层。

    这里踩过一个坑：PyMuPDF **不会**因为格式不对而报错，它会「打开」PPTX 并
    谎报一个页数（实测 72 页的 PPTX 被报成 11 页），于是比对结果全是垃圾。
    所以必须按扩展名分派：PDF 走 PyMuPDF，PPTX 走 python-pptx。
    """
    suffix = source_path.suffix.lower()

    if suffix == ".pptx":
        from pptx import Presentation

        presentation = Presentation(str(source_path))
        slides = list(presentation.slides)
        if page_no > len(slides):
            return ""
        buffer: list[str] = []
        for shape in slides[page_no - 1].shapes:
            if getattr(shape, "has_text_frame", False):
                for paragraph in shape.text_frame.paragraphs:  # type: ignore[union-attr]
                    line = "".join(run.text for run in paragraph.runs).strip()
                    if line:
                        buffer.append(line)
        return "\n".join(buffer)

    if suffix == ".pdf":
        try:
            import pymupdf
        except ImportError:  # pragma: no cover
            import fitz as pymupdf  # type: ignore

        with pymupdf.open(source_path) as document:
            if page_no > document.page_count:
                return ""
            return document[page_no - 1].get_text("text")

    return ""


def build_review_section(
    doc_key: str,
    page_no: int,
    parsed_file: Path,
    source_text: str,
    parsed_text: str,
    suspicious: list[dict[str, object]],
    numbers_missing: list[str],
    number_coverage: float,
    run_coverage: float,
) -> str:
    """把一页的「机器判读 + 原图 + 原文 + 解析结果」拼成一段审核材料。

    做这一步的原因：审核时最费时间的不是判断，而是在图片查看器、
    原始 PDF、解析结果之间来回切换找同一页。拼在一起就只需往下滚。
    """
    image = parsed_file.with_suffix(".jpg")
    lines = [
        f"## {doc_key} · 第 {page_no} 页",
        "",
        f"**机器判读**：数字覆盖率 {number_coverage:.0%}，中文串覆盖率 {run_coverage:.0%}",
        "",
        "**原页图片**（和下面两段对照着看）：",
        "",
        f"    {image}",
        "",
    ]
    if numbers_missing:
        lines.append("**疑似丢失的数值**：")
        lines.append("")
        for value in numbers_missing:
            lines.append(f"- `{value}`")
        lines.append("")
    if suspicious:
        lines.append("**有差异的片段**：")
        lines.append("")
        for item in suspicious:
            if item["level"] == "疑似缺失":
                lines.append(f"- 【疑似缺失】{item['run']}")
            else:
                lines.append(
                    f"- 【待抽查】{item['run']}"
                    f" → 解析里像的是「{item['hint']}」"
                    f"（重合 {float(item['coverage']):.0%}）"
                )
        lines.append("")

    lines.extend(["<details><summary>原文文字层（PDF 提取，公式是乱序碎片）</summary>", ""])
    lines.extend(["```text", source_text.strip(), "```", "", "</details>", ""])
    lines.extend(["<details><summary>解析结果</summary>", ""])
    lines.extend(["```markdown", parsed_text.strip(), "```", "", "</details>", ""])
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


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
    parser.add_argument(
        "--fix-commands",
        action="store_true",
        help="只输出「把有问题的页重新解析一遍」的命令，不打印明细",
    )
    parser.add_argument(
        "--report",
        help="把需要人工确认的页写成一份审核工作单（图片路径 + 原文 + 解析结果）",
    )
    args = parser.parse_args(argv)

    configure_stdio()
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
    flagged: dict[str, list[int]] = {}
    review: list[str] = []

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

        if source_path.suffix.lower() not in (".pdf", ".pptx", ".docx"):
            print(f"{entry.doc_key}：{source_path.suffix} 暂不支持比对，跳过")
            continue

        print(f"\n=== {entry.doc_key} · {entry.title} ===")

        try:
            source_pages = source_page_count(source_path)
        except Exception as exc:
            print(f"  无法读取原始文件页数（{type(exc).__name__}: {exc}），跳过")
            continue
        if source_pages and source_pages < len(parsed_files):
            print(
                f"  !! 原始文件报告 {source_pages} 页，但已有 {len(parsed_files)} 页解析结果，"
                f"页数对不上，比对结果不可信，已跳过"
            )
            continue

        rows: list[str] = []
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
            suspicious: list[dict[str, object]] = list(result["suspicious"])
            numbers_missing: list[str] = list(result["numbers_missing"])

            if not suspicious and not numbers_missing:
                continue

            bad_pages += 1
            flagged.setdefault(entry.doc_key, []).append(page_no)
            review.append(
                build_review_section(
                    entry.doc_key,
                    page_no,
                    parsed_file,
                    source_text,
                    parsed_text,
                    suspicious,
                    numbers_missing,
                    number_coverage,
                    run_coverage,
                )
            )
            rows.append(
                f"  第 {page_no} 页"
                f"（数字 {number_coverage:.0%}，中文串 {run_coverage:.0%}）"
            )
            for value in numbers_missing[: args.show]:
                rows.append(f"      数字缺失  {value}")
            for item in suspicious[: args.show]:
                if item["level"] == "疑似缺失":
                    rows.append(f"      [{item['level']}] {item['run']}")
                else:
                    shown = item["hint"] or "（找不到相似内容）"
                    rows.append(
                        f"      [{item['level']}] {item['run']}"
                        f"   ->   解析里像的是：{shown}"
                        f"（重合 {float(item['coverage']):.0%}）"
                    )

        if args.fix_commands:
            continue
        if rows:
            print(f"{'页':>5}  需要看的地方")
            for row in rows:
                print(row)
        else:
            print("  全部页通过，无需人工确认。")

    if args.fix_commands:
        if not flagged:
            print("没有需要重解析的页。")
            return 0
        print("# 把下面几条依次执行，即可用当前提示词重解析所有有问题的页")
        print("# （只重跑这些页，不是整份资料，花费很小）")
        for doc_key, pages in flagged.items():
            spec = ",".join(str(value) for value in sorted(pages))
            print(f"python -m pipeline parse --doc {doc_key} --pages {spec}")
        for doc_key in flagged:
            print(f"python -m pipeline chunk --doc {doc_key}")
        return 1

    if args.report:
        header = [
            "# 解析审核工作单",
            "",
            f"共 {len(flagged)} 份资料、{bad_pages} 页需要人工确认"
            f"（全部 {checked_pages} 页）。",
            "",
            "## 怎么用这份清单",
            "",
            "1. 每页先读「机器判读」，再打开那一页的原图对照。",
            "2. 【待抽查】通常不用管：括号里已经给出解析里长得像的内容，",
            "   多数是模型把原文笔误改顺了、或繁体字形写成简体。",
            "3. 【疑似缺失】要重点确认：那段原文在解析结果里基本找不到。",
            "4. 确认真的漏了，就把这一页记进 `parse_errors` 表，",
            "   或者把整页重新解析一遍（`python -m pipeline parse --doc <编号> --pages <页>`）。",
            "",
            "---",
            "",
        ]
        report_path = Path(args.report)
        report_path.write_text(
            "\n".join(header + review), encoding="utf-8"
        )
        print(f"审核工作单已写入：{report_path.resolve()}")
        print(f"共 {bad_pages} 页待确认。用记事本或 VS Code 打开即可，图片路径可直接点击。")
        return 1 if bad_pages else 0

    print(f"\n共核对 {checked_pages} 页，其中 {bad_pages} 页有缺失需要人工确认。")
    print(
        "判读方式：「疑似缺失」= 这段原文在新结果里基本找不到，要重点看；"
        "「待抽查」= 有差异但重合度高（可能是模型改顺了笔误或繁简字形），"
        "对照括号里的提示一眼即可判断。"
    )
    return 1 if bad_pages else 0


if __name__ == "__main__":
    raise SystemExit(main())
