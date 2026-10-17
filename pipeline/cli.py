"""命令行入口：``python -m pipeline <命令>``。"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import unicodedata
from pathlib import Path

from .ai_parser import build_client, parse_document
from .answer import answer_question
from .convert import ConvertError, render_document
from .dsn import inspect_dsn
from .env import get_env
from .llm import LLMError, ModelConfig, OpenAICompatClient
from .manifest import ManifestEntry, ManifestError, load_manifest, select
from .retrieval import retrieve
from .register import (
    AUTO_NOTE,
    SUPPORTED_SUFFIXES,
    append_rows,
    build_drafts,
    scan_files,
)
from .runner import (
    RunOptions,
    load_chunks,
    run_chunking,
    run_document,
    save_render,
    write_combined_markdown,
)
from .sampler import build_sample, write_sample_report
from .settings import ConfigError, Settings
from .store import open_store


def info(message: str) -> None:
    print(message, flush=True)


def configure_stdio() -> None:
    """统一按 UTF-8 输出，并把装不下的字符替换掉，绝不让打印本身把程序搞崩。

    这里前后踩了三次坑，记录一下，免得再改回去：

    1. 不做任何处理：中文 Windows 上 Python 默认用 GBK 输出 `✓`，直接抛
       UnicodeEncodeError，程序崩在打印上。
    2. 一律强制 UTF-8：终端（Windows Terminal、VS Code、Codex 的终端）本来就按
       UTF-8 解码，所以显示正常；但 `python -m pipeline ... | Select-String`
       这种管道是 PowerShell 5.1 用控制台代码页（GBK）解码的，会乱码。
    3. 改为跟随控制台代码页：管道正常了，终端反而全乱——因为终端要的是 UTF-8。

    结论：**终端是主要的消费方，按 UTF-8 输出**；
    管道那个问题用两条办法解决——PowerShell 侧把 [Console]::OutputEncoding 也设成
    UTF-8（见 scripts/setup-console.ps1），或者干脆用命令自带的 --grep 不走管道。
    想强制指定编码就设 PYTHONIOENCODING 环境变量，这里不覆盖。
    """
    if os.environ.get("PYTHONIOENCODING"):
        return

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - 环境不支持时忽略
            pass


def marks() -> tuple[str, str]:
    """返回（通过标记, 未通过标记）。

    终端编码装不下 ✓/✗ 时自动退化成 ASCII，避免它们变成 `?` 或直接报错。
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        for symbol in ("✓", "✗"):
            symbol.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return "OK", "!!"
    return "✓", "✗"


def warn(message: str) -> None:
    print(f"[警告] {message}", flush=True)


def fail(message: str) -> None:
    print(f"[错误] {message}", file=sys.stderr, flush=True)


def display_width(text: str) -> int:
    """按终端显示宽度算长度：中文、全角标点占两格。"""
    return sum(
        2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text
    )


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """打印对齐的表格。中英混排时列宽按显示宽度算，避免中文列错位。"""
    table = [[str(cell) for cell in row] for row in rows]
    widths = [
        max([display_width(header)] + [display_width(row[index]) for row in table])
        for index, header in enumerate(headers)
    ]

    def render(cells: list[str]) -> str:
        parts = [
            cell + " " * (widths[index] - display_width(cell))
            for index, cell in enumerate(cells)
        ]
        return "  ".join(parts).rstrip()

    info(render(headers))
    info(render(["-" * width for width in widths]))
    for row in table:
        info(render(row))


def summarize_failures(details: list[tuple[int, str]], *, limit: int = 3) -> None:
    """把失败原因按内容归类后打印。

    同一个原因（配额、密钥、网络）往往会让整份资料的每一页都失败，
    逐页重复打印同一个报错反而看不出重点。
    """
    if not details:
        return

    groups: dict[str, list[int]] = {}
    for page_no, error in details:
        first_line = (error.splitlines() or [""])[0].strip() or "(没有错误信息)"
        groups.setdefault(first_line, []).append(page_no)

    warn(f"失败 {len(details)} 页，共 {len(groups)} 种原因：")
    for index, (message, pages) in enumerate(groups.items()):
        if index >= limit:
            info(f"  …… 另有 {len(groups) - limit} 种原因，完整内容见 parse_errors.json")
            break
        shown = pages[:6]
        suffix = "…" if len(pages) > len(shown) else ""
        info(f"  第 {shown}{suffix} 页：{message}")


def parse_pages(spec: str | None) -> set[int] | None:
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


def parse_hints(spec: str | None) -> dict[int, list[str]]:
    """解析 ``--hints``，形如 ``26:1.5A|1A|5Ω,13:独立源``。

    把自检程序怀疑漏掉的具体内容告诉补漏模型，任务就从「开放地找漏」
    变成「逐一确认这几处」——后者实测明显更可靠。
    """
    if not spec:
        return {}
    mapping: dict[int, list[str]] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        page_part, _, items_part = chunk.partition(":")
        try:
            page_no = int(page_part.strip())
        except ValueError:
            continue
        items = [
            item.strip()
            for item in items_part.replace("、", "|").split("|")
            if item.strip()
        ]
        if items:
            mapping[page_no] = items
    return mapping


def resolve_entries(settings: Settings, args: argparse.Namespace) -> list[ManifestEntry]:
    manifest_path = settings.root / "config" / "manifest.csv"
    entries = load_manifest(manifest_path)
    if not entries:
        raise ManifestError(
            f"{manifest_path} 里还没有登记任何资料。"
            f"请参考 config/manifest.example.csv 填好资料台账。"
        )

    course_id = getattr(args, "course", None) or settings.default_course_id
    settings.require_course(course_id)

    if not args.doc and not args.all:
        raise ManifestError("请用 --doc <资料编号> 指定一份资料，或用 --all 处理该课程全部资料。")

    return select(
        entries,
        doc_keys=args.doc or None,
        course_id=course_id,
    )


def open_clients(settings: Settings, *, need_vision: bool = True, need_embed: bool = True):
    vision = None
    embed = None
    if need_vision:
        vision = build_client(settings)
    if need_embed:
        embed = OpenAICompatClient(
            ModelConfig.from_dict("embedding", settings.model_config("embedding"))
        )
    return vision, embed


def embed_client(settings: Settings) -> OpenAICompatClient:
    return OpenAICompatClient(
        ModelConfig.from_dict("embedding", settings.model_config("embedding"))
    )


def rerank_client(settings: Settings) -> OpenAICompatClient | None:
    """重排客户端。config/models.yaml 里没有 rerank 段就返回 None（退回纯向量）。"""
    try:
        cfg = settings.model_config("rerank")
    except ConfigError:
        return None
    return OpenAICompatClient(ModelConfig.from_dict("rerank", cfg))


# ---------------------------------------------------------------------------
# 各子命令
# ---------------------------------------------------------------------------

def cmd_doctor(settings: Settings, args: argparse.Namespace) -> int:
    problems: list[str] = []
    ok, no = marks()
    info("=== 环境自检 ===")
    info(f"Python：{sys.version.split()[0]}（{sys.executable}）")
    info(f"项目根目录：{settings.root}")

    info("\n[配置]")
    for name in ("pipeline.yaml", "models.yaml", "retrieval.yaml", "prompts.yaml"):
        path = settings.root / "config" / name
        info(f"  {ok if path.is_file() else no} config/{name}")
        if not path.is_file():
            problems.append(f"缺少 config/{name}")

    env_path = settings.root / ".env"
    info(f"  {ok if env_path.is_file() else no} .env（已加载 {settings.env_loaded} 项）")
    if not env_path.is_file():
        problems.append("还没有 .env，请从 .env.example 复制并填写密钥")

    info("\n[依赖]")
    for module in ("httpx", "yaml", "pymupdf", "PIL", "pptx"):
        try:
            importlib.import_module(module)
            info(f"  {ok} {module}")
        except ImportError:
            info(f"  {no} {module}")
            problems.append(f"缺少依赖 {module}，请执行 pip install -r requirements.txt")

    backend = str(settings.storage.get("backend", "local"))
    if backend == "postgres":
        try:
            importlib.import_module("psycopg")
            info(f"  {ok} psycopg")
        except ImportError:
            info(f"  {no} psycopg")
            problems.append("storage.backend=postgres 但缺少 psycopg，请执行 pip install 'psycopg[binary]'")

    info("\n[本机 Office 能力（PPT/Word → 图片）]")
    try:
        import win32com.client  # noqa: F401

        info(f"  {ok} pywin32（可调用本机 PowerPoint / Word 导出）")
    except ImportError:
        info(f"  {no} pywin32，PPTX 将无法导出为图片（PDF 不受影响）")
        problems.append("建议 pip install pywin32，或安装 LibreOffice 作为替代")

    info("\n[密钥]")
    for role in ("vision", "embedding", "generate"):
        try:
            cfg = settings.model_config(role)
        except ConfigError as exc:
            info(f"  {no} {role}：{exc}")
            problems.append(str(exc))
            continue
        key_env = str(cfg["api_key_env"])
        has_key = bool(get_env(key_env))
        info(f"  {ok if has_key else no} {role}：{cfg['name']}  ← {key_env}")
        if not has_key:
            problems.append(f"{role} 缺少 {key_env}")

    info("\n[知识库]")
    info(f"  存储后端：{backend}")
    if backend == "local":
        info(f"  本地输出目录：{settings.paths.out_dir}")
    else:
        dsn = get_env("SUPABASE_DB_URL")
        info(f"  {ok if dsn else no} SUPABASE_DB_URL")
        if not dsn:
            problems.append("storage.backend=postgres 但缺少 SUPABASE_DB_URL")
        else:
            report = inspect_dsn(dsn)
            if report.ok:
                info(
                    f"      解析结果：主机 {report.host}｜库 {report.dbname}"
                    f"｜用户 {report.user}"
                )
            else:
                info("      连接串体检未通过：")
                for problem in report.problems:
                    info(f"        · {problem}")
                problems.extend(report.problems)
            for hint in report.hints:
                info(f"      提示：{hint}")

    info("\n[资料台账]")
    manifest_path = settings.root / "config" / "manifest.csv"
    try:
        entries = load_manifest(manifest_path)
        info(f"  已登记 {len(entries)} 份资料")
        for entry in entries:
            raw = entry.local_path(settings.paths)
            mark = ok if raw.is_file() else ("—" if entry.is_remote else no)
            info(f"    {mark} {entry.doc_key}  {entry.trust_label}  {entry.title}")
    except ManifestError as exc:
        info(f"  {no} {exc}")
        problems.append("资料台账有问题（见上）")

    info("\n=== 结论 ===")
    if problems:
        for problem in problems:
            info(f"  · {problem}")
        return 1
    info("  全部通过，可以开始建库。")
    return 0


def cmd_manifest(settings: Settings, args: argparse.Namespace) -> int:
    manifest_path = settings.root / "config" / "manifest.csv"
    entries = load_manifest(manifest_path)
    if not entries:
        warn(f"{manifest_path} 里还没有登记任何资料。")
        return 0
    print_table(
        ["编号", "类型", "级别", "优先级", "文件状态", "标题"],
        [
            [
                entry.doc_key,
                entry.doc_type_label,
                str(entry.trust_level),
                str(entry.priority),
                "网络资料"
                if entry.is_remote
                else ("已就位" if entry.local_path(settings.paths).is_file() else "缺文件"),
                entry.title,
            ]
            for entry in entries
        ],
    )
    return 0


def cmd_keys(settings: Settings, args: argparse.Namespace) -> int:
    """密钥体检：确认每个密钥有效、位置正确、且能访问配置里的模型。

    只调用各家的 /models 列表接口，免费且不消耗额度。
    """
    from .keycheck import probe_key

    roles = list(args.role) if args.role else ["vision", "embedding", "generate"]
    probes = []
    for role in roles:
        try:
            cfg = settings.model_config(role)
        except ConfigError as exc:
            fail(str(exc))
            return 2
        key_env = str(cfg["api_key_env"])
        probes.append(
            probe_key(
                role=role,
                key_env=key_env,
                api_key=get_env(key_env),
                base_url=str(cfg["base_url"]),
                model=str(cfg["name"]),
                timeout_s=float(args.timeout),
            )
        )

    labels = {
        "ok": "可用",
        "missing": "未配置",
        "unauthorized": "鉴权失败",
        "not_found": "地址不对",
        "network": "网络不通",
        "error": "调用出错",
        "unknown": "未知",
    }
    print_table(
        ["用途", "密钥变量", "结果", "说明"],
        [
            [probe.role, probe.key_env, labels.get(probe.status, probe.status), probe.detail]
            for probe in probes
        ],
    )

    for probe in probes:
        if probe.notes:
            info("")
            info(f"[{probe.role}] {probe.key_env}")
            for note in probe.notes:
                info(f"  · {note}")

    if args.list:
        keyword = (args.grep or "").lower()
        for probe in probes:
            if not probe.all_models:
                continue
            names = [item for item in probe.all_models if keyword in item.lower()]
            if keyword and not names:
                info("")
                info(
                    f"[{probe.role}] {probe.key_env}："
                    f"{len(probe.all_models)} 个可用模型里没有名字含 {args.grep!r} 的。"
                )
                continue
            info("")
            title = f"[{probe.role}] {probe.key_env} 的可用模型"
            if keyword:
                title += (
                    f"（含 {args.grep!r} 的，共 {len(names)} / {len(probe.all_models)} 个）"
                )
            else:
                title += f"（共 {len(probe.all_models)} 个）"
            info(title + "：")
            for model_name in names:
                info(f"    {model_name}")

    info("")
    broken = [probe for probe in probes if probe.status != "ok"]
    wrong_model = [probe for probe in probes if probe.model_missing]

    if not broken and not wrong_model:
        info("密钥有效、模型名也都对得上，可以开始建库。")
        return 0

    if not broken and wrong_model:
        info("密钥全部有效，问题只在模型名——不解决的话，建库会在解析阶段整份失败：")
        for probe in wrong_model:
            target = (
                "、".join(probe.suggestions)
                if probe.suggestions
                else "（列表里没有相近的模型）"
            )
            info(f"  · {probe.role}：不用 {probe.model!r}，改用 {target}")
        info("改 config/models.yaml 里对应的 name，改完再跑一次本命令确认。")
        return 1

    info("上面标了「未配置 / 鉴权失败 / 网络不通」的都要先解决，再往下走。")
    return 1


def cmd_register(settings: Settings, args: argparse.Namespace) -> int:
    """扫描 data/raw/，为未登记的文件生成台账草稿行。"""
    from .manifest import DOC_TYPES, TRUST_LABELS

    raw_dir = Path(args.dir) if args.dir else settings.paths.raw_dir
    if not raw_dir.is_absolute():
        raw_dir = (settings.root / raw_dir).resolve()

    course_id = args.course or settings.default_course_id
    settings.require_course(course_id)

    if args.doc_type and args.doc_type not in DOC_TYPES:
        fail(
            f"--doc-type 不认识 {args.doc_type!r}。可选：{', '.join(sorted(DOC_TYPES))}"
        )
        return 2
    if args.trust_level is not None and args.trust_level not in (1, 2, 3):
        fail("--trust-level 只能是 1 / 2 / 3")
        return 2

    files = scan_files(raw_dir, recursive=args.recursive)
    if not files:
        warn(
            f"{raw_dir} 里没有找到可识别的资料文件。"
            f"支持的格式：{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
        return 0

    manifest_path = Path(args.out) if args.out else settings.root / "config" / "manifest.csv"
    existing = load_manifest(manifest_path) if manifest_path.is_file() else []
    term = args.term or next(
        (str(course.get("term") or "") for course in settings.courses if course.get("id") == course_id),
        "",
    )

    drafts = build_drafts(
        files,
        course_id=course_id,
        term=term,
        existing=existing,
        doc_type=args.doc_type,
        trust_level=args.trust_level,
    )

    info(f"扫描目录：{raw_dir}")
    info(f"发现 {len(files)} 个资料文件，其中 {len(files) - len(drafts)} 个已登记，{len(drafts)} 个待登记。")
    if not drafts:
        return 0

    info("")
    print_table(
        ["编号", "类型", "可信级别", "文件名"],
        [
            [
                draft.doc_key,
                DOC_TYPES.get(draft.doc_type, draft.doc_type),
                TRUST_LABELS.get(draft.trust_level, str(draft.trust_level)),
                draft.filename,
            ]
            for draft in drafts
        ],
    )

    guessed = [draft for draft in drafts if draft.notes]
    info("")
    if guessed:
        warn(
            f"其中 {len(guessed)} 行的类型是靠扩展名猜的（编号以 slide- 开头居多），"
            f"请打开台账逐行确认。"
        )

    if args.dry_run:
        info("")
        info("dry-run：没有写入台账。确认无误后去掉 --dry-run 再跑一次。")
        return 0

    written = append_rows(manifest_path, drafts)
    info("")
    info(f"已追加 {written} 行到 {manifest_path}")
    info(
        "下一步：打开这个文件，逐行确认 doc_type（资料类型）与 trust_level（可信级别），"
        f"确认完删掉 notes 里的「{AUTO_NOTE}」。"
    )
    info("然后用 python -m pipeline manifest 复核一遍。")
    return 0


def cmd_render(settings: Settings, args: argparse.Namespace) -> int:
    entries = resolve_entries(settings, args)
    store = open_store(settings)
    exit_code = 0
    for entry in entries:
        try:
            rendered, warnings = render_document(entry, settings)
        except ConvertError as exc:
            fail(f"{entry.doc_key}：{exc}")
            exit_code = 1
            continue
        save_render(settings, rendered)
        for message in warnings:
            warn(message)
        info(f"{entry.doc_key}：{rendered.page_count} 页 → {rendered.converter}")
    store.close()
    return exit_code


def cmd_parse(settings: Settings, args: argparse.Namespace) -> int:
    entries = resolve_entries(settings, args)
    from .runner import load_render

    try:
        client = build_client(settings)
    except (ConfigError, LLMError) as exc:
        fail(str(exc))
        return 1

    pages = parse_pages(getattr(args, "pages", None))
    exit_code = 0
    for entry in entries:
        rendered = load_render(settings, entry.doc_key)
        if rendered is None:
            fail(f"{entry.doc_key}：还没有渲染结果，请先执行 render。")
            exit_code = 1
            continue
        report, _ = parse_document(
            rendered,
            entry,
            settings,
            client,
            pages=sorted(pages) if pages else None,
            use_cache=not args.no_cache,
            progress=lambda page, done, total: (
                info(f"  解析 {done}/{total} 页")
                if (done % 5 == 0 or done == total or page.error)
                else None
            ),
        )
        _, missing = write_combined_markdown(settings, entry, rendered.page_count)
        info(f"{entry.doc_key}：{report.summary()}")
        if missing:
            warn(f"{entry.doc_key}：仍有 {len(missing)} 页没有解析结果 {missing[:10]}")
        summarize_failures([(page.page_no, page.error or "") for page in report.failures])
    client.close()
    return exit_code


def cmd_golden(settings: Settings, args: argparse.Namespace) -> int:
    """黄金测试集：跑一遍固定题集，给出命中率与编造率。"""
    from datetime import datetime

    from .golden import (
        GoldenError,
        build_worksheet,
        load_cases,
        plan_status,
        render_report,
        run_golden,
        summarize_cases,
    )

    path = Path(args.file) if args.file else settings.root / "tests" / "golden" / "qa.jsonl"
    if not path.is_absolute():
        path = (settings.root / path).resolve()

    try:
        cases = load_cases(path)
    except GoldenError as exc:
        fail(str(exc))
        return 2

    if not cases:
        warn(f"{path} 里还没有题目。格式说明见 tests/golden/README.md。")
        return 2

    info(f"题集：{path}")
    info(summarize_cases(cases))
    status = plan_status(cases)
    for line in status.lines():
        info(line)
    if args.only:
        wanted = set(args.only)
        known = {case.id for case in cases}
        unknown = sorted(wanted - known)
        if unknown:
            fail(f"题集里没有这些 id：{', '.join(unknown)}")
            return 2
        cases = [case for case in cases if case.id in wanted]
        info(f"只跑指定的 {len(cases)} 题")

    if args.check:
        for case in cases:
            kind = "负样本" if case.negative else f"正样本（期望第 {case.expect_pages} 页）"
            info(f"  {case.status:<5} {case.id:<16} {kind:<26} {case.question}")
        info("")
        info("格式校验通过。加 --only-retrieval 可以只测检索（最便宜）。")
        return 0

    if args.worksheet:
        target = (
            Path(args.worksheet)
            if args.worksheet not in ("", "auto")
            else settings.root / "tests" / "golden" / "填写工作单.md"
        )
        target.write_text(build_worksheet(cases, settings), encoding="utf-8")
        info(f"填写工作单已写入：{target.resolve()}")
        info(
            "打开它，一个槽位一个槽位地看原文、改 qa.jsonl；"
            "改完想试就跑 golden --only <id>。"
        )
        return 0

    store = open_store(settings)
    reranker = None
    try:
        embed = embed_client(settings)
        if not getattr(args, "no_rerank", False):
            reranker = rerank_client(settings)
        try:
            generator = OpenAICompatClient(
                ModelConfig.from_dict("generate", settings.model_config("generate"))
            )
        except (ConfigError, LLMError) as exc:
            fail(str(exc))
            embed.close()
            if reranker is not None:
                reranker.close()
            store.close()
            return 1
    except (ConfigError, LLMError) as exc:
        fail(str(exc))
        store.close()
        return 1

    def show(result, index: int, total: int) -> None:
        case = result.case
        form = "选择" if case.is_choice else "问答"
        if case.negative:
            if not result.mode:
                mark = "未判定（仅检索模式）"
            elif result.fabricated:
                mark = "编造嫌疑"
            elif result.handed_off:
                mark = "转人工"
            elif result.mode == "general":
                mark = "通用知识兜底（已标注）"
            elif result.mode == "web":
                mark = "联网兜底（已标注）"
            else:
                mark = "已说明缺口"
            info(
                f"  [{index}/{total}] {case.id} 负样本（{form}）→ {mark}"
                f"（模式 {result.mode or '—'}，最高分 {result.top_score:.3f}）"
            )
        else:
            mark = "命中" if result.retrieval_hit else "未命中"
            front = "前置命中" if result.front_hit else "前置未命中"
            shown = "、".join(
                f"{label}({score:.3f})" for label, score in result.top_hits
            ) or "（没召回任何内容）"
            info(
                f"  [{index}/{total}] {case.id} 正样本（{form}）→ 检索{mark}、{front}"
                f"（期望第 {case.expect_pages} 页）"
            )
            info(f"        最相关的几条：{shown}")
            if case.is_choice and result.mode:
                if result.choice_correct:
                    info(f"        选择：{result.chosen} ✅")
                elif result.chosen:
                    info(f"        选择：{result.chosen} ❌（正确 {case.expected_choice}）")
                else:
                    info("        选择：未能从回答里识别出选项")
            if result.note:
                info(f"        说明：{result.note}")
            if result.mode:
                info(f"        作答依据：{result.mode}")
        if result.error:
            warn(f"      运行出错：{result.error.splitlines()[0]}")

    try:
        report = run_golden(
            cases,
            settings,
            store,
            embed,
            generator,
            retrieval_only=args.only_retrieval,
            progress=show,
            include_ids=set(args.only or []),
            rerank_client=reranker,
        )
    except LLMError as exc:
        fail(str(exc))
        return 1
    finally:
        embed.close()
        if reranker is not None:
            reranker.close()
        generator.close()
        store.close()

    thresholds = {
        "hit_rate": args.min_hit_rate,
        "fabrication_rate": args.max_fabrication_rate,
    }
    report_path = (
        Path(args.report)
        if args.report
        else settings.root / "tests" / "golden" / "report.md"
    )
    report_path.write_text(
        render_report(
            report,
            thresholds=thresholds,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
        encoding="utf-8",
    )

    info("")
    info(f"检索命中率：{report.hit_rate:.0%}（目标 ≥ {args.min_hit_rate:.0%}）")
    if not args.only_retrieval:
        if report.negative:
            info(
                f"冒充课件率：{report.fabrication_rate:.0%}"
                f"（目标 ≤ {args.max_fabrication_rate:.0%}）"
            )
        else:
            info("冒充课件率：本次没有跑到负样本，未评估")
        info(f"回答可用率：{report.answer_rate:.0%}")
        info(f"课件依据率：{report.kb_rate:.0%}（其余为联网/通用知识兜底）")
        if report.negative:
            info(f"负样本转人工率：{report.handoff_rate:.0%}")
        if report.choice_cases:
            info(
                f"选择题正确率：{report.choice_accuracy:.0%}"
                f"（{len(report.choice_cases)} 道）"
            )
        info(f"要点覆盖率：{report.point_rate:.0%}（低于 60% 就该人工打开回答看）")
    info(f"报告：{report_path}")

    fabricated = [item for item in report.results if item.fabricated]
    if fabricated:
        warn(
            f"有 {len(fabricated)} 道负样本把课外内容当成了课件内容："
            f"{', '.join(item.case.id for item in fabricated)}"
        )

    ok = (
        report.hit_rate >= args.min_hit_rate
        and report.fabrication_rate <= args.max_fabrication_rate
    )
    if not ok:
        fail("有指标不达标，详见报告。")
    return 0 if ok else 1


def cmd_recall(settings: Settings, args: argparse.Namespace) -> int:
    """补漏校对：对指定的页，拿原图和已有结果再问一遍，只补漏掉的文字。

    为什么需要它：第一遍转写会稳定地漏掉「图上的短标签、口诀、例题答案」
    这类孤立文字。全量重跑代价高、而且 temperature=0 时结果不会变；
    只对自检挑出来的页做一次「找漏」既便宜又有效。
    """
    from .ai_parser import merge_recall, recall_page
    from .runner import load_render, write_combined_markdown

    entries = resolve_entries(settings, args)
    pages = parse_pages(args.pages)
    if not pages:
        fail("请用 --pages 指定要校对的页，例如 --pages 14,26,35")
        return 2

    try:
        client = build_client(settings)
    except (ConfigError, LLMError) as exc:
        fail(str(exc))
        return 1

    exit_code = 0
    for entry in entries:
        rendered = load_render(settings, entry.doc_key)
        if rendered is None:
            fail(f"{entry.doc_key}：还没有渲染结果，请先执行 render。")
            exit_code = 1
            continue

        assets = {asset.page_no: asset for asset in rendered.pages}
        pages_dir = settings.paths.doc_pages(entry.doc_key)
        hint_map = parse_hints(args.hints)
        info(f"\n=== {entry.doc_key} · {entry.title} ===")

        for page_no in sorted(pages):
            asset = assets.get(page_no)
            page_path = pages_dir / f"{entry.doc_key}_page_{page_no:04d}.md"
            if asset is None or not page_path.is_file():
                warn(f"第 {page_no} 页：找不到页面或解析结果，跳过")
                continue

            existing = page_path.read_text(encoding="utf-8")
            result = recall_page(
                entry,
                settings,
                asset,
                existing,
                client,
                hints=hint_map.get(page_no, ()),
            )
            if result.error:
                fail(f"第 {page_no} 页校对失败：{result.error}")
                exit_code = 1
                continue
            if not result.added:
                info(f"  第 {page_no} 页：没有发现遗漏")
                continue

            if result.suspicious:
                warn(
                    f"第 {page_no} 页：补漏结果异常（{result.reason}），"
                    f"**没有写入**。多半是模型跑偏了，请打开原图和"
                    f"`{page_path}` 人工确认。"
                )
                for line in result.added.splitlines()[:5]:
                    info(f"      {line.strip()}")
                exit_code = 1
                continue

            page_path.write_text(
                merge_recall(existing, result.added), encoding="utf-8"
            )
            info(f"  第 {page_no} 页：补上 {len(result.added.splitlines())} 行")
            for line in result.added.splitlines()[:6]:
                info(f"      {line.strip()}")

        _, missing = write_combined_markdown(settings, entry, rendered.page_count)
        if missing:
            warn(f"{entry.doc_key}：仍有 {len(missing)} 页没有解析结果")
        info(f"{entry.doc_key}：已更新页面文件，接着执行 chunk 重新切块")

    client.close()
    return exit_code


def cmd_chunk(settings: Settings, args: argparse.Namespace) -> int:
    entries = resolve_entries(settings, args)
    exit_code = 0
    for entry in entries:
        markdown_path = settings.paths.doc_markdown(entry.doc_key)
        if not markdown_path.is_file():
            fail(f"{entry.doc_key}：找不到 {markdown_path}，请先执行 parse。")
            exit_code = 1
            continue
        chunks = run_chunking(settings, entry, allow_missing=True)
        sizes = [chunk.char_count for chunk in chunks] or [0]
        info(
            f"{entry.doc_key}：{len(chunks)} 块｜字数 {min(sizes)}-{max(sizes)}｜"
            f"平均 {sum(sizes) // len(sizes)}"
        )
    return exit_code


def cmd_load(settings: Settings, args: argparse.Namespace) -> int:
    entries = resolve_entries(settings, args)
    store = open_store(settings)
    exit_code = 0
    try:
        client = embed_client(settings)
    except (ConfigError, LLMError) as exc:
        fail(str(exc))
        store.close()
        return 1

    from .embedder import embed_chunks, load_cached_embeddings, save_embeddings
    from .store import ChunkRecord, DocumentRecord

    for entry in entries:
        try:
            chunks = load_chunks(settings, entry.doc_key)
        except FileNotFoundError as exc:
            fail(str(exc))
            exit_code = 1
            continue
        if not chunks:
            warn(f"{entry.doc_key}：切块结果为空，跳过。")
            continue
        emb_path = settings.paths.doc_embeddings(entry.doc_key)
        vectors, note = load_cached_embeddings(emb_path, chunks)
        if vectors is not None:
            info(f"  {note}，不重复调用向量模型")
        else:
            info(f"  需要重新向量化（{note}）")
            try:
                vectors = embed_chunks(
                    chunks,
                    client,
                    batch_size=int(settings.embedding.get("batch_size", 32)),
                    expected_dim=int(settings.embedding.get("dim", 0)) or None,
                    progress=lambda done, total: (
                        info(f"  向量化 {done}/{total} 块")
                        if (done % 40 == 0 or done == total)
                        else None
                    ),
                )
            except LLMError as exc:
                fail(f"{entry.doc_key}：向量化失败 {exc}")
                exit_code = 1
                continue
        # 无论复用还是新算，都按带内容指纹的新格式重存一次：
        # 旧格式（每行一个裸数组）没有指纹，内容改了但条数没变时察觉不到，
        # 复用一次之后就自动升级，下次比对就能逐条核对。
        save_embeddings(emb_path, chunks, vectors)
        records = [
            ChunkRecord.from_chunk(chunk, trust_level=entry.trust_level, embedding=vector)
            for chunk, vector in zip(chunks, vectors)
        ]
        document = DocumentRecord.from_entry(
            entry, status="loaded", chunk_count=len(records)
        )
        loaded = store.replace_chunks(document, records)
        info(f"{entry.doc_key}：已写入 {loaded} 块 → {store.backend}")
    client.close()
    store.close()
    return exit_code


def cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    entries = resolve_entries(settings, args)
    store = open_store(settings)

    try:
        vision = build_client(settings)
        embed = embed_client(settings)
    except (ConfigError, LLMError) as exc:
        fail(str(exc))
        store.close()
        return 1

    options = RunOptions(
        pages=parse_pages(getattr(args, "pages", None)),
        use_cache=not args.no_cache,
        allow_partial=bool(getattr(args, "allow_partial", False)),
        skip_embed=bool(getattr(args, "skip_embed", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
    )

    exit_code = 0
    for entry in entries:
        info(f"\n=== {entry.doc_key} · {entry.title}（{entry.trust_label}）===")
        try:
            summary = run_document(
                entry,
                settings,
                store,
                vision_client=vision,
                embed_client=embed,
                options=options,
                log=info,
            )
        except (ConvertError, FileNotFoundError, LLMError) as exc:
            fail(f"{entry.doc_key}：{exc}")
            exit_code = 1
            continue

        for message in summary.warnings:
            warn(message)
        summarize_failures(summary.failure_details)
        if summary.parsed_pages and summary.failed_pages == summary.parsed_pages:
            warn(
                "本份资料「全部页」都失败，这不是单页问题——"
                "优先查密钥、账号额度与网络，而不是去挑某一页。"
            )
        if summary.skipped_stages:
            warn(f"{entry.doc_key}：已跳过 {'、'.join(summary.skipped_stages)}")
        info(
            f"=== 小结：{summary.page_count} 页｜解析 {summary.parsed_pages} 页"
            f"（失败 {summary.failed_pages}）｜{summary.chunk_count} 块｜入库 {summary.loaded} ==="
        )

    vision.close()
    embed.close()
    store.close()
    return exit_code


def cmd_sample(settings: Settings, args: argparse.Namespace) -> int:
    entries = resolve_entries(settings, args)
    from .ai_parser import PageParse, ParseReport, annotate_page

    for entry in entries:
        pages_dir = settings.paths.doc_pages(entry.doc_key)
        files = sorted(pages_dir.glob(f"{entry.doc_key}_page_*.md"))
        if not files:
            fail(f"{entry.doc_key}：还没有解析结果，请先执行 parse 或 run。")
            return 1
        report = ParseReport()
        for path in files:
            page_no = int(path.stem.rsplit("_", 1)[1])
            content = path.read_text(encoding="utf-8")
            page = PageParse(page_no=page_no, markdown=content)
            if content.startswith("[本页解析失败"):
                page.error = content.strip("[]")
            annotate_page(page)
            report.pages.append(page)
        items = build_sample(
            entry, settings, report, ratio=float(getattr(args, "ratio", 0.05))
        )
        path = write_sample_report(settings, entry, items, summary=report.summary())
        info(f"{entry.doc_key}：抽查 {len(items)} 页 → {path}")
    return 0


def cmd_search(settings: Settings, args: argparse.Namespace) -> int:
    store = open_store(settings)
    client = embed_client(settings)
    reranker = None if getattr(args, "no_rerank", False) else rerank_client(settings)
    try:
        result = retrieve(
            settings,
            store,
            client,
            args.query,
            course_id=getattr(args, "course", None),
            top_k=args.top_k,
            rerank=reranker is not None,
            rerank_client=reranker,
        )
    except LLMError as exc:
        fail(str(exc))
        return 1
    finally:
        client.close()

    info(f"问题：{args.query}")
    info(f"最高相关度：{result.top_score:.3f}｜判定：{'有依据' if result.has_evidence else '无依据'}"
         + (f"（{result.reason}）" if result.reason else ""))
    for rank, hit in enumerate(result.hits[: args.top_k or 5], start=1):
        preview = hit.content.replace("\n", " ")[:90]
        extra = f"（向量 {hit.vector_score:.3f}）" if hit.vector_score else ""
        info(
            f"  {rank}. [{hit.score:.3f}]{extra} {hit.page_label}"
            f"｜{hit.section_title}｜{preview}"
        )
    store.close()
    if reranker is not None:
        reranker.close()
    return 0


def cmd_ask(settings: Settings, args: argparse.Namespace) -> int:
    store = open_store(settings)
    reranker = None
    embed = None
    generator = None
    try:
        embed = embed_client(settings)
        reranker = None if getattr(args, "no_rerank", False) else rerank_client(settings)
        generator = OpenAICompatClient(
            ModelConfig.from_dict("generate", settings.model_config("generate"))
        )
    except (ConfigError, LLMError) as exc:
        fail(str(exc))
        for client in (embed, reranker, generator):
            if client is not None:
                client.close()
        store.close()
        return 1

    try:
        answer = answer_question(
            settings,
            store,
            embed,
            generator,
            args.query,
            course_id=getattr(args, "course", None),
            rerank_client=reranker,
        )
    except LLMError as exc:
        fail(str(exc))
        return 1
    finally:
        for client in (embed, reranker, generator):
            if client is not None:
                client.close()
        store.close()

    info(f"\n模式：{answer.mode}｜最高相关度 {answer.top_score:.3f}｜"
         f"引用页码 {answer.citations or '（无）'}")
    if answer.note:
        info(f"说明：{answer.note}")
    info(f"\n{answer.text}")
    info("\n--- 引用到的内容块 ---")
    for rank, hit in enumerate(answer.hits[:5], start=1):
        info(f"  {rank}. {hit.page_label}｜{hit.section_title}｜{hit.score:.3f}")

    return 0


def cmd_stats(settings: Settings, args: argparse.Namespace) -> int:
    store = open_store(settings)
    course_id = getattr(args, "course", None) or settings.default_course_id
    stats = store.stats(course_id)
    store.close()
    if not stats:
        warn("统计信息不可用。")
        return 0
    info(f"后端：{stats.get('backend')}｜课程：{settings.course_name(course_id)}")
    info(f"资料 {stats.get('documents', 0)} 份｜内容块 {stats.get('chunks', 0)} 块")
    by_trust = stats.get("chunks_by_trust") or {}
    if by_trust:
        info("按可信级别：" + "｜".join(f"级别{k}：{v}块" for k, v in sorted(by_trust.items())))
    doc_list = stats.get("doc_list") or []
    if doc_list:
        info("\n资料明细：")
        for row in doc_list:
            info(
                f"  {row.get('doc_key'):<20}级别{row.get('trust_level')}  "
                f"{row.get('status'):<10}{row.get('chunk_count') or 0} 块  {row.get('title')}"
            )
    return 0


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="课程辅导智能体 · 离线建库流水线",
    )
    parser.add_argument("--root", help="项目根目录（默认按本文件位置推断）")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_doc_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--doc", action="append", help="资料编号，可重复")
        target.add_argument("--all", action="store_true", help="处理该课程全部资料")
        target.add_argument("--course", help="课程编号，默认取 config 里的 default_course_id")

    sub.add_parser("doctor", help="环境与配置自检（不联网、不花钱）")

    p = sub.add_parser("keys", help="密钥体检：确认密钥有效、位置正确（免费接口，不消耗额度）")
    p.add_argument(
        "--role",
        action="append",
        choices=("vision", "embedding", "generate"),
        help="只查某一类，可重复；默认三类都查",
    )
    p.add_argument("--timeout", type=float, default=25.0, help="单个请求超时秒数")
    p.add_argument("--list", action="store_true", help="顺带列出每个平台可用的全部模型名")
    p.add_argument("--grep", help="与 --list 搭配：只列出名字里含该关键字的模型")

    p = sub.add_parser("manifest", help="查看并校验资料台账")
    p.add_argument("--course", help="只看某门课")

    p = sub.add_parser("register", help="步骤 1 辅助：扫描 data/raw，生成台账草稿行")
    p.add_argument("--dir", help="要扫描的目录，默认 data/raw")
    p.add_argument("--out", help="台账文件，默认 config/manifest.csv")
    p.add_argument("--course", help="课程编号，默认取 config 里的 default_course_id")
    p.add_argument("--term", help="学期，默认取该课程在 config 里的 term")
    p.add_argument("--doc-type", help="强制指定资料类型（不填则按文件名猜）")
    p.add_argument("--trust-level", type=int, choices=(1, 2, 3), help="强制指定可信级别")
    p.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    p.add_argument("--dry-run", action="store_true", help="只预览，不写台账")

    p = sub.add_parser("render", help="步骤 2：课件 → 逐页图片")
    add_doc_args(p)

    p = sub.add_parser("parse", help="步骤 3：AI 逐页结构化解析")
    add_doc_args(p)
    p.add_argument("--pages", help="只解析指定页，如 1-3,7")
    p.add_argument("--no-cache", action="store_true", help="忽略缓存，强制重新解析")

    p = sub.add_parser("chunk", help="步骤 5：清洗与切块")
    add_doc_args(p)

    p = sub.add_parser(
        "recall",
        help="补漏校对：对指定的页，拿原图和已有结果再问一遍，只补漏掉的文字",
    )
    add_doc_args(p)
    p.add_argument("--pages", required=True, help="要校对的页，如 14,26,35")
    p.add_argument(
        "--hints",
        help="自检怀疑漏掉的内容，形如 26:1.5A|1A|5Ω,13:独立源。"
        "带上它补漏会明显更准（把开放式找漏变成定向确认）",
    )

    p = sub.add_parser(
        "golden",
        help="黄金测试集：跑一遍固定题集，给出检索命中率与编造率",
    )
    p.add_argument("--file", help="题集文件，默认 tests/golden/qa.jsonl")
    p.add_argument("--check", action="store_true", help="只校验题集格式，不联网")
    p.add_argument(
        "--worksheet",
        nargs="?",
        const="auto",
        help="生成填写工作单（槽位 + 对应课件原文放一起），可选指定输出路径",
    )
    p.add_argument(
        "--only-retrieval",
        action="store_true",
        help="只测检索命中率（只调用向量模型，最便宜）",
    )
    p.add_argument("--only", action="append", help="只跑指定 id，可重复")
    p.add_argument("--no-rerank", action="store_true", help="关闭重排，对比纯向量的效果")
    p.add_argument("--limit", type=int, help="只跑前 N 题")
    p.add_argument("--report", help="报告输出路径，默认 tests/golden/report.md")
    p.add_argument("--min-hit-rate", type=float, default=0.8, help="检索命中率下限，默认 0.8")
    p.add_argument(
        "--max-fabrication-rate",
        type=float,
        default=0.05,
        help="编造率上限，默认 0.05",
    )

    p = sub.add_parser("load", help="步骤 6：向量化并写入知识库")
    add_doc_args(p)

    p = sub.add_parser("run", help="一次跑完 2→6")
    add_doc_args(p)
    p.add_argument("--pages", help="只解析指定页，如 1-3,7")
    p.add_argument("--no-cache", action="store_true", help="忽略缓存，强制重新解析")
    p.add_argument("--allow-partial", action="store_true", help="允许在缺页的情况下继续切块入库")
    p.add_argument("--skip-embed", action="store_true", help="只到切块为止")
    p.add_argument("--dry-run", action="store_true", help="只产出本地文件，不写知识库")

    p = sub.add_parser("sample", help="步骤 4：生成人工抽查报告")
    add_doc_args(p)
    p.add_argument("--ratio", type=float, default=0.05, help="抽查比例，默认 0.05")

    p = sub.add_parser("search", help="检索自检：看问题能召回哪些页")
    p.add_argument("query", help="要检索的问题")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--course")
    p.add_argument("--no-rerank", action="store_true", help="关闭重排，只看纯向量召回")

    p = sub.add_parser("ask", help="答疑自检：走完整决策链并校验引用")
    p.add_argument("query", help="学生的问题")
    p.add_argument("--course")
    p.add_argument("--no-rerank", action="store_true", help="关闭重排，对比纯向量的效果")

    p = sub.add_parser("stats", help="知识库统计")
    p.add_argument("--course")

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = Settings.load(args.root)
    except ConfigError as exc:
        fail(str(exc))
        return 2

    handlers = {
        "doctor": cmd_doctor,
        "keys": cmd_keys,
        "manifest": cmd_manifest,
        "register": cmd_register,
        "render": cmd_render,
        "parse": cmd_parse,
        "recall": cmd_recall,
        "golden": cmd_golden,
        "chunk": cmd_chunk,
        "load": cmd_load,
        "run": cmd_run,
        "sample": cmd_sample,
        "search": cmd_search,
        "ask": cmd_ask,
        "stats": cmd_stats,
    }

    try:
        return handlers[args.command](settings, args)
    except (ManifestError, ConfigError, ConvertError) as exc:
        fail(str(exc))
        return 2
    except KeyboardInterrupt:
        fail("已中断。已完成的阶段结果都保留在 data/work/ 下，可继续重跑。")
        return 130
