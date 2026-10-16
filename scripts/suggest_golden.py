"""从已入库的内容块里拟出黄金测试集的候选题。

思路：**反向出题**。从课件原文拟出「答案就在这段原文里」的学生式提问，
于是 expect_pages 天然正确（就是这一块的页码），你只需要审核
「问法自不自然、是不是学生真会问」。

负样本没法这样生成（按定义它们的内容不在知识库里），所以脚本里内置了一批
候选，并用检索逐个校验：如果某道负样本的相关度很高，说明资料里其实有
相关内容，它就不是合格负样本，会被标出来让你换掉。

用法：
    python scripts/suggest_golden.py --count 40
    python scripts/suggest_golden.py --negatives
    python scripts/suggest_golden.py --count 40 --negatives

产出写入 tests/golden/candidates.jsonl（status=draft，不参与计分）。
审核通过后手动挪进 tests/golden/qa.jsonl 并把 status 改成 ready。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.cli import configure_stdio, embed_client  # noqa: E402
from pipeline.llm import LLMError, ModelConfig, OpenAICompatClient  # noqa: E402
from pipeline.retrieval import retrieve  # noqa: E402
from pipeline.settings import Settings  # noqa: E402
from pipeline.store import open_store  # noqa: E402

# 负样本候选：这些主题不在当前 19 份资料覆盖的章节里
# （资料覆盖第 1、2、3、4、7、8、10-15、20、21 章）。
# 挑的是「学生真可能问、但课件没讲」的题，不是一眼就假的题。
NEGATIVE_CANDIDATES = [
    "回转器怎么用在有源滤波器设计里",
    "非线性电阻电路的小信号分析法怎么做",
    "运放虚短虚断的使用条件是什么",
    "怎么用状态变量法分析高阶动态电路",
    "二端口网络的 Y 参数怎么测量",
    "磁路的基本定律和电路有什么对应关系",
    "拉普拉斯变换在电路分析里怎么用",
    "分布参数电路和集中参数电路的分界线怎么定",
    "有源滤波器的设计步骤是怎样的",
    "怎么判断一个电路是稳定的",
    "小信号等效电路是怎么推出来的",
    "傅里叶级数展开后各次谐波的有效值怎么算",
]

# 相关度到这个量级才开始怀疑「资料里其实有答案」。刻意定得高：
# 实测「回转器」那题相关度 0.627，模型照样如实回答「现有资料没有覆盖」，
# 表现完全正确——相关度偏高只说明话题邻近，不等于资料里有答案。
NEGATIVE_SCORE_WARN = 0.75


@dataclass
class Chunk:
    doc_key: str
    title: str
    page_no: int | None
    page_end: int | None
    chapter: str
    section_title: str
    content: str
    char_count: int


def load_chunks(settings: Settings) -> list[Chunk]:
    """从本地切块文件读取内容块。

    刻意读 data/work/<编号>/chunks.jsonl 而不是查数据库：这个文件
    无论用 local 还是 postgres 后端都存在，且不需要联网。
    """
    manifest = settings.root / "config" / "manifest.csv"
    titles: dict[str, str] = {}
    if manifest.is_file():
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("doc_key") or "").strip()
                titles[key] = (row.get("title") or "").strip()

    chunks: list[Chunk] = []
    for path in sorted(settings.paths.work_dir.glob("*/chunks.jsonl")):
        doc_key = path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            content = row.get("content") or ""
            chunks.append(
                Chunk(
                    doc_key=doc_key,
                    title=titles.get(doc_key, doc_key),
                    page_no=row.get("page_no"),
                    page_end=row.get("page_end"),
                    chapter=row.get("chapter") or "",
                    section_title=row.get("section_title") or "",
                    content=content,
                    char_count=int(row.get("char_count") or len(content)),
                )
            )
    return chunks


def pick_chunks(chunks: list[Chunk], *, total: int, per_doc_floor: int = 2) -> list[Chunk]:
    """按资料均匀挑选，优先「内容长、有章节标题」的块。

    均匀是为了让题集覆盖每一章——只挑最长的块会让题集挤在少数几章里。
    """
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_key, []).append(chunk)

    usable: dict[str, list[Chunk]] = {}
    for doc_key, items in by_doc.items():
        candidates = [
            item for item in items if item.char_count >= 250 and item.section_title
        ]
        if candidates:
            usable[doc_key] = sorted(
                candidates, key=lambda item: item.char_count, reverse=True
            )
    if not usable:
        return []

    per_doc = max(per_doc_floor, total // len(usable))
    picked: list[Chunk] = []
    for doc_key in sorted(usable):
        picked.extend(usable[doc_key][:per_doc])
    return picked[:total]


def ask_questions(
    chunk: Chunk, settings: Settings, client: OpenAICompatClient, *, count: int
) -> list[str]:
    system = settings.prompt("golden_question", "system")
    user = settings.prompt("golden_question", "user").format(
        doc_title=chunk.title,
        page_no=chunk.page_no,
        section_title=chunk.section_title or "（未标章节）",
        content=chunk.content[:2500],
        count=count,
    )
    raw = client.chat(system=system, user=user)
    questions: list[str] = []
    for line in raw.splitlines():
        text = line.strip().lstrip("-—•*").strip()
        head, sep, tail = text.partition(". ")
        if sep and head.strip().isdigit():
            text = tail.strip()
        if len(text) >= 6:
            questions.append(text)
    return questions[:count]


def verify_negatives(
    questions: list[str],
    settings: Settings,
    embed: OpenAICompatClient,
    store,
) -> list[dict[str, object]]:
    """给每道负样本候选跑一次检索，看相关度是不是真的低。"""
    verified: list[dict[str, object]] = []
    for question in questions:
        found = retrieve(settings, store, embed, question)
        # 显示**最相关的几条**命中的页码，而不是页码最小的几条——
        # 后者会让人误以为检索去了无关的页面。
        top = "、".join(
            (
                f"第{hit.page_no}-{hit.page_end}页"
                if hit.page_end and hit.page_end != hit.page_no
                else f"第{hit.page_no}页"
            )
            + f"({hit.score:.3f})"
            for hit in found.hits[:3]
            if hit.page_no
        ) or "（没召回任何内容）"
        if not found.hits:
            verdict = "，资料里没有相关内容，是合格的负样本"
        elif found.top_score < NEGATIVE_SCORE_WARN:
            verdict = "，相关度不高，合格；不过还是打开那几页扫一眼确认没有答案"
        else:
            verdict = "，相关度偏高，务必打开那几页确认里面确实没有答案"
        note = f"检索最高 {found.top_score:.3f}：{top}{verdict}"
        ok = found.top_score < NEGATIVE_SCORE_WARN
        verified.append(
            {
                "question": question,
                "ok": ok,
                "top_score": found.top_score,
                "note": note,
            }
        )
    return verified


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="生成黄金测试集候选题")
    parser.add_argument("--root", help="项目根目录")
    parser.add_argument("--count", type=int, default=40, help="正样本候选题总数，默认 40")
    parser.add_argument("--per-chunk", type=int, default=1, help="每块拟几个问题，默认 1")
    parser.add_argument("--negatives", action="store_true", help="顺带校验负样本候选")
    parser.add_argument("--out", help="输出文件，默认 tests/golden/candidates.jsonl")
    args = parser.parse_args(argv)

    settings = Settings.load(args.root)
    chunks = load_chunks(settings)
    if not chunks:
        print("找不到任何切块结果（data/work/*/chunks.jsonl）。先跑 chunk。")
        return 1

    picked = pick_chunks(chunks, total=args.count)
    print(f"共有 {len(chunks)} 个内容块，从中挑出 {len(picked)} 个作为出题素材")

    generator = OpenAICompatClient(
        ModelConfig.from_dict("generate", settings.model_config("generate"))
    )

    rows: list[dict[str, object]] = []
    counter = 0
    for chunk in picked:
        try:
            questions = ask_questions(chunk, settings, generator, count=args.per_chunk)
        except LLMError as exc:
            print(f"  出题失败（{chunk.doc_key} 第 {chunk.page_no} 页）：{exc}")
            continue
        for question in questions:
            counter += 1
            if chunk.page_no:
                pages = list(range(chunk.page_no, (chunk.page_end or chunk.page_no) + 1))
            else:
                pages = []
            rows.append(
                {
                    "id": f"cand-{counter:03d}",
                    "question": question,
                    "negative": False,
                    "expect_pages": pages,
                    "expect_points": [],
                    "chapter": chunk.chapter,
                    "source_doc": chunk.doc_key,
                    "notes": "自动生成，待人工审核：确认问法自然、且答案确实在这几页",
                    "status": "draft",
                }
            )
        print(f"  {chunk.doc_key} 第 {chunk.page_no} 页 → {len(questions)} 题")
    generator.close()

    if args.negatives:
        print()
        print("校验负样本候选（相关度过高说明资料里其实有相关内容）")
        embed = embed_client(settings)
        store = open_store(settings)
        try:
            verified = verify_negatives(NEGATIVE_CANDIDATES, settings, embed, store)
        except LLMError as exc:
            print(f"  校验失败：{exc}")
            verified = []
        finally:
            embed.close()
            store.close()
        for item in verified:
            mark = "OK" if item["ok"] else "换掉"
            print(f"  [{mark}] {item['question']}｜{item['note']}")
        for item in verified:
            counter += 1
            rows.append(
                {
                    "id": f"negcand-{counter:03d}",
                    "question": item["question"],
                    "negative": True,
                    "expect_pages": [],
                    "expect_points": [],
                    "chapter": "（资料未覆盖）",
                    "source_doc": "",
                    "notes": str(item["note"]) + "。自动生成，待人工审核",
                    "status": "draft",
                }
            )

    out = (
        Path(args.out)
        if args.out
        else settings.root / "tests" / "golden" / "candidates.jsonl"
    )
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"已写入 {len(rows)} 条候选：{out}")
    print("下一步：逐条审核，把认可的挪进 tests/golden/qa.jsonl 并把 status 改成 ready。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
