"""黄金测试集：加载、校验、运行、出报告。

为什么它是整个项目里最重要的基础设施（方案 §8.1）：

没有它，判断「这次调教是变好了还是变坏了」只能凭感觉。改一句提示词、
换一个模型、补一份资料，都可能让某些问题从能答变成乱答，而你不会立刻发现。
有了固定题集，一条命令就能给出命中率和编造率两个数字。

题集文件是 ``tests/golden/qa.jsonl``，每题一行 JSON。两种题型：

* **正样本**（``negative: false``）：资料里确实有，用于测命中率；
* **负样本**（``negative: true``）：资料里根本没有，用于测**会不会把课外内容
  冒充成课件内容**——这 10 道题是整套测试里价值最高的部分。

阶段 2 的产品口径变了：课件没覆盖时允许联网/通用知识作答，但必须标注来源。
所以「编造」不再等于「答了」，而是「走了课件分支、却没有说明课件没覆盖」。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .answer import Answer, answer_question
from .llm import LLMError, OpenAICompatClient
from .retrieval import RetrievalResult, retrieve
from .settings import Settings
from .store import Store
from .textutils import has_external_supplement, mentions_missing_coverage

VALID_STATUS = ("draft", "ready")

#: 选项字母上限：再多就不像选择题了
MAX_CHOICES = 6

#: 从回答里抠出模型选的选项字母。
#: 主要靠提示词要求的第一行「答案：X」；抓不到时再退到「应选X」「正确选项是X」这类说法。
_CHOICE_PATTERNS = (
    re.compile(r"答案\s*[:：]\s*\**\s*([A-Fa-f])(?![A-Za-z])"),
    re.compile(r"(?:应选|选择|选)\s*([A-Fa-f])(?![A-Za-z])"),
    re.compile(
        r"正确(?:选项|答案)?\s*(?:的)?\s*(?:是|为|[:：])\s*\**\s*([A-Fa-f])(?![A-Za-z])"
    ),
    re.compile(r"^\s*([A-Fa-f])\s*[.、．)）]", re.MULTILINE),
)


def extract_choice(answer: str) -> str:
    """从自由回答里抠出模型选的选项字母，抠不到返回空串。

    刻意按「答案：X」优先，因为提示词就是这么要求的；
    其余模式只是兜底。单个字母 A-F 容易和英文缩写撞车，所以每个模式都加了
    边界断言（前后不能紧挨着英文字母）。
    """
    for pattern in _CHOICE_PATTERNS:
        match = pattern.search(answer)
        if match:
            return match.group(1).upper()
    return ""

#: 比对「要点有没有被答到」时抹掉的符号：空白、标点、LaTeX 与 Markdown 标记。
#: 学生的回答和题集里写的要点措辞必然不同，不归一化就全是误报。
_POINT_NORM_RE = re.compile(r"[\s，,。.；;：:!！?？（）()【】\[\]$\\{}\^_*#`~“”\"']+")

#: 要点被改写时允许的差异程度：3-gram 重合到这个比例就算答到了。
#: 定得宽松是有意的——这一项是「提醒你去看」的信号，不是判定对错的裁判。
POINT_MATCH_RATIO = 0.5


def _norm_point(text: str) -> str:
    return _POINT_NORM_RE.sub("", text).lower()


def point_covered(point: str, answer_key: str) -> bool:
    """回答里是否答到了这个要点。

    用 3-gram 重合率而不是整串匹配：模型会把要点改写成自己的话
    （「必须有唯一解」答成「要保证解是唯一的」），逐字比对会全是误报。

    **能力边界**：它擅长发现「整个要点根本没答到」，对「答到了但某个字写错」
    不敏感。回答是被约束在课件原文上生成的，措辞通常与原文高度重合，
    所以真的差得远时，说明那个要点压根没出现。
    """
    key = _norm_point(point)
    if not key:
        return True
    if len(key) < 4:
        return key in answer_key
    grams = [key[index : index + 3] for index in range(len(key) - 2)]
    hits = sum(1 for gram in grams if gram in answer_key)
    return hits / len(grams) >= POINT_MATCH_RATIO


class GoldenError(ValueError):
    """题集文件写错时抛出，消息里要指明是哪一行、错在哪。"""


@dataclass
class GoldenCase:
    id: str
    question: str
    negative: bool = False
    expect_pages: list[int] = field(default_factory=list)
    expect_points: list[str] = field(default_factory=list)
    #: 选择题的选项正文，按 A、B、C… 顺序排列。为空表示这是道问答题。
    choices: list[str] = field(default_factory=list)
    #: 选择题的正确选项字母（大写）
    expected_choice: str = ""
    chapter: str = ""
    source_doc: str = ""
    notes: str = ""
    status: str = "ready"

    @property
    def scorable(self) -> bool:
        """只有 status=ready 的题参与计分；draft 是待你审核的候选。"""
        return self.status == "ready"

    @property
    def is_choice(self) -> bool:
        return bool(self.choices)

    @property
    def choice_letters(self) -> list[str]:
        return [chr(ord("A") + index) for index in range(len(self.choices))]


def load_cases(path: Path) -> list[GoldenCase]:
    """读取并校验题集。任何一处写错都直接报错，不做静默兜底。"""
    if not path.is_file():
        raise GoldenError(
            f"找不到黄金测试集：{path}\n"
            f"格式说明见 tests/golden/README.md。"
        )

    cases: list[GoldenCase] = []
    seen: dict[str, int] = {}

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenError(f"第 {line_no} 行不是合法 JSON：{exc}") from exc
        if not isinstance(payload, dict):
            raise GoldenError(f"第 {line_no} 行必须是 JSON 对象")

        case_id = str(payload.get("id") or "").strip()
        if not case_id:
            raise GoldenError(f"第 {line_no} 行：id 不能为空")
        if case_id in seen:
            raise GoldenError(f"第 {line_no} 行：id {case_id!r} 与第 {seen[case_id]} 行重复")
        seen[case_id] = line_no

        question = str(payload.get("question") or "").strip()
        if not question:
            raise GoldenError(f"第 {line_no} 行：question 不能为空")

        negative = bool(payload.get("negative", False))
        expect_pages = _int_list(payload.get("expect_pages"), line_no, "expect_pages")
        expect_points = _str_list(payload.get("expect_points"), line_no, "expect_points")
        choices = _str_list(payload.get("choices"), line_no, "choices")
        expected_choice = str(payload.get("expected_choice") or "").strip().upper()
        status = str(payload.get("status") or "ready").strip()
        if status not in VALID_STATUS:
            raise GoldenError(
                f"第 {line_no} 行：status 只能是 {'/'.join(VALID_STATUS)}，当前是 {status!r}"
            )

        # 格式约束的意义在于防止「题集看起来能用其实是废的」
        if negative and expect_pages:
            raise GoldenError(
                f"第 {line_no} 行：负样本（negative: true）的 expect_pages 必须为空——"
                f"它本来就是资料里没有的题，不该期望命中任何页"
            )
        if not negative and not expect_pages and status == "ready":
            raise GoldenError(
                f"第 {line_no} 行：正样本必须填 expect_pages（期望命中的页码），"
                f"否则无法判断检索对不对。还没想好的话把 status 写成 draft"
            )

        # ---- 选择题的额外约束 ----
        if choices:
            if len(choices) < 2:
                raise GoldenError(f"第 {line_no} 行：选择题至少要有 2 个选项")
            if len(choices) > MAX_CHOICES:
                raise GoldenError(
                    f"第 {line_no} 行：选项最多 {MAX_CHOICES} 个，当前 {len(choices)} 个"
                )
            if any(not choice.strip() for choice in choices):
                raise GoldenError(f"第 {line_no} 行：选项内容不能为空")
            valid_letters = [chr(ord("A") + i) for i in range(len(choices))]
            if expected_choice not in valid_letters:
                raise GoldenError(
                    f"第 {line_no} 行：expected_choice 必须是 "
                    f"{'/'.join(valid_letters)} 之一，当前是 {expected_choice!r}"
                )
        elif expected_choice:
            raise GoldenError(
                f"第 {line_no} 行：填了 expected_choice 却没填 choices"
            )

        cases.append(
            GoldenCase(
                id=case_id,
                question=question,
                negative=negative,
                expect_pages=expect_pages,
                expect_points=expect_points,
                choices=choices,
                expected_choice=expected_choice,
                chapter=str(payload.get("chapter") or ""),
                source_doc=str(payload.get("source_doc") or ""),
                notes=str(payload.get("notes") or ""),
                status=status,
            )
        )

    return cases


def _int_list(value: Any, line_no: int, field_name: str) -> list[int]:
    if value in (None, [], ""):
        return []
    if not isinstance(value, list):
        raise GoldenError(f"第 {line_no} 行：{field_name} 必须是数组")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise GoldenError(f"第 {line_no} 行：{field_name} 里必须都是整数") from exc


def _str_list(value: Any, line_no: int, field_name: str) -> list[str]:
    if value in (None, [], ""):
        return []
    if not isinstance(value, list):
        raise GoldenError(f"第 {line_no} 行：{field_name} 必须是数组")
    return [str(item) for item in value]


def summarize_cases(cases: Sequence[GoldenCase]) -> str:
    ready = [case for case in cases if case.scorable]
    positive = [case for case in ready if not case.negative]
    negative = [case for case in ready if case.negative]
    choice = [case for case in ready if case.is_choice]
    drafts = [case for case in cases if not case.scorable]
    return (
        f"共 {len(cases)} 题：待计分 {len(ready)} 题"
        f"（正样本 {len(positive)}、负样本 {len(negative)}）"
        f"（选择题共 {len(choice)} 道，其中正样本 "
        f"{sum(1 for case in choice if not case.negative)} 道）"
        f"，草稿 {len(drafts)} 题"
    )


#: 题集规模目标（方案 §8.1：30-50 题，负样本不少于 10 道）。
#: 选择题按「约占正样本的 1/4」定，因为它能机械判分，性价比最高。
PLAN_TARGETS = {
    "total": 40,
    "positive": 30,
    "negative": 10,
    "choice": 8,
}


@dataclass
class PlanStatus:
    """题集离「够用」还差多少。骨架的意义就在于让这个差距一直是可见的。"""

    ready_positive: int = 0
    ready_negative: int = 0
    ready_choice: int = 0
    drafts: int = 0

    @property
    def ready_total(self) -> int:
        return self.ready_positive + self.ready_negative

    @property
    def missing(self) -> dict[str, int]:
        return {
            "正样本": max(0, PLAN_TARGETS["positive"] - self.ready_positive),
            "负样本": max(0, PLAN_TARGETS["negative"] - self.ready_negative),
            "选择题": max(0, PLAN_TARGETS["choice"] - self.ready_choice),
        }

    @property
    def complete(self) -> bool:
        return not any(self.missing.values())

    def lines(self) -> list[str]:
        out = [
            f"目标：{PLAN_TARGETS['total']} 题"
            f"（正样本 {PLAN_TARGETS['positive']}、负样本 {PLAN_TARGETS['negative']}、"
            f"其中选择题 {PLAN_TARGETS['choice']}）",
            f"当前：已就绪 {self.ready_total} 题"
            f"（正 {self.ready_positive}、负 {self.ready_negative}、"
            f"选择题 {self.ready_choice}），草稿 {self.drafts} 题",
        ]
        if self.complete:
            out.append("进度：已达标。每次改动后跑 `python -m pipeline golden` 做回归。")
        else:
            gap = "、".join(f"{name} {count} 道" for name, count in self.missing.items() if count)
            out.append(f"进度：还差 {gap}。")
        return out


def plan_status(cases: Sequence[GoldenCase]) -> PlanStatus:
    ready = [case for case in cases if case.scorable]
    return PlanStatus(
        ready_positive=sum(1 for case in ready if not case.negative),
        ready_negative=sum(1 for case in ready if case.negative),
        ready_choice=sum(1 for case in ready if case.is_choice and not case.negative),
        drafts=sum(1 for case in cases if not case.scorable),
    )


#: 工作单里每页原文最多显示这么多字，太长会把文件撑爆
WORKSHEET_PAGE_CHARS = 1100


def build_worksheet(cases: Sequence[GoldenCase], settings: "Settings") -> str:
    """生成填写工作单：把每个待填槽位和它对应的课件原文放在一起。

    为什么需要它：填题集最费时间的是「在 JSON 和课件之间来回翻」。
    把槽位、原文、图片路径排到同一个文件里，就能一次只盯一道题。
    """
    from .settings import Paths  # 仅用于类型提示，避免循环引用顾虑

    paths: Paths = settings.paths
    titles = {}
    manifest = settings.root / "config" / "manifest.csv"
    if manifest.is_file():
        import csv

        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                titles[(row.get("doc_key") or "").strip()] = (row.get("title") or "").strip()

    drafts = [case for case in cases if not case.scorable]
    lines = [
        "# 黄金测试集填写工作单",
        "",
        f"共 {len(drafts)} 个待填槽位。",
        "",
        "## 怎么用这份工作单",
        "",
        "1. 从上往下，一次只处理一道题；",
        "2. 看下面的**原文解析结果**，判断这道题值不值得考、答案到底在哪几页；",
        "3. 回到 `tests/golden/qa.jsonl` 改这一行：去掉【待写】、改成学生大白话、补 `expect_points`；",
        "4. 想先试试，就在**还是 draft 状态**时跑：",
        "   `python -m pipeline golden --only <id> --only-retrieval`（只测检索，便宜）",
        "   或 `python -m pipeline golden --only <id>`（完整跑）；",
        "5. 满意了再把 `status` 改成 `ready`；",
        "6. 全部填完跑 `python -m pipeline golden --check` 看进度。",
        "",
        "> 负样本没有原文可看。确认它们是「课程内但课件确实没讲」的办法：",
        "> `python -m pipeline golden --only <id> --only-retrieval` 会显示检索到的前几条，",
        "> 打开那几页确认里面真的没有答案。",
        "",
        "> ⚠️ **预填的页码只是起点，不保证覆盖答案。** 它是按切块元数据里的章节范围推出来的，",
        "> 可能只是「这一节从第10页开始」，而答案在第12页。以你下面看到的原文为准，",
        "> 该改就改——`expect_pages` 填错了，这道题就会变成永久误报。",
        "",
        "---",
        "",
    ]

    for number, case in enumerate(drafts, start=1):
        kind = "负样本" if case.negative else ("选择题" if case.is_choice else "问答题")
        lines.append(f"## {number}. `{case.id}` · {kind}")
        lines.append("")
        lines.append(f"- **当前题目**：{case.question}")
        lines.append(f"- **章节**：{case.chapter or '（未标）'}")
        if case.source_doc:
            lines.append(
                f"- **资料**：`{case.source_doc}`"
                f"（{titles.get(case.source_doc, '')}）"
            )
        if case.negative:
            lines.append("- **期望页码**：（负样本不填）")
        else:
            lines.append(f"- **期望页码**：第 {case.expect_pages} 页")
        if case.notes:
            lines.append(f"- **生成时的备注**：{case.notes}")
        lines.append("")

        if case.negative:
            lines.append("> 这是负样本：它按定义不该命中任何页。")
            lines.append("> 用 `--only-retrieval` 跑一次，确认检索到的页里确实没有答案。")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue

        for page_no in case.expect_pages:
            page_path = (
                paths.doc_pages(case.source_doc) / f"{case.source_doc}_page_{page_no:04d}.md"
            )
            image_path = page_path.with_suffix(".jpg")
            lines.append(f"### 第 {page_no} 页")
            lines.append("")
            if image_path.is_file():
                lines.append(f"原页图片：`{image_path}`")
                lines.append("")
            if not page_path.is_file():
                lines.append("（找不到这一页的解析结果，可能页码填错了）")
                lines.append("")
                continue
            text = page_path.read_text(encoding="utf-8").strip()
            if len(text) > WORKSHEET_PAGE_CHARS:
                text = text[:WORKSHEET_PAGE_CHARS] + "\n……（后面还有，篇幅原因省略）"
            lines.extend(["```markdown", text, "```", ""])

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def compose_question(case: GoldenCase) -> str:
    """把选择题的选项拼进问题里。

    刻意走同一条答疑链（检索→生成→引用校验），不单独做「考试模式」——
    否则测出来的是另一个系统的行为，跟学生实际用到的东西对不上。
    """
    if not case.is_choice:
        return case.question
    lines = [case.question, ""]
    for letter, text in zip(case.choice_letters, case.choices):
        lines.append(f"{letter}. {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case: GoldenCase
    top_score: float = 0.0
    hit_pages: list[int] = field(default_factory=list)
    #: 真正交给模型的那几条（前 rerank_top_k 条）命中的页码。
    #: 和 hit_pages 分开记录：召回 20 条里命中、和前 5 条里命中，
    #: 对「能不能答对」的意义完全不同。
    front_pages: list[int] = field(default_factory=list)
    #: 按相关度排序的前几条，形如 [("第25-26页", 0.687), ...]，用于报告和终端显示
    top_hits: list[tuple[str, float]] = field(default_factory=list)
    #: 期望的结论要点里，回答答到了哪些、漏了哪些
    points_hit: list[str] = field(default_factory=list)
    points_missed: list[str] = field(default_factory=list)
    #: 选择题：模型选的选项字母（抠不到为空），以及是否答对
    chosen: str = ""
    choice_correct: bool | None = None
    mode: str = ""
    citations: list[int] = field(default_factory=list)
    citation_valid: bool = True
    latency_ms: int = 0
    error: str = ""
    answer: str = ""
    #: 决策链给出的说明（为什么转人工、引用为什么没通过），排查时最关键的一行
    note: str = ""

    @property
    def retrieval_hit(self) -> bool:
        """期望命中的页码，有没有出现在召回结果里（方案 §7 阶段 1 的口径）。"""
        if self.case.negative:
            return False
        return bool(set(self.hit_pages) & set(self.case.expect_pages))

    @property
    def choice_unreadable(self) -> bool:
        """选择题没答出可识别的选项字母——可能是没答，也可能是格式没遵守。"""
        return self.case.is_choice and not self.chosen

    @property
    def front_hit(self) -> bool:
        """期望命中的页码，有没有出现在**真正交给模型**的那几条里。

        这个比 retrieval_hit 严格得多，也更接近「模型到底有没有拿到依据」。
        两者都报，是为了能把「召回没问题但排序有问题」这种情况区分出来。
        """
        if self.case.negative:
            return False
        return bool(set(self.front_pages) & set(self.case.expect_pages))

    @property
    def answered(self) -> bool:
        """正样本是否给出了回答（课件、联网、通用知识都算）。"""
        if self.mode == "kb":
            return self.citation_valid
        return self.mode in ("web", "general")

    @property
    def answered_from_kb(self) -> bool:
        """是否真的用课件内容作答（而不是靠联网或模型自己的知识兜底）。"""
        return self.mode == "kb" and self.citation_valid

    @property
    def point_rate(self) -> float:
        """要点覆盖率：回答把期望的结论要点答到了多少。

        这是唯一能发现「页码引用正确、但结论说反了」的自动指标。
        它不参与达标判定（措辞差异会让它天然偏低），只作为「去人工看一眼」的信号。
        """
        if self.case.negative or not self.case.expect_points:
            return 1.0
        total = len(self.points_hit) + len(self.points_missed)
        if not total:
            return 1.0
        return len(self.points_hit) / total

    @property
    def choice_rate(self) -> float:
        """选择题正确率：答对的比例。没答出可识别选项的算错。"""
        if not self.case.is_choice or self.case.negative:
            return 1.0
        return 1.0 if self.choice_correct else 0.0

    @property
    def handed_off(self) -> bool:
        return self.mode == "handoff"

    @property
    def fabricated(self) -> bool:
        """负样本里最需要警惕的行为：把课件外的内容说成课件依据。

        阶段 2 起，「课件没覆盖」时会先标注「网络搜索 / 通用知识」再作答，
        这不算编造——学生看得到来源，能自己判断要不要信。
        只有走了 ``kb`` 分支（回答里带课件页码、暗示这是课件内容），
        又没有说明课件里其实没有，才是真正的误导。
        """
        if not self.case.negative:
            return False
        # 仅检索模式没调用生成模型，无从判断说没说「资料里没有」
        if not self.mode:
            return False
        if self.mode != "kb":
            return False
        # 回答里明确开了「课件外补充」小节，等于告诉学生这段不是课件内容
        if has_external_supplement(self.answer):
            return False
        return not mentions_missing_coverage(self.answer)


def describe_hits(hits: Sequence[Any], *, limit: int = 3) -> list[tuple[str, float]]:
    """给报告用的「前几条命中」摘要，按相关度排序。"""
    described: list[tuple[str, float]] = []
    for hit in list(hits)[:limit]:
        page_no = getattr(hit, "page_no", None)
        page_end = getattr(hit, "page_end", None)
        if page_no is None:
            label = "页码未知"
        elif page_end and page_end != page_no:
            label = f"第{page_no}-{page_end}页"
        else:
            label = f"第{page_no}页"
        described.append((label, float(getattr(hit, "score", 0.0))))
    return described


def _format_top_hits(item: CaseResult) -> str:
    if not item.top_hits:
        return "—"
    return "、".join(f"{label}({score:.3f})" for label, score in item.top_hits)


def pages_from_hits(hits: Sequence[Any], *, doc_key: str | None = None) -> list[int]:
    """把命中的内容块展开成页码集合。

    跨页块（page_no..page_end）要把中间每一页都算进去，
    否则「期望命中第 26 页、块是 25-26 页」会被误判成没命中。

    ``doc_key`` 非空时**只统计这份资料命中的块**。这一点非常重要：
    知识库是整个课程混在一起的，19 份资料各有自己的第 25 页。
    如果不区分资料，一道题会「命中」另一份资料的同号页——实测出过
    期望 slide-08 第 9、13 页、实际命中别的资料第 52、53 页却被判为命中的情况。
    """
    pages: set[int] = set()
    for hit in hits:
        if doc_key:
            chunk_key = str(getattr(hit, "chunk_key", "") or "")
            if not chunk_key.startswith(f"{doc_key}#"):
                continue
        start = getattr(hit, "page_no", None)
        if start is None:
            continue
        end = getattr(hit, "page_end", None) or start
        pages.update(range(int(start), int(end) + 1))
    return sorted(pages)


def foreign_docs(hits: Sequence[Any], doc_key: str) -> list[str]:
    """命中的内容块里，有多少来自别的资料。

    这个数字能说明「检索是不是跑偏到别的章节去了」，比单纯的命中率更能
    反映排序质量。
    """
    others: dict[str, int] = {}
    for hit in hits:
        chunk_key = str(getattr(hit, "chunk_key", "") or "")
        name = chunk_key.split("#", 1)[0] if "#" in chunk_key else "（未知）"
        if name and name != doc_key:
            others[name] = others.get(name, 0) + 1
    return [f"{name}×{count}" for name, count in sorted(others.items(), key=lambda kv: -kv[1])]


@dataclass
class GoldenReport:
    results: list[CaseResult] = field(default_factory=list)
    retrieval_only: bool = False
    elapsed_s: float = 0.0

    @property
    def positive(self) -> list[CaseResult]:
        return [item for item in self.results if not item.case.negative]

    @property
    def negative(self) -> list[CaseResult]:
        return [item for item in self.results if item.case.negative]

    @property
    def hit_rate(self) -> float:
        if not self.positive:
            return 1.0
        return sum(1 for item in self.positive if item.retrieval_hit) / len(self.positive)

    @property
    def front_hit_rate(self) -> float:
        if not self.positive:
            return 1.0
        return sum(1 for item in self.positive if item.front_hit) / len(self.positive)

    @property
    def fabrication_rate(self) -> float:
        if not self.negative:
            return 0.0
        return sum(1 for item in self.negative if item.fabricated) / len(self.negative)

    @property
    def answer_rate(self) -> float:
        if self.retrieval_only or not self.positive:
            return 1.0
        return sum(1 for item in self.positive if item.answered) / len(self.positive)

    @property
    def handoff_rate(self) -> float:
        if self.retrieval_only or not self.negative:
            return 1.0
        return sum(1 for item in self.negative if item.handed_off) / len(self.negative)

    @property
    def point_rate(self) -> float:
        """要点覆盖率。只统计「真的给出了回答」的题，否则转人工会把均值拉低。"""
        graded = [
            item
            for item in self.results
            if not item.case.negative and item.case.expect_points and item.answered
        ]
        if not graded:
            return 1.0
        return sum(item.point_rate for item in graded) / len(graded)

    @property
    def kb_rate(self) -> float:
        """正样本里「真的用课件作答」的比例。

        这是阶段 2 最重要的质量指标：回答可用率接近 100% 是兜底策略带来的，
        课件依据率才说明知识库本身有没有命中。两者差距越大，越说明这份资料
        还需要补，或者检索还需要调。
        """
        graded = [item for item in self.positive if item.mode]
        if not graded:
            return 1.0
        return sum(1 for item in graded if item.answered_from_kb) / len(graded)

    @property
    def mode_counts(self) -> dict[str, int]:
        """各档依据出现的次数，用来一眼看出兜底比例。"""
        counts: dict[str, int] = {}
        for item in self.results:
            if item.mode:
                counts[item.mode] = counts.get(item.mode, 0) + 1
        return counts

    @property
    def choice_cases(self) -> list[CaseResult]:
        return [
            item
            for item in self.results
            if item.case.is_choice and not item.case.negative
        ]

    @property
    def choice_accuracy(self) -> float:
        if not self.choice_cases:
            return 1.0
        return sum(item.choice_rate for item in self.choice_cases) / len(self.choice_cases)


def run_case(
    case: GoldenCase,
    settings: Settings,
    store: Store,
    embed_client: OpenAICompatClient,
    generate_client: OpenAICompatClient,
    *,
    retrieval_only: bool,
    rerank_client: OpenAICompatClient | None = None,
) -> CaseResult:
    started = time.perf_counter()
    result = CaseResult(case=case)
    front_n = int(settings.retrieval.get("rerank_top_k", 5))
    asked = compose_question(case)
    try:
        if retrieval_only:
            found: RetrievalResult = retrieve(
                settings, store, embed_client, asked, rerank_client=rerank_client
            )
            result.top_score = found.top_score
            result.hit_pages = pages_from_hits(found.hits, doc_key=case.source_doc)
            result.front_pages = pages_from_hits(
                found.hits[:front_n], doc_key=case.source_doc
            )
            if case.source_doc:
                others = foreign_docs(found.hits, case.source_doc)
                if others:
                    result.note = f"跨资料命中：{'、'.join(others)}"
            result.top_hits = describe_hits(found.hits)
        else:
            answer: Answer = answer_question(
                settings,
                store,
                embed_client,
                generate_client,
                asked,
                rerank_client=rerank_client,
            )
            result.top_score = answer.top_score
            result.hit_pages = pages_from_hits(answer.hits, doc_key=case.source_doc)
            result.front_pages = pages_from_hits(
                answer.hits[:front_n], doc_key=case.source_doc
            )
            result.top_hits = describe_hits(answer.hits)
            result.mode = answer.mode
            result.citations = list(answer.citations)
            result.citation_valid = answer.citation_valid
            result.answer = answer.text
            if answer.failed:
                # 模型故障导致的降级不算「系统正确地转人工」，
                # 否则指标会被虚高的「已转人工」撑起来
                result.error = answer.note or "模型调用失败，已降级为转人工"
            if case.source_doc:
                others = foreign_docs(answer.hits, case.source_doc)
                if others:
                    result.note = (
                        f"跨资料命中：{'、'.join(others)}"
                        + (f"；{answer.note}" if answer.note else "")
                    )
                else:
                    result.note = answer.note
            else:
                result.note = answer.note
            if case.is_choice and not case.negative:
                result.chosen = extract_choice(answer.text)
                if result.chosen:
                    result.choice_correct = result.chosen == case.expected_choice
            if case.expect_points:
                answer_key = _norm_point(answer.text)
                for point in case.expect_points:
                    if point_covered(point, answer_key):
                        result.points_hit.append(point)
                    else:
                        result.points_missed.append(point)
    except LLMError as exc:
        result.error = str(exc)
    result.latency_ms = int((time.perf_counter() - started) * 1000)
    return result


def run_golden(
    cases: Sequence[GoldenCase],
    settings: Settings,
    store: Store,
    embed_client: OpenAICompatClient,
    generate_client: OpenAICompatClient,
    *,
    retrieval_only: bool = False,
    progress: Callable[[CaseResult, int, int], None] | None = None,
    limit: int | None = None,
    include_ids: set[str] | None = None,
    rerank_client: OpenAICompatClient | None = None,
) -> GoldenReport:
    """跑题集。

    ``include_ids`` 是给「试跑草稿」用的：默认只跑 status=ready 的题，
    但你需要在把一道题改成 ready **之前**先试一下它能不能答、页码对不对。
    显式点了 id 就应当跑，不受 status 限制。
    """
    started = time.perf_counter()
    report = GoldenReport(retrieval_only=retrieval_only)
    wanted = include_ids or set()
    targets = [case for case in cases if case.scorable or case.id in wanted]
    if limit:
        targets = targets[:limit]
    for index, case in enumerate(targets, start=1):
        result = run_case(
            case,
            settings,
            store,
            embed_client,
            generate_client,
            retrieval_only=retrieval_only,
            rerank_client=rerank_client,
        )
        report.results.append(result)
        if progress:
            progress(result, index, len(targets))
    report.elapsed_s = time.perf_counter() - started
    return report


def render_report(
    report: GoldenReport,
    *,
    thresholds: dict[str, float],
    generated_at: str,
) -> str:
    """把结果写成 markdown 报告。列表里最前面永远是「出了什么问题」。"""
    mode_label = "仅检索（未调用生成模型）" if report.retrieval_only else "完整（检索 + 生成 + 引用校验）"
    lines = [
        "# 黄金测试集报告",
        "",
        f"- 生成时间：{generated_at}",
        f"- 运行模式：{mode_label}",
        f"- 耗时：{report.elapsed_s:.1f} 秒",
        f"- 题量：正样本 {len(report.positive)} 题、负样本 {len(report.negative)} 题",
        "",
        "## 结论",
        "",
        "| 指标 | 实测 | 目标 | 判定 |",
        "| --- | --- | --- | --- |",
    ]

    hit_target = thresholds.get("hit_rate", 0.8)
    fab_target = thresholds.get("fabrication_rate", 0.05)
    hit_ok = report.hit_rate >= hit_target
    fab_ok = report.fabrication_rate <= fab_target
    lines.append(
        f"| 检索命中率 | {report.hit_rate:.0%} | ≥ {hit_target:.0%} | "
        f"{'✅ 达标' if hit_ok else '❌ 不达标'} |"
    )
    lines.append(f"| 前置命中率 | {report.front_hit_rate:.0%} | 观察项 | — |")
    lines.append(
        f"| 冒充课件率 | {report.fabrication_rate:.0%} | ≤ {fab_target:.0%} | "
        f"{'✅ 达标' if fab_ok else '❌ 不达标'} |"
    )
    if not report.retrieval_only:
        lines.append(f"| 回答可用率 | {report.answer_rate:.0%} | 观察项 | — |")
        lines.append(f"| 课件依据率 | {report.kb_rate:.0%} | 观察项 | — |")
        if report.choice_cases:
            lines.append(
                f"| 选择题正确率 | {report.choice_accuracy:.0%} | 100% | "
                f"{'✅ 达标' if report.choice_accuracy >= 1.0 else '❌ 有答错'} |"
            )
        lines.append(
            f"| 要点覆盖率 | {report.point_rate:.0%} | 观察项 | — |"
        )
        lines.append(f"| 负样本转人工率 | {report.handoff_rate:.0%} | 观察项 | — |")
    if not report.negative:
        lines.append("> 本次没有跑到负样本，编造率与转人工率不作数。")
    if report.mode_counts:
        dist = "、".join(
            f"{mode} {count}" for mode, count in sorted(report.mode_counts.items())
        )
        lines.append(f"> 本次依据分布：{dist}。`general`/`web` 越多，说明课件覆盖越不足。")
    lines.append("")
    lines.extend(
        [
            "> 检索命中率 = 期望页码出现在**召回的全部 20 条**里；"
            "前置命中率 = 出现在**真正交给模型的前 5 条**里。",
            "> 冒充课件率 = 负样本里「走了课件分支、却没说明课件其实没讲」的比例。"
            "联网/通用知识作答、以及带「课件外补充」小节的回答都不计入。",
            "> 两者差距大说明召回没问题但排序有问题，往往比命中率本身更值得调。",
            ">",
            "> **要点覆盖率**是唯一能发现「页码引用正确、但结论说反了」的指标。"
            "它不参与达标判定（模型会改写措辞，天然有些对不上），"
            "低于 60% 就该人工打开回答看一眼。",
            "",
        ]
    )

    low_points = [
        item
        for item in report.results
        if not item.case.negative
        and item.case.expect_points
        and item.mode == "kb"
        and item.point_rate < 0.6
    ]
    if low_points:
        lines.extend(["## 要点覆盖偏低（去人工看一眼回答）", ""])
        for item in low_points:
            lines.append(f"### {item.case.id}：{item.case.question}")
            lines.append("")
            lines.append(f"- 要点覆盖率 {item.point_rate:.0%}，引用页码 {item.citations}")
            lines.append(f"- 回答开头：{item.answer[:300] or '（无回答）'}")
            for point in item.points_missed:
                lines.append(f"  - ❌ 没答到：{point}")
            for point in item.points_hit:
                lines.append(f"  - ✅ 答到了：{point}")
            lines.append("")

    problems = [item for item in report.results if item.error]
    fabricated = [item for item in report.results if item.fabricated]
    missed = [item for item in report.positive if not item.retrieval_hit]

    if fabricated:
        lines.extend(
            [
                "## ⚠️ 编造嫌疑（最需要处理的一类）",
                "",
                "负样本走了「依据课件」分支，回答里却没有说明课件其实没讲这件事——"
                "等于把课外内容冒充成了课件内容。这类必须逐个打开看。",
                "",
            ]
        )
        for item in fabricated:
            lines.append(f"### {item.case.id}：{item.case.question}")
            lines.append("")
            lines.append(f"- 模式：`{item.mode}`，最高相关度 {item.top_score:.3f}")
            lines.append(f"- 回答开头：{item.answer[:200]}")
            lines.append("")

    if missed:
        lines.extend(["## 检索未命中（正样本）", ""])
        for item in missed:
            lines.append(
                f"- `{item.case.id}`：期望第 {item.case.expect_pages} 页，"
                f"前几条命中的是 {_format_top_hits(item)}"
                f"　{item.case.question}"
            )
        lines.append("")

    unanswered = [
        item
        for item in report.positive
        if not item.error and not item.answered
    ]
    if unanswered and not report.retrieval_only:
        lines.extend(
            [
                "## 正样本没能作答（检索到了依据、但没给出可用回答）",
                "",
                "这类问题检索是对的，卡在后面一步。原因写在下面：",
                "",
            ]
        )
        for item in unanswered:
            lines.append(
                f"- `{item.case.id}`（模式 `{item.mode}`）：{item.note or '（没有说明）'}"
            )
        lines.append("")

    fallback = [
        item
        for item in report.positive
        if not item.error and item.mode in ("web", "general")
    ]
    if fallback and not report.retrieval_only:
        lines.extend(
            [
                "## 正样本靠兜底作答（课件没命中，但学生仍拿到了答案）",
                "",
                "这些题不算失败——学生的问题被回答了；但它们是最该优先补资料的清单。",
                "",
            ]
        )
        for item in fallback:
            lines.append(
                f"- `{item.case.id}`（`{item.mode}`）：{item.case.question}"
                f"　期望第 {item.case.expect_pages} 页，实际最高分 {item.top_score:.3f}"
            )
        lines.append("")

    wrong_choices = [
        item for item in report.choice_cases if item.choice_correct is not True
    ]
    if wrong_choices and not report.retrieval_only:
        lines.extend(
            [
                "## 选择题答错或没答出选项（最需要处理）",
                "",
                "选择题有唯一正确答案，这类错误没有解释空间。",
                "",
            ]
        )
        for item in wrong_choices:
            if item.chosen:
                verdict = f"选了 {item.chosen}，正确是 {item.case.expected_choice}"
            else:
                verdict = "没能从回答里识别出选项（可能没答，也可能没按格式写「答案：X」）"
            lines.append(f"### {item.case.id}：{item.case.question}")
            lines.append("")
            lines.append(f"- {verdict}｜模式 `{item.mode}`｜引用页码 {item.citations or '（无）'}")
            lines.append(f"- 回答开头：{item.answer[:200] or '（无回答）'}")
            lines.append("")

    if problems:
        lines.extend(["## 运行出错", ""])
        for item in problems:
            lines.append(f"- `{item.case.id}`：{item.error.splitlines()[0]}")
        lines.append("")

    lines.extend(
        [
            "## 逐题明细",
            "",
            "| id | 题型 | 格式 | 检索命中 | 前置命中 | 模式 | 选了什么 | 引用页码 | 最高分 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in report.results:
        kind = "负样本" if item.case.negative else "正样本"
        if item.error:
            hit = "出错"
        elif item.case.negative:
            hit = "不适用"
        else:
            hit = "✅" if item.retrieval_hit else "❌"
        front = "—" if item.case.negative else ("✅" if item.front_hit else "❌")
        citations = "—" if item.case.negative else (
            ",".join(str(page) for page in item.citations) or "（无）"
        )
        form = "选择" if item.case.is_choice else "问答"
        if not item.case.is_choice:
            picked = "—"
        elif item.choice_correct:
            picked = f"{item.chosen} ✅"
        elif item.chosen:
            picked = f"{item.chosen} ❌（正确 {item.case.expected_choice}）"
        else:
            picked = "未识别"
        lines.append(
            f"| `{item.case.id}` | {kind} | {form} | {hit} | {front} | "
            f"{item.mode or '—'} | {picked} | {citations} | {item.top_score:.3f} |"
        )
    lines.append("")

    lines.extend(
        [
            "## 怎么用这份报告",
            "",
            "1. **先看「编造嫌疑」和「检索未命中」两节**，其余是背景信息。",
            "2. 编造嫌疑要逐个打开回答看：模型是真的在编，还是只是忘了说明「资料里没有」。",
            "3. 检索未命中先怀疑题集的 `expect_pages` 填得对不对，再怀疑检索。",
            "4. 每次改提示词、换模型、补资料之后重跑一遍，把两次数字对比着看——",
            "   这才是「这次调教变好了还是变坏了」的客观依据。",
            "",
        ]
    )
    return "\n".join(lines)
