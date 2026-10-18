"""首次使用的 5 题基础测评。

只做概念理解、公式含义、适用条件、易错点这几类题，不出复杂计算。
题库在 ``config/assessment.yaml``，每题都带课件出处，方便学生答完回看。
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from .llm import LLMError, OpenAICompatClient
from .settings import Settings


class AssessmentError(ValueError):
    """题库缺失或格式不对。"""


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    points: list[str]
    chapter: str
    doc: str
    pages: list[int]
    level: int
    type: str = "concept"
    review: str = ""


@dataclass
class Grade:
    score: float
    feedback: str
    weak_point: str = ""


#: 每次测评的题型配比：2 概念判断 + 2 简单计算 + 1 小型综合
TYPE_PLAN = (("concept", 2), ("calculation", 2), ("comprehensive", 1))
TYPE_ORDER = {"concept": 0, "calculation": 1, "comprehensive": 2}
TYPE_LABELS = {
    "concept": "概念判断",
    "calculation": "简单计算",
    "comprehensive": "小型综合",
}


def load_bank(root: Path) -> list[Question]:
    path = Path(root) / "config" / "assessment.yaml"
    if not path.is_file():
        raise AssessmentError(f"找不到测评题库：{path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_items = payload.get("questions") or []
    if not isinstance(raw_items, list) or not raw_items:
        raise AssessmentError("config/assessment.yaml 里没有 questions")

    bank: list[Question] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            raise AssessmentError(f"第 {index} 题不是对象")
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if not question or not answer:
            raise AssessmentError(f"第 {index} 题缺少 question 或 answer")
        bank.append(
            Question(
                id=str(item.get("id") or f"diag-{index:02d}"),
                question=question,
                answer=answer,
                points=[str(p) for p in item.get("points") or []],
                chapter=str(item.get("chapter") or ""),
                doc=str(item.get("doc") or ""),
                pages=[int(p) for p in item.get("pages") or []],
                level=int(item.get("level") or 1),
                type=str(item.get("type") or "concept"),
                review=str(item.get("review") or ""),
            )
        )
        if bank[-1].type not in TYPE_ORDER:
            raise AssessmentError(
                f"第 {index} 题的 type 只能是 concept/calculation/comprehensive"
            )
    for question_type, need in TYPE_PLAN:
        available = sum(1 for item in bank if item.type == question_type)
        if available < need:
            raise AssessmentError(
                f"题库里 {question_type} 类型只有 {available} 道，至少需要 {need} 道"
            )
    return bank


def pick_questions(bank: Sequence[Question], seed: str, count: int = 5) -> list[Question]:
    """按「2 概念 + 2 计算 + 1 综合」抽题，并尽量分散到不同章节。

    同一个匿名代号每次抽到的题固定（seed 就是匿名代号），
    这样学生刷新页面不会换一套题，也便于复盘。
    """
    if len(bank) < count:
        return list(bank)
    rng = random.Random(str(seed))

    by_type: dict[str, list[Question]] = {}
    for item in bank:
        by_type.setdefault(item.type, []).append(item)

    used_chapters: set[str] = set()
    picked: list[Question] = []
    for question_type, need in TYPE_PLAN:
        pool = list(by_type.get(question_type, []))
        rng.shuffle(pool)
        chosen = 0
        # 先挑不同章节的
        for item in pool:
            if chosen >= need:
                break
            if item.chapter in used_chapters:
                continue
            picked.append(item)
            used_chapters.add(item.chapter)
            chosen += 1
        # 章节不够用时放宽限制
        for item in pool:
            if chosen >= need:
                break
            if item in picked:
                continue
            picked.append(item)
            used_chapters.add(item.chapter)
            chosen += 1

    picked.sort(key=lambda item: (TYPE_ORDER.get(item.type, 9), item.level))
    return picked[:count]


def _parse_grade(text: str) -> Grade:
    body = text.strip()
    body = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", body)
    start, end = body.find("{"), body.rfind("}")
    if start >= 0 and end > start:
        body = body[start : end + 1]
    payload = json.loads(body)
    score = float(payload.get("score", 0))
    score = min((0.0, 0.5, 1.0), key=lambda value: abs(value - score))
    return Grade(
        score=score,
        feedback=str(payload.get("feedback") or "").strip()[:120],
        weak_point=str(payload.get("weak_point") or "").strip()[:80],
    )


def grade_answer(
    settings: Settings,
    client: OpenAICompatClient,
    item: Question,
    student_answer: str,
    *,
    course_name: str,
) -> Grade:
    system = settings.prompt("assessment_grade", "system").format(course_name=course_name)
    user = settings.prompt("assessment_grade", "user").format(
        question=item.question,
        answer=item.answer,
        points="；".join(item.points) or item.answer,
        student_answer=student_answer.strip(),
    )
    for _ in range(2):
        try:
            text = client.chat(system=system, user=user, max_tokens=500)
            return _parse_grade(text)
        except (LLMError, ValueError, TypeError):
            # 偶尔会被截断或返回非 JSON，重试一次；两次都失败才降级
            continue
    # 批改失败不阻断测评：按「部分掌握」记录，继续下一题
    return Grade(
        score=0.5,
        feedback="这道题我暂时没能判断清楚，先按「部分掌握」记录，我们继续下一题。",
        weak_point=f"{item.chapter}：{item.question[:20]}",
    )


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


DISCLAIMER = "AI 诊断结果仅供辅导同学参考，最终学情以人工沟通为准"


def _diagnose(concept: float, calc: float, comp: float, total: float, chapters: list[str]) -> tuple[str, str]:
    """根据三类题型的得分，判断问题出在哪一层。"""
    if concept <= 0.5:
        return (
            "基本概念模糊",
            "该生基本概念模糊，建议从定义和物理意义讲起，先做概念辨析，不宜直接刷难题。",
        )
    if total >= 4:
        return (
            "基础较扎实",
            "该生基础较扎实，可以直接进入章节答疑和典型例题，不必做完整摸底。",
        )
    if calc <= 1.0:
        return (
            "概念基本掌握，但计算能力不足",
            "该生概念能听懂，但落到计算容易出错；建议先练等效化简、分压分流、时间常数等基础计算，强调步骤和单位。",
        )
    if comp < 1:
        return (
            "基础题掌握较好，综合应用需要训练",
            "该生基础题掌握较好，综合题缺少拆解思路；建议带着画图、列方程、分步求解，不必从头补概念。",
        )
    return (
        "部分章节薄弱",
        "该生薄弱点分散在" + "、".join(chapters) + "，建议按章节逐个补，每次只攻一个知识点。",
    )


def summarize(results: Iterable[tuple[Question, Grade]]) -> dict[str, object]:
    items = list(results)
    total = sum(grade.score for _, grade in items)
    if total >= 4:
        level, label = "strong", "基础较扎实"
    elif total >= 2.5:
        level, label = "medium", "掌握部分核心概念，仍有薄弱点"
    else:
        level, label = "weak", "需要优先巩固基础知识"

    detail: list[dict[str, object]] = []
    for index, (item, grade) in enumerate(items, start=1):
        detail.append(
            {
                "index": index,
                "id": item.id,
                "type": item.type,
                "type_label": TYPE_LABELS.get(item.type, item.type),
                "chapter": item.chapter,
                "question": item.question,
                "score": grade.score,
                "correct": grade.score >= 1,
                "review": item.review,
                "weak_point": grade.weak_point,
            }
        )

    correct_ids = [int(row["index"]) for row in detail if row["correct"]]
    wrong_ids = [int(row["index"]) for row in detail if not row["correct"]]
    weak_chapters = _unique(
        item.chapter for item, grade in items if grade.score < 1
    )
    review_points = _unique(
        item.review for item, grade in items if grade.score < 1 and item.review
    )

    concept = sum(grade.score for item, grade in items if item.type == "concept")
    calc = sum(grade.score for item, grade in items if item.type == "calculation")
    comp = sum(grade.score for item, grade in items if item.type == "comprehensive")
    diagnosis, tutor_note = _diagnose(concept, calc, comp, total, weak_chapters)

    def numbers(values: list[int]) -> str:
        return "、".join(f"第 {value} 题" for value in values) if values else "无"

    report_lines = [
        DISCLAIMER,
        "",
        f"【成绩】{total:g} / 5（{label}）",
        f"【答对】{numbers(correct_ids)}",
        f"【答错】{numbers(wrong_ids)}",
        "【薄弱章节】" + ("；".join(weak_chapters) if weak_chapters else "无"),
        "【优先复习】" + ("；".join(review_points) if review_points else "无"),
        f"【给辅导学长】{tutor_note}",
    ]

    strengths_text = (
        "已经掌握：" + "、".join(_unique(item.chapter for item, g in items if g.score >= 1)) + "。"
        if any(grade.score >= 1 for _, grade in items)
        else "这次没有完全答对的题，先从最基础的概念补起。"
    )
    weak_text = (
        "需要再巩固：" + "、".join(weak_chapters) + "。"
        if weak_chapters
        else "没有发现明显薄弱点。"
    )

    return {
        "score": total,
        "max_score": 5,
        "level": level,
        "level_label": label,
        "diagnosis": diagnosis,
        "correct_ids": correct_ids,
        "wrong_ids": wrong_ids,
        "items": detail,
        "weak_chapters": weak_chapters,
        "review_points": review_points,
        "tutor_note": tutor_note,
        "disclaimer": DISCLAIMER,
        "report_text": "\n".join(report_lines),
        "strengths": strengths_text,
        "weak_points": weak_text,
        "weak_point_list": _unique(
            grade.weak_point
            for _, grade in items
            if grade.score < 1 and grade.weak_point and grade.weak_point != "无明显薄弱点"
        ),
    }
