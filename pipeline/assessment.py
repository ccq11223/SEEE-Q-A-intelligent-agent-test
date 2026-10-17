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


@dataclass
class Grade:
    score: float
    feedback: str
    weak_point: str = ""


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
            )
        )
    return bank


def pick_questions(bank: Sequence[Question], seed: str, count: int = 5) -> list[Question]:
    """按章节分散、难度由浅入深地抽题。

    同一个匿名代号每次抽到的题固定（seed 就是匿名代号），
    这样学生刷新页面不会换一套题，也便于复盘。
    """
    if len(bank) < count:
        return list(bank)
    rng = random.Random(str(seed))

    by_chapter: dict[str, list[Question]] = {}
    for item in bank:
        by_chapter.setdefault(item.chapter, []).append(item)

    chapters = sorted(by_chapter)
    rng.shuffle(chapters)

    picked: list[Question] = []
    for chapter in chapters:
        if len(picked) >= count:
            break
        picked.append(rng.choice(by_chapter[chapter]))

    if len(picked) < count:
        rest = [item for item in bank if item not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: count - len(picked)])

    picked.sort(key=lambda item: item.level)
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
    try:
        text = client.chat(system=system, user=user, max_tokens=300)
        return _parse_grade(text)
    except (LLMError, ValueError, TypeError):
        # 批改失败不阻断测评：按「部分掌握」记录，继续下一题
        return Grade(
            score=0.5,
            feedback="这道题我暂时没能判断清楚，先按「部分掌握」记录，我们继续下一题。",
            weak_point=f"{item.chapter}：{item.question[:20]}",
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

    strengths = []
    weak_points = []
    for item, grade in items:
        if grade.score >= 1:
            strengths.append(item.chapter)
        elif grade.weak_point and grade.weak_point != "无明显薄弱点":
            weak_points.append(grade.weak_point)

    if not strengths:
        strengths_text = "这次没有完全答对的题，先从最基础的概念补起。"
    else:
        strengths_text = "已经掌握：" + "、".join(dict.fromkeys(strengths)) + "。"
    if not weak_points:
        weak_text = "没有发现明显薄弱点。"
    else:
        weak_text = "需要再巩固：" + "、".join(dict.fromkeys(weak_points)) + "。"

    return {
        "score": total,
        "max_score": 5,
        "level": level,
        "level_label": label,
        "strengths": strengths_text,
        "weak_points": weak_text,
        "weak_point_list": list(dict.fromkeys(weak_points)),
    }
