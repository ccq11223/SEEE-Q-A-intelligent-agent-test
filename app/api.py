"""学生端接口：配置、健康检查、提问。"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline import websearch
from pipeline.answer import answer_question
from pipeline.assessment import (
    AssessmentError,
    grade_answer,
    load_bank,
    pick_questions,
    summarize,
)
from pipeline.llm import LLMError

router = APIRouter()

#: 匿名代号只允许字母数字下划线连字符，避免有人填学号姓名
ANON_CODE_RE = re.compile(r"^[A-Za-z0-9_\-]{2,16}$")
#: 资料编号只允许字母数字下划线连字符，防止用 ../ 之类的路径越界
DOC_KEY_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _doc_titles(services) -> dict[str, str]:
    """资料编号 → 标题，来自 config/manifest.csv。"""
    import csv

    path = services.settings.root / "config" / "manifest.csv"
    titles: dict[str, str] = {}
    if not path.is_file():
        return titles
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("doc_key") or "").strip()
            if key:
                titles[key] = (row.get("title") or "").strip()
    return titles


def _image_label(services, doc: str, page: int) -> str:
    title = _doc_titles(services).get(doc, "")
    return f"《{title}》第 {page} 页" if title else f"课件第 {page} 页"


class ChatTurn(BaseModel):
    role: str = "user"
    content: str = ""


class AskRequest(BaseModel):
    anon_code: str = ""
    question: str = ""
    #: 最近几轮对话，由前端带上；服务端不留历史，避免隐私问题
    history: list[ChatTurn] = []
    #: 测评得出的水平与薄弱点，用来调整讲解深度
    level: str = ""
    weak_points: list[str] = []


class AssessStartRequest(BaseModel):
    anon_code: str = ""


class AssessAnswerRequest(BaseModel):
    anon_code: str = ""
    answer: str = ""


class ImageRef(BaseModel):
    doc: str
    page: int
    url: str
    label: str


class AskResponse(BaseModel):
    mode: str
    #: 依据等级：kb / web / general / handoff。前端据此显示不同颜色的标签
    source_level: str = "kb"
    source_label: str = ""
    answer: str
    citations: list[int] = []
    sources: list[str] = []
    images: list[ImageRef] = []
    handoff: bool = False
    latency_ms: int = 0
    top_score: float = 0.0


@router.get("/api/config")
def get_config(request: Request) -> dict[str, object]:
    cfg = request.app.state.config
    services = request.app.state.services
    return {
        "title": cfg.title,
        "intro": cfg.intro,
        "disclaimer": cfg.disclaimer,
        "placeholder": cfg.placeholder,
        "handoff_hint": cfg.handoff_hint,
        "max_question_chars": cfg.max_question_chars,
        "course": services.course_name,
    }


@router.get("/api/page-image/{doc_key}/{page_no}")
def page_image(doc_key: str, page_no: int, request: Request) -> FileResponse:
    """返回课件某一页的原图，给回答里「带出知识库图片」用。"""
    if not DOC_KEY_RE.match(doc_key):
        raise HTTPException(status_code=400, detail="非法资料编号")
    if not (1 <= page_no <= 99999):
        raise HTTPException(status_code=400, detail="非法页码")

    services = request.app.state.services
    pages_dir = Path(services.settings.paths.doc_pages(doc_key))
    base = pages_dir / f"{doc_key}_page_{page_no:04d}"
    image_path = base.with_suffix(".jpg")
    if not image_path.is_file():
        image_path = base.with_suffix(".png")
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="找不到这一页的课件图片")
    return FileResponse(image_path, media_type="image/jpeg" if image_path.suffix == ".jpg" else "image/png")


@router.get("/api/health")
def health(request: Request) -> dict[str, object]:
    services = request.app.state.services
    return {
        "ok": True,
        "course": services.course_name,
        "backend": services.store.backend,
        "generate_model": services.generate.config.name,
        "rerank_enabled": services.rerank is not None,
        "web_search": websearch.configured_provider() or "off",
    }


@router.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    cfg = request.app.state.config
    services = request.app.state.services

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if len(question) > cfg.max_question_chars:
        raise HTTPException(
            status_code=400,
            detail=f"问题太长了，请控制在 {cfg.max_question_chars} 字以内",
        )

    anon_code = payload.anon_code.strip()
    if not ANON_CODE_RE.match(anon_code):
        raise HTTPException(
            status_code=400,
            detail="请先填一个匿名代号（2-16 位字母、数字、下划线或连字符，不要填姓名学号）",
        )

    client_ip = request.client.host if request.client else "unknown"
    if not request.app.state.limiter.allow(client_ip):
        raise HTTPException(
            status_code=429,
            detail="提问太频繁了，休息一下再来（每小时有次数上限）",
        )

    session_id = services.store.ensure_session(services.course_id, anon_code)

    history = [
        {"role": turn.role, "content": turn.content[:600]}
        for turn in payload.history[-6:]
        if turn.role in ("user", "assistant") and turn.content.strip()
    ]

    try:
        result = answer_question(
            services.settings,
            services.store,
            services.embed,
            services.generate,
            question,
            course_id=services.course_id,
            rerank_client=services.rerank,
            session_id=session_id,
            history=history,
            level=payload.level,
            weak_points=payload.weak_points[:5],
        )
    except LLMError as exc:
        # 不把模型报错原文甩给学生；日志里能查到
        raise HTTPException(
            status_code=503,
            detail="模型服务暂时不可用，请稍后再试，或直接找学长学姐",
        ) from exc

    if result.failed:
        # answer_question 在模型故障时会降级成「资料未覆盖」，但对学生要说实话
        raise HTTPException(
            status_code=503,
            detail="模型服务暂时不可用，请稍后再试，或直接找学长学姐",
        )

    images = [
        ImageRef(
            doc=doc,
            page=page,
            url=f"/api/page-image/{doc}/{page}",
            label=_image_label(services, doc, page),
        )
        for doc, page in result.image_pages
    ]

    return AskResponse(
        mode=result.mode,
        source_level=result.source_level,
        source_label=result.source_label,
        answer=result.text,
        citations=result.citations,
        sources=result.sources,
        images=images,
        handoff=result.mode == "handoff",
        latency_ms=result.latency_ms,
        top_score=round(result.top_score, 4),
    )


# ---------------------------------------------------------------------------
# 首次使用的基础测评
# ---------------------------------------------------------------------------

ASSESS_TOTAL = 5


def _question_payload(item, index: int, total: int) -> dict[str, object]:
    return {
        "id": item.id,
        "index": index,
        "total": total,
        "question": item.question,
        "chapter": item.chapter,
    }


@router.post("/api/assess/start")
def assess_start(payload: AssessStartRequest, request: Request) -> dict[str, object]:
    anon = payload.anon_code.strip()
    if not ANON_CODE_RE.match(anon):
        raise HTTPException(
            status_code=400,
            detail="请先填一个匿名代号（2-16 位字母、数字、下划线或连字符）",
        )

    services = request.app.state.services
    try:
        bank = load_bank(services.settings.root)
    except AssessmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    questions = pick_questions(bank, seed=anon, count=ASSESS_TOTAL)
    request.app.state.assessments[anon] = {
        "questions": questions,
        "index": 0,
        "results": [],
    }
    return {
        "total": len(questions),
        "question": _question_payload(questions[0], 1, len(questions)),
        "intro": (
            "先做 5 道基础概念题，我了解一下你的掌握情况，后面的讲解会更有针对性。"
            "一次只问一题，答错也没关系，不会计入任何成绩。"
        ),
    }


@router.post("/api/assess/answer")
def assess_answer(payload: AssessAnswerRequest, request: Request) -> dict[str, object]:
    anon = payload.anon_code.strip()
    answer = payload.answer.strip()
    if not ANON_CODE_RE.match(anon):
        raise HTTPException(status_code=400, detail="匿名代号不合法")
    if not answer:
        raise HTTPException(status_code=400, detail="先写一下你的想法吧")

    state = request.app.state.assessments.get(anon)
    if not state:
        raise HTTPException(status_code=400, detail="测评还没开始，请先点「开始测评」")
    index = int(state["index"])
    questions = state["questions"]
    if index >= len(questions):
        raise HTTPException(status_code=400, detail="这次测评已经结束了")

    services = request.app.state.services
    item = questions[index]
    grade = grade_answer(
        services.settings,
        services.generate,
        item,
        answer,
        course_name=services.course_name,
    )
    state["results"].append((item, grade))
    state["index"] = index + 1
    total = len(questions)

    next_question = None
    summary = None
    if state["index"] < total:
        next_question = _question_payload(
            questions[state["index"]], state["index"] + 1, total
        )
    else:
        summary = summarize(state["results"])
        request.app.state.assessments.pop(anon, None)

    return {
        "score": grade.score,
        "feedback": grade.feedback,
        "index": index + 1,
        "total": total,
        "next": next_question,
        "summary": summary,
    }
