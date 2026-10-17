"""学生端接口：配置、健康检查、提问。"""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from pipeline.answer import answer_question
from pipeline.llm import LLMError

router = APIRouter()

#: 匿名代号只允许字母数字下划线连字符，避免有人填学号姓名
ANON_CODE_RE = re.compile(r"^[A-Za-z0-9_\-]{2,16}$")


class AskRequest(BaseModel):
    anon_code: str = ""
    question: str = ""


class AskResponse(BaseModel):
    mode: str
    answer: str
    citations: list[int] = []
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


@router.get("/api/health")
def health(request: Request) -> dict[str, object]:
    services = request.app.state.services
    return {
        "ok": True,
        "course": services.course_name,
        "backend": services.store.backend,
        "generate_model": services.generate.config.name,
        "rerank_enabled": services.rerank is not None,
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

    return AskResponse(
        mode=result.mode,
        answer=result.text,
        citations=result.citations,
        handoff=result.mode == "handoff",
        latency_ms=result.latency_ms,
        top_score=round(result.top_score, 4),
    )
