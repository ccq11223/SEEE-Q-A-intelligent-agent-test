"""FastAPI 应用入口。

启动：
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

试点期对外暴露用 Cloudflare Tunnel（方案 §6.2），不需要备案。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.config import AppConfig
from app.ratelimit import HourlyLimiter
from app.services import Services, build_services

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent


def create_app(
    services: Services | None = None,
    config: AppConfig | None = None,
) -> FastAPI:
    app_config = config or AppConfig.load(PROJECT_ROOT / "config" / "app.yaml")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_services = services is None
        app.state.services = services or build_services(PROJECT_ROOT)
        app.state.owns_services = owns_services
        try:
            yield
        finally:
            if owns_services:
                app.state.services.close()

    app = FastAPI(
        title=app_config.title,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.config = app_config
    app.state.limiter = HourlyLimiter(app_config.asks_per_hour_per_ip)
    #: 测评进行中的状态，只放内存：服务重启后学生重新开始即可
    app.state.assessments = {}
    app.include_router(router)

    # 静态页面挂在最后：先匹配 /api/*，其余都当静态文件
    app.mount(
        "/",
        StaticFiles(directory=APP_DIR / "web", html=True),
        name="web",
    )
    return app


app = create_app()
