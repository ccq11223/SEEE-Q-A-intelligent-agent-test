"""读取 config/app.yaml。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    max_question_chars: int = 300
    asks_per_hour_per_ip: int = 30
    title: str = "课程答疑"
    intro: str = ""
    disclaimer: str = ""
    placeholder: str = "输入你的问题"
    handoff_hint: str = ""

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        if not path.is_file():
            return cls()
        with path.open("r", encoding="utf-8") as handle:
            data: dict[str, Any] = yaml.safe_load(handle) or {}

        server = data.get("server") or {}
        limits = data.get("limits") or {}
        copy = data.get("copy") or {}
        return cls(
            host=str(server.get("host", "127.0.0.1")),
            port=int(server.get("port", 8000)),
            max_question_chars=int(limits.get("max_question_chars", 300)),
            asks_per_hour_per_ip=int(limits.get("asks_per_hour_per_ip", 30)),
            title=str(copy.get("title", "课程答疑")),
            intro=str(copy.get("intro", "")).strip(),
            disclaimer=str(copy.get("disclaimer", "")).strip(),
            placeholder=str(copy.get("placeholder", "输入你的问题")),
            handoff_hint=str(copy.get("handoff_hint", "")).strip(),
        )
