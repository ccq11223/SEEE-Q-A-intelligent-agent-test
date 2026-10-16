"""极简 .env 读取。

不引入 python-dotenv：规则简单到不值得多一个依赖，而依赖越少，
别人接手时出问题的概率越低。
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path) -> int:
    """把 .env 里的键值对写进 os.environ，返回加载条数。

    已存在的环境变量优先（方便临时用 shell 覆盖），文件中后出现的定义覆盖先出现的。
    """
    if not path.is_file():
        return 0

    loaded = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        if key not in os.environ:
            os.environ[key] = value
            loaded += 1

    return loaded


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def require_env(name: str, *, purpose: str) -> str:
    value = get_env(name)
    if not value:
        raise RuntimeError(
            f"缺少环境变量 {name}（用途：{purpose}）。\n"
            f"请把 .env.example 复制为 .env，填入该值后重试。"
        )
    return value
