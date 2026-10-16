"""配置加载：把 config/*.yaml 与 .env 汇成一个 Settings 对象。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .env import load_env_file


class ConfigError(RuntimeError):
    """配置缺失或写错时抛出的错误，消息要求能直接指导使用者怎么改。"""


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path
    raw_dir: Path
    work_dir: Path
    cache_dir: Path
    out_dir: Path

    def doc_work(self, doc_key: str) -> Path:
        return self.work_dir / doc_key

    def doc_pages(self, doc_key: str) -> Path:
        return self.doc_work(doc_key) / "pages"

    def doc_markdown(self, doc_key: str) -> Path:
        return self.doc_work(doc_key) / "doc.md"

    def doc_chunks(self, doc_key: str) -> Path:
        return self.doc_work(doc_key) / "chunks.jsonl"

    def doc_embeddings(self, doc_key: str) -> Path:
        return self.doc_work(doc_key) / "embeddings.jsonl"

    def vision_cache(self) -> Path:
        return self.cache_dir / "vision"

    def ensure(self) -> None:
        for path in (
            self.data_dir,
            self.raw_dir,
            self.work_dir,
            self.cache_dir,
            self.out_dir,
            self.vision_cache(),
        ):
            path.mkdir(parents=True, exist_ok=True)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"找不到配置文件：{path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是键值对：{path}")
    return data


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    root: Path
    pipeline: dict[str, Any]
    models: dict[str, Any]
    retrieval: dict[str, Any]
    prompts: dict[str, Any]
    env_loaded: int = 0

    # ---------- 路径 ----------
    @property
    def paths(self) -> Paths:
        raw = self.pipeline.get("paths") or {}
        root = self.root

        def resolve(key: str, default: str) -> Path:
            value = Path(str(raw.get(key, default)))
            return value if value.is_absolute() else (root / value)

        data_dir = resolve("data_dir", "data")
        return Paths(
            root=root,
            data_dir=data_dir,
            raw_dir=resolve("raw_dir", "data/raw"),
            work_dir=resolve("work_dir", "data/work"),
            cache_dir=resolve("cache_dir", "data/cache"),
            out_dir=resolve("out_dir", "data/out"),
        )

    # ---------- 课程 ----------
    @property
    def courses(self) -> list[dict[str, Any]]:
        return list(self.pipeline.get("courses") or [])

    @property
    def default_course_id(self) -> str:
        course_id = self.pipeline.get("default_course_id")
        if not course_id:
            raise ConfigError("config/pipeline.yaml 里缺少 default_course_id")
        return str(course_id)

    def course_name(self, course_id: str) -> str:
        for course in self.courses:
            if course.get("id") == course_id:
                return str(course.get("name") or course_id)
        return course_id

    def require_course(self, course_id: str) -> str:
        if not any(course.get("id") == course_id for course in self.courses):
            known = ", ".join(str(c.get("id")) for c in self.courses) or "（无）"
            raise ConfigError(
                f"课程 {course_id!r} 未在 config/pipeline.yaml 的 courses 中登记。已登记：{known}"
            )
        return course_id

    # ---------- 分区访问器 ----------
    @property
    def parsing(self) -> dict[str, Any]:
        return dict(self.pipeline.get("parsing") or {})

    @property
    def chunking(self) -> dict[str, Any]:
        return dict(self.pipeline.get("chunking") or {})

    @property
    def storage(self) -> dict[str, Any]:
        return dict(self.pipeline.get("storage") or {})

    @property
    def embedding(self) -> dict[str, Any]:
        return {**dict(self.pipeline.get("embedding") or {}), **dict(self.models.get("embedding") or {})}

    def model_config(self, role: str) -> dict[str, Any]:
        cfg = self.models.get(role)
        if not isinstance(cfg, dict):
            raise ConfigError(f"config/models.yaml 里缺少 {role} 段落")
        for required in ("name", "base_url", "api_key_env"):
            if not cfg.get(required):
                raise ConfigError(f"config/models.yaml 的 {role} 段缺少 {required}")
        return cfg

    @property
    def prompts_version(self) -> str:
        return str(self.prompts.get("version") or "unversioned")

    def prompt(self, *keys: str) -> str:
        node: Any = self.prompts
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                raise ConfigError(f"config/prompts.yaml 里找不到 {' > '.join(keys)}")
            node = node[key]
        return str(node).rstrip("\n")

    # ---------- 加载 ----------
    @classmethod
    def load(cls, root: Path | str | None = None) -> "Settings":
        root_path = Path(root).resolve() if root else project_root()
        env_loaded = load_env_file(root_path / ".env")
        config_dir = root_path / "config"
        settings = cls(
            root=root_path,
            pipeline=_read_yaml(config_dir / "pipeline.yaml"),
            models=_read_yaml(config_dir / "models.yaml"),
            retrieval=_read_yaml(config_dir / "retrieval.yaml"),
            prompts=_read_yaml(config_dir / "prompts.yaml"),
            env_loaded=env_loaded,
        )
        settings.paths.ensure()
        return settings
