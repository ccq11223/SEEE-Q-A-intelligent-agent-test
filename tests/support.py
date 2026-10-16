"""测试用替身：假模型客户端、假知识库与临时配置。

原则：单元测试一律不联网、不花钱、不碰真实 data/ 目录。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pipeline.settings import Settings
from pipeline.store import Hit, Store


@dataclass
class FakeModelConfig:
    name: str = "fake-model"
    base_url: str = "http://fake.local/v1"
    api_key_env: str = "FAKE_API_KEY"


class FakeEmbedder:
    """按关键词给出确定性向量，用来构造「命中」与「未命中」两种情形。"""

    keywords = ("节点", "KCL", "戴维南", "正弦")

    def __init__(self, name: str = "fake-embed") -> None:
        self.config = FakeModelConfig(name=name)
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * len(FakeEmbedder.keywords) + [0.1]
        for index, keyword in enumerate(FakeEmbedder.keywords):
            if keyword in text:
                vector[index] = 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def close(self) -> None:
        pass


class FakeGenerator:
    """按顺序返回预设回答，用来测引用校验与重试。"""

    def __init__(self, replies: Sequence[str] | None = None) -> None:
        self.config = FakeModelConfig(name="fake-gen")
        self.replies = list(replies or ["这是一个默认回答（第 1 页）。"])
        self.prompts: list[str] = []

    def chat(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        self.prompts.append(user)
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0]

    def close(self) -> None:
        pass


class FakeStore(Store):
    backend = "fake"

    def __init__(self, hits: Sequence[Hit] | None = None) -> None:
        self.hits = list(hits or [])
        self.qa_logs: list[dict] = []
        self.gaps: list[str] = []

    def search(self, course_id, query_vector, *, limit=20, max_trust=3) -> list[Hit]:
        return self.hits[:limit]

    def log_qa(self, payload) -> None:
        self.qa_logs.append(payload)

    def record_gap(self, course_id, question) -> None:
        self.gaps.append(question)


def make_settings(root: Path, **overrides) -> Settings:
    """构造一个完全指向临时目录的 Settings，避免污染真实 data/。"""
    pipeline = {
        "courses": [{"id": "c1", "name": "电路理论（上）", "term": "2026秋"}],
        "default_course_id": "c1",
        "paths": {
            "data_dir": "data",
            "raw_dir": "data/raw",
            "work_dir": "data/work",
            "cache_dir": "data/cache",
            "out_dir": "data/out",
        },
        "storage": {"backend": "local"},
        "parsing": {"mode": "vision", "concurrency": 2, "retry": 1, "render_max_pixels": 1200},
        "chunking": {
            "target_chars": 200,
            "max_chars": 320,
            "min_chars": 80,
            "overlap_chars": 0,
            "flush_on_section_change": True,
        },
        "embedding": {"batch_size": 4},
    }
    pipeline.update(overrides.pop("pipeline", {}))

    models = {
        "vision": {"name": "fake-vision", "base_url": "http://fake/v1", "api_key_env": "FAKE"},
        "embedding": {
            "name": "fake-embed",
            "base_url": "http://fake/v1",
            "api_key_env": "FAKE",
            "dim": 5,
        },
        "generate": {"name": "fake-gen", "base_url": "http://fake/v1", "api_key_env": "FAKE"},
    }
    models.update(overrides.pop("models", {}))

    retrieval = {
        "retrieve_top_k": 5,
        "rerank_top_k": 3,
        "min_score": 0.3,
        "require_citation": True,
        "allow_web": False,
        "max_context_chars": 4000,
        "max_trust_level": 2,
        "handoff_patterns": ["综合题"],
    }
    retrieval.update(overrides.pop("retrieval", {}))

    prompts = {
        "version": "test",
        "page_parse": {
            "system": "系统提示",
            "user": "{course_name}/{doc_title}/{page_no}/{page_count}",
        },
        "qa_answer": {"system": "《{course_name}》助教", "user": "{context}\n---\n{question}"},
        "handoff_message": "这个知识点现有资料里没有覆盖，建议找学长学姐线下确认。",
    }
    prompts.update(overrides.pop("prompts", {}))

    settings = Settings(
        root=root,
        pipeline=pipeline,
        models=models,
        retrieval=retrieval,
        prompts=prompts,
    )
    settings.paths.ensure()
    return settings
