"""把流水线里的组件组装成在线服务需要的依赖。

刻意**不重新实现**检索和答疑逻辑：直接用 pipeline/retrieval.py 与
pipeline/answer.py。这样「建库自检时怎么判、学生用时也怎么判」，
不会出现两套标准。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline.llm import ModelConfig, OpenAICompatClient
from pipeline.settings import ConfigError, Settings
from pipeline.store import Store, open_store


@dataclass
class Services:
    settings: Settings
    store: Store
    embed: OpenAICompatClient
    generate: OpenAICompatClient
    rerank: OpenAICompatClient | None = None

    @property
    def course_id(self) -> str:
        return self.settings.default_course_id

    @property
    def course_name(self) -> str:
        return self.settings.course_name(self.course_id)

    def close(self) -> None:
        for client in (self.embed, self.generate, self.rerank):
            if client is not None:
                client.close()
        self.store.close()


def build_services(root: Path | None = None) -> Services:
    settings = Settings.load(root)
    store = open_store(settings)
    embed = OpenAICompatClient(
        ModelConfig.from_dict("embedding", settings.model_config("embedding"))
    )
    generate = OpenAICompatClient(
        ModelConfig.from_dict("generate", settings.model_config("generate"))
    )

    rerank: OpenAICompatClient | None = None
    try:
        rerank = OpenAICompatClient(
            ModelConfig.from_dict("rerank", settings.model_config("rerank"))
        )
    except ConfigError:
        # 没配重排就退回纯向量，功能不中断（但「有没有依据」的判定会松很多）
        rerank = None

    return Services(
        settings=settings,
        store=store,
        embed=embed,
        generate=generate,
        rerank=rerank,
    )
