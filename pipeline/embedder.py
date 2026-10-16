"""步骤 6 的前半段：把切块文本转成向量（写入数据库是 store.py 的事）。"""

from __future__ import annotations

from typing import Callable, Sequence

from .chunker import Chunk
from .llm import OpenAICompatClient


class EmbeddingDimError(RuntimeError):
    """向量维度与数据库表定义不一致——这类错必须立刻停下，不能带病入库。"""


def embed_chunks(
    chunks: Sequence[Chunk],
    client: OpenAICompatClient,
    *,
    batch_size: int = 32,
    expected_dim: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    texts = [chunk.content for chunk in chunks]
    if not texts:
        return []

    vectors: list[list[float]] = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        vectors.extend(client.embed(batch))
        if progress:
            progress(min(offset + len(batch), len(texts)), len(texts))

    if expected_dim:
        for index, vector in enumerate(vectors):
            if len(vector) != expected_dim:
                raise EmbeddingDimError(
                    f"第 {index + 1} 条向量维度是 {len(vector)}，但配置里写的是 {expected_dim}。\n"
                    f"向量模型：{client.config.name}。"
                    f"请把 config/models.yaml 的 embedding.dim 与实际模型对齐，"
                    f"并同步修改 db/migrations/001_init.sql 里 chunks.embedding 的 vector(N)。"
                )
    return vectors
