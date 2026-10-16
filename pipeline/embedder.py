"""步骤 6 的前半段：把切块文本转成向量（写入数据库是 store.py 的事）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

from .chunker import Chunk
from .llm import OpenAICompatClient


class EmbeddingDimError(RuntimeError):
    """向量维度与数据库表定义不一致——这类错必须立刻停下，不能带病入库。"""


def save_embeddings(path: Path, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
    """把向量连同切块的内容指纹一起存盘。

    存指纹是为了**复用**：换存储后端（local → postgres）时需要重新入库，
    但内容一个字都没变，不该为同一批文本再付一次向量化的钱。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"hash": chunk.content and _chunk_hash(chunk), "vector": list(vector)})
        for chunk, vector in zip(chunks, vectors)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _chunk_hash(chunk: Chunk) -> str:
    import hashlib

    return hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()[:32]


def load_cached_embeddings(
    path: Path, chunks: Sequence[Chunk]
) -> tuple[list[list[float]] | None, str]:
    """尝试复用已存盘的向量。返回（向量列表或 None, 说明）。

    两种情况都能复用：

    * 新格式：每行 ``{"hash": ..., "vector": [...]}``，逐条核对内容指纹，最可靠；
    * 旧格式：每行就是一个裸数组。没有指纹可比，只好在条数一致时复用，
      并如实告诉使用者「只按条数核对过」。

    任何一处对不上就返回 None，让调用方老老实实重新向量化——
    宁可多花几分钱，也不能把错位的向量写进知识库。
    """
    if not path.is_file():
        return None, "没有缓存文件"

    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"缓存文件读不出来（{exc}）"

    if len(rows) != len(chunks):
        return None, f"缓存有 {len(rows)} 条，当前切块 {len(chunks)} 条，对不上"

    vectors: list[list[float]] = []
    for row, chunk in zip(rows, chunks):
        if isinstance(row, dict):
            if row.get("hash") != _chunk_hash(chunk):
                return None, f"第 {len(vectors) + 1} 条的切块内容变过了"
            vectors.append(list(row["vector"]))
        elif isinstance(row, list):
            vectors.append(row)
        else:
            return None, "缓存文件格式不认识"

    if rows and isinstance(rows[0], list):
        return vectors, f"复用 {len(vectors)} 条已有向量（旧格式，只按条数核对过内容）"
    return vectors, f"复用 {len(vectors)} 条已有向量（已逐条核对内容指纹）"


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
