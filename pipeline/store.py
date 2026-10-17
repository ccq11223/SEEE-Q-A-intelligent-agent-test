"""步骤 6 的后半段：把切块与向量写进知识库。

两种后端：
  - ``postgres``：云端 Supabase / 自建 PostgreSQL + pgvector，正式形态；
  - ``local``   ：只写 JSONL，不需要数据库。用来在没申请到云账号前先把整条链路跑通，
                  也用于离线回归测试。

两者实现同一套接口，上层（检索、答疑）不区分。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .chunker import Chunk
from .manifest import ManifestEntry
from .settings import Settings


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ChunkRecord:
    chunk_key: str
    page_no: int | None
    page_end: int | None
    chapter: str
    section_title: str
    content: str
    trust_level: int
    char_count: int
    token_count: int
    content_hash: str = ""
    embedding: list[float] = field(default_factory=list)

    @classmethod
    def from_chunk(
        cls, chunk: Chunk, *, trust_level: int, embedding: Sequence[float] | None = None
    ) -> "ChunkRecord":
        return cls(
            chunk_key=chunk.chunk_key,
            page_no=chunk.page_no,
            page_end=chunk.page_end,
            chapter=chunk.chapter,
            section_title=chunk.section_title,
            content=chunk.content,
            trust_level=trust_level,
            char_count=chunk.char_count,
            token_count=chunk.token_count,
            content_hash=content_hash(chunk.content),
            embedding=list(embedding or []),
        )


@dataclass
class DocumentRecord:
    course_id: str
    doc_key: str
    filename: str
    doc_type: str
    trust_level: int
    title: str
    term: str = ""
    source_note: str = ""
    status: str = "registered"
    page_count: int | None = None
    chunk_count: int = 0
    parse_error_count: int = 0
    content_hash: str = ""

    @classmethod
    def from_entry(cls, entry: ManifestEntry, **overrides: Any) -> "DocumentRecord":
        record = cls(
            course_id=entry.course_id,
            doc_key=entry.doc_key,
            filename=entry.filename,
            doc_type=entry.doc_type,
            trust_level=entry.trust_level,
            title=entry.title,
            term=entry.term,
            source_note=entry.notes,
        )
        for key, value in overrides.items():
            if not hasattr(record, key):
                raise TypeError(f"DocumentRecord 没有字段 {key!r}")
            setattr(record, key, value)
        return record


@dataclass
class Hit:
    chunk_key: str
    page_no: int | None
    page_end: int | None
    chapter: str
    section_title: str
    content: str
    trust_level: int
    score: float
    document_id: int | None = None
    chunk_id: int | None = None
    #: 向量召回时的分数。重排之后 score 会被替换成重排分，原始分数留在这里备查。
    vector_score: float = 0.0

    @property
    def page_label(self) -> str:
        if self.page_no is None:
            return "页码未知"
        if self.page_end and self.page_end != self.page_no:
            return f"第 {self.page_no}-{self.page_end} 页"
        return f"第 {self.page_no} 页"


class Store:
    """两种后端的公共接口。"""

    backend = "base"

    def upsert_document(self, record: DocumentRecord) -> int | None:  # pragma: no cover - 接口
        raise NotImplementedError

    def replace_chunks(
        self,
        record: DocumentRecord,
        chunks: Sequence[ChunkRecord],
    ) -> int:  # pragma: no cover - 接口
        raise NotImplementedError

    def search(
        self,
        course_id: str,
        query_vector: Sequence[float],
        *,
        limit: int = 20,
        max_trust: int = 3,
    ) -> list[Hit]:  # pragma: no cover - 接口
        raise NotImplementedError

    def log_qa(self, payload: dict[str, Any]) -> None:
        """问答日志（阶段 2 在线服务写入；流水线自检也会写一条）。"""

    def ensure_session(self, course_id: str, anon_code: str) -> int | None:
        """按匿名代号登记/复用一次学生会话，返回会话 id（本地后端返回 None）。

        刻意只存 anon_code：方案 §1.3 明确不收集姓名学号，学情报告也用匿名代号。
        """
        return None

    def record_gap(self, course_id: str, question: str) -> None:
        """知识缺口登记。"""

    def record_parse_error(
        self,
        *,
        course_id: str,
        doc_key: str,
        page_no: int | None,
        error_type: str,
        detail: str = "",
        reporter: str | None = None,
    ) -> None:
        """解析质量问题登记（人工抽查的落点）。"""

    def stats(self, course_id: str) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        """释放连接。"""


# ---------------------------------------------------------------------------
# 本地 JSONL 后端
# ---------------------------------------------------------------------------

def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=str(path.parent), suffix=".tmp"
    )
    try:
        with handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


class LocalStore(Store):
    backend = "local"

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.documents_path = out_dir / "documents.jsonl"
        self.chunks_path = out_dir / "chunks.jsonl"
        self.qa_path = out_dir / "qa_logs.jsonl"
        self.gaps_path = out_dir / "gaps.jsonl"

    def upsert_document(self, record: DocumentRecord) -> int | None:
        rows = [
            row
            for row in _read_jsonl(self.documents_path)
            if not (
                row.get("course_id") == record.course_id
                and row.get("doc_key") == record.doc_key
            )
        ]
        payload = asdict(record)
        payload["updated_at"] = _now()
        rows.append(payload)
        _write_jsonl_atomic(self.documents_path, rows)
        return len(rows)

    def replace_chunks(self, record: DocumentRecord, chunks: Sequence[ChunkRecord]) -> int:
        # 与 PostgresStore 保持一致：写块的同时更新资料台账
        self.upsert_document(record)
        keep = [
            row
            for row in _read_jsonl(self.chunks_path)
            if not (
                row.get("course_id") == record.course_id
                and row.get("doc_key") == record.doc_key
            )
        ]
        for chunk in chunks:
            payload = asdict(chunk)
            payload["course_id"] = record.course_id
            payload["doc_key"] = record.doc_key
            keep.append(payload)
        _write_jsonl_atomic(self.chunks_path, keep)
        return len(chunks)

    def search(
        self,
        course_id: str,
        query_vector: Sequence[float],
        *,
        limit: int = 20,
        max_trust: int = 3,
    ) -> list[Hit]:
        rows = [
            row
            for row in _read_jsonl(self.chunks_path)
            if row.get("course_id") == course_id
            and int(row.get("trust_level", 3)) <= max_trust
            and row.get("embedding")
        ]
        if not rows:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        try:
            import numpy as np

            matrix = np.asarray([row["embedding"] for row in rows], dtype="float32")
            query = np.asarray(list(query_vector), dtype="float32")
            matrix_norm = np.linalg.norm(matrix, axis=1)
            query_norm = float(np.linalg.norm(query)) or 1.0
            sims = (matrix @ query) / (matrix_norm * query_norm + 1e-12)
            for row, score in zip(rows, sims.tolist()):
                scored.append((float(score), row))
        except ImportError:  # 没装 numpy 时的纯 Python 兜底
            query_norm = math.sqrt(sum(value * value for value in query_vector)) or 1.0
            for row in rows:
                vector = row["embedding"]
                dot = sum(a * b for a, b in zip(vector, query_vector))
                norm = math.sqrt(sum(value * value for value in vector)) or 1.0
                scored.append((dot / (norm * query_norm), row))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            Hit(
                chunk_key=row.get("chunk_key", ""),
                page_no=row.get("page_no"),
                page_end=row.get("page_end"),
                chapter=row.get("chapter", ""),
                section_title=row.get("section_title", ""),
                content=row.get("content", ""),
                trust_level=int(row.get("trust_level", 3)),
                score=score,
            )
            for score, row in scored[:limit]
        ]

    def log_qa(self, payload: dict[str, Any]) -> None:
        rows = _read_jsonl(self.qa_path)
        payload = {**payload, "created_at": _now()}
        rows.append(payload)
        _write_jsonl_atomic(self.qa_path, rows)

    def ensure_session(self, course_id: str, anon_code: str) -> int | None:
        # 本地后端没有会话表，匿名代号随日志一起存下来即可
        return None

    def record_gap(self, course_id: str, question: str) -> None:
        rows = _read_jsonl(self.gaps_path)
        for row in rows:
            if row.get("course_id") == course_id and row.get("question") == question:
                row["hit_count"] = int(row.get("hit_count", 1)) + 1
                row["last_seen_at"] = _now()
                break
        else:
            rows.append(
                {
                    "course_id": course_id,
                    "question": question,
                    "hit_count": 1,
                    "status": "open",
                    "first_seen_at": _now(),
                    "last_seen_at": _now(),
                }
            )
        _write_jsonl_atomic(self.gaps_path, rows)

    def record_parse_error(
        self,
        *,
        course_id: str,
        doc_key: str,
        page_no: int | None,
        error_type: str,
        detail: str = "",
        reporter: str | None = None,
    ) -> None:
        path = self.out_dir / "parse_errors.jsonl"
        rows = _read_jsonl(path)
        rows.append(
            {
                "course_id": course_id,
                "doc_key": doc_key,
                "page_no": page_no,
                "error_type": error_type,
                "detail": detail,
                "reporter": reporter,
                "created_at": _now(),
            }
        )
        _write_jsonl_atomic(path, rows)

    def stats(self, course_id: str) -> dict[str, Any]:
        documents = [
            row
            for row in _read_jsonl(self.documents_path)
            if row.get("course_id") == course_id
        ]
        chunks = [
            row for row in _read_jsonl(self.chunks_path) if row.get("course_id") == course_id
        ]
        by_trust: dict[str, int] = {}
        for row in chunks:
            key = str(row.get("trust_level"))
            by_trust[key] = by_trust.get(key, 0) + 1
        return {
            "backend": self.backend,
            "documents": len(documents),
            "chunks": len(chunks),
            "chunks_by_trust": by_trust,
            "doc_list": [
                {
                    "doc_key": row.get("doc_key"),
                    "title": row.get("title"),
                    "status": row.get("status"),
                    "trust_level": row.get("trust_level"),
                    "chunk_count": row.get("chunk_count"),
                }
                for row in sorted(documents, key=lambda item: str(item.get("doc_key")))
            ],
        }


# ---------------------------------------------------------------------------
# PostgreSQL / Supabase 后端
# ---------------------------------------------------------------------------

def vector_literal(values: Sequence[float]) -> str:
    """把向量转成 pgvector 认的字符串字面量。

    刻意不用 pgvector 的 Python 适配器：直接在 SQL 里 ``%s::vector`` 转型，
    少一个依赖、少一层版本兼容问题。
    """
    return "[" + ",".join(f"{float(value):.7g}" for value in values) + "]"


class PostgresStore(Store):
    backend = "postgres"

    def __init__(self, dsn: str, *, schema: str = "public") -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的提示
            raise RuntimeError(
                "未安装 psycopg，无法连接云端数据库。请执行：pip install 'psycopg[binary]'"
            ) from exc

        self.dsn = dsn
        self.schema = schema
        self._connect()

    def _connect(self) -> None:
        import psycopg

        kwargs: dict[str, Any] = {"autocommit": False}
        if self.schema and self.schema != "public":
            kwargs["options"] = f"-c search_path={self.schema}"
        self._conn = psycopg.connect(self.dsn, **kwargs)

    # ---------- 写入 ----------
    def upsert_document(self, record: DocumentRecord) -> int | None:
        sql = """
            insert into documents (
                course_id, doc_key, filename, doc_type, trust_level, title, term,
                source_note, status, page_count, chunk_count, parse_error_count,
                content_hash, updated_at
            ) values (
                %(course_id)s, %(doc_key)s, %(filename)s, %(doc_type)s, %(trust_level)s,
                %(title)s, %(term)s, %(source_note)s, %(status)s, %(page_count)s,
                %(chunk_count)s, %(parse_error_count)s, %(content_hash)s, now()
            )
            on conflict (course_id, doc_key) do update set
                filename = excluded.filename,
                doc_type = excluded.doc_type,
                trust_level = excluded.trust_level,
                title = excluded.title,
                term = excluded.term,
                source_note = excluded.source_note,
                status = excluded.status,
                page_count = excluded.page_count,
                chunk_count = excluded.chunk_count,
                parse_error_count = excluded.parse_error_count,
                content_hash = excluded.content_hash,
                updated_at = now()
            returning id
        """
        payload = asdict(record)
        with self._conn.cursor() as cur:
            cur.execute(sql, payload)
            row = cur.fetchone()
        self._conn.commit()
        return int(row[0]) if row else None

    def replace_chunks(self, record: DocumentRecord, chunks: Sequence[ChunkRecord]) -> int:
        document_id = self.upsert_document(record)
        if document_id is None:
            raise RuntimeError(f"写入 documents 失败：{record.doc_key}")

        with self._conn.cursor() as cur:
            cur.execute("delete from chunks where document_id = %s", (document_id,))

            sql = """
                insert into chunks (
                    course_id, document_id, chunk_key, page_no, page_end, chapter,
                    section_title, content, embedding, trust_level, char_count,
                    token_count, content_hash
                ) values (
                    %(course_id)s, %(document_id)s, %(chunk_key)s, %(page_no)s,
                    %(page_end)s, %(chapter)s, %(section_title)s, %(content)s,
                    %(embedding)s::vector, %(trust_level)s, %(char_count)s,
                    %(token_count)s, %(content_hash)s
                )
            """
            payloads = []
            for chunk in chunks:
                payload = asdict(chunk)
                payload["course_id"] = record.course_id
                payload["document_id"] = document_id
                payload["embedding"] = vector_literal(chunk.embedding)
                payloads.append(payload)

            if payloads:
                cur.executemany(sql, payloads)
        self._conn.commit()
        return len(chunks)

    # ---------- 检索 ----------
    def search(
        self,
        course_id: str,
        query_vector: Sequence[float],
        *,
        limit: int = 20,
        max_trust: int = 3,
    ) -> list[Hit]:
        sql = """
            select id, document_id, chunk_key, page_no, page_end, chapter,
                   section_title, content, trust_level, score
              from match_chunks(%s::vector, %s, %s, %s)
        """
        with self._conn.cursor() as cur:
            cur.execute(
                sql,
                (vector_literal(query_vector), course_id, max_trust, limit),
            )
            rows = cur.fetchall()
        return [
            Hit(
                chunk_id=int(row[0]),
                document_id=int(row[1]),
                chunk_key=row[2],
                page_no=row[3],
                page_end=row[4],
                chapter=row[5] or "",
                section_title=row[6] or "",
                content=row[7],
                trust_level=int(row[8]),
                score=float(row[9]),
            )
            for row in rows
        ]

    # ---------- 日志与缺口 ----------
    def log_qa(self, payload: dict[str, Any]) -> None:
        sql = """
            insert into qa_logs (
                course_id, session_id, question, retrieved_chunk_ids, answer, mode,
                citations, citation_valid, top_score, model, latency_ms
            ) values (
                %(course_id)s, %(session_id)s, %(question)s, %(retrieved_chunk_ids)s,
                %(answer)s, %(mode)s, %(citations)s, %(citation_valid)s,
                %(top_score)s, %(model)s, %(latency_ms)s
            )
        """
        payload = {
            "session_id": None,
            "retrieved_chunk_ids": None,
            "answer": None,
            "citations": None,
            "citation_valid": None,
            "top_score": None,
            "model": None,
            "latency_ms": None,
            **payload,
        }
        with self._conn.cursor() as cur:
            cur.execute(sql, payload)
        self._conn.commit()

    def ensure_session(self, course_id: str, anon_code: str) -> int | None:
        with self._conn.cursor() as cur:
            # on conflict 里做一次无效更新，只是为了能 RETURNING 拿到已有 id
            cur.execute(
                """
                insert into student_sessions (course_id, anon_code)
                values (%s, %s)
                on conflict (course_id, anon_code)
                do update set anon_code = excluded.anon_code
                returning id
                """,
                (course_id, anon_code),
            )
            row = cur.fetchone()
        self._conn.commit()
        return int(row[0]) if row else None

    def record_gap(self, course_id: str, question: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "select id from gaps where course_id = %s and question = %s",
                (course_id, question),
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    "update gaps set hit_count = hit_count + 1, last_seen_at = now() where id = %s",
                    (row[0],),
                )
            else:
                cur.execute(
                    "insert into gaps (course_id, question) values (%s, %s)",
                    (course_id, question),
                )
        self._conn.commit()

    def record_parse_error(
        self,
        *,
        course_id: str,
        doc_key: str,
        page_no: int | None,
        error_type: str,
        detail: str = "",
        reporter: str | None = None,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                insert into parse_errors (document_id, page_no, error_type, detail, reporter)
                select id, %s, %s, %s, %s from documents
                 where course_id = %s and doc_key = %s
                """,
                (page_no, error_type, detail, reporter, course_id, doc_key),
            )
        self._conn.commit()

    def stats(self, course_id: str) -> dict[str, Any]:
        with self._conn.cursor() as cur:
            cur.execute(
                "select count(*), coalesce(sum(chunk_count), 0) from documents where course_id = %s",
                (course_id,),
            )
            doc_count, chunk_total = cur.fetchone()
            cur.execute(
                """
                select trust_level, count(*) from chunks
                 where course_id = %s group by trust_level order by trust_level
                """,
                (course_id,),
            )
            by_trust = {str(row[0]): int(row[1]) for row in cur.fetchall()}
            cur.execute(
                """
                select doc_key, title, status, trust_level, chunk_count from documents
                 where course_id = %s order by trust_level, doc_key
                """,
                (course_id,),
            )
            doc_list = [
                {
                    "doc_key": row[0],
                    "title": row[1],
                    "status": row[2],
                    "trust_level": row[3],
                    "chunk_count": row[4],
                }
                for row in cur.fetchall()
            ]
        return {
            "backend": self.backend,
            "documents": int(doc_count),
            "chunks": int(chunk_total),
            "chunks_by_trust": by_trust,
            "doc_list": doc_list,
        }

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def open_store(settings: Settings) -> Store:
    backend = str(settings.storage.get("backend", "local")).lower()
    if backend == "local":
        return LocalStore(settings.paths.out_dir)
    if backend == "postgres":
        from .dsn import inspect_dsn
        from .env import require_env

        dsn = require_env("SUPABASE_DB_URL", purpose="写入云端知识库")
        report = inspect_dsn(dsn)
        if not report.ok:
            details = "\n".join(f"  · {item}" for item in report.problems)
            hints = "\n".join(f"  提示：{item}" for item in report.hints)
            raise RuntimeError(
                f"SUPABASE_DB_URL 有问题，先修好连接串再连数据库：\n{details}\n{hints}\n"
                f"可以运行 python -m pipeline doctor 看到同样的体检结果。"
            )
        return PostgresStore(dsn, schema=str(settings.storage.get("schema", "public")))
    raise RuntimeError(
        f"config/pipeline.yaml 里的 storage.backend={backend!r} 不认识，可选 local / postgres"
    )
