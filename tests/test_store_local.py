from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.chunker import Chunk, ChunkConfig, chunk_document
from pipeline.embedder import EmbeddingDimError, embed_chunks
from pipeline.store import ChunkRecord, DocumentRecord, LocalStore, vector_literal
from pipeline.textutils import page_marker
from tests.support import FakeEmbedder


def sample_chunks() -> list[Chunk]:
    markdown = "\n\n".join(
        [
            page_marker(1),
            "# 第 1 章",
            "## 1.1 节点电流定律",
            "节点电流定律（KCL）：流入任一节点的电流之和等于流出电流之和。" * 4,
            page_marker(2),
            "## 1.2 戴维南定理",
            "戴维南定理：含源二端网络可等效为电压源与电阻串联。" * 4,
        ]
    )
    return chunk_document(
        markdown,
        doc_key="doc-x",
        config=ChunkConfig(target_chars=120, max_chars=200, min_chars=40, overlap_chars=0),
    )


class TestVectorLiteral(unittest.TestCase):
    def test_formats_values_for_pgvector(self) -> None:
        self.assertEqual(vector_literal([1.0, 0.5, 0.0]), "[1,0.5,0]")


class TestEmbedChunks(unittest.TestCase):
    def test_embeds_every_chunk(self) -> None:
        chunks = sample_chunks()
        vectors = embed_chunks(chunks, FakeEmbedder(), batch_size=2, expected_dim=5)
        self.assertEqual(len(vectors), len(chunks))
        self.assertTrue(all(len(vector) == 5 for vector in vectors))

    def test_wrong_dimension_raises_with_guidance(self) -> None:
        chunks = sample_chunks()
        with self.assertRaises(EmbeddingDimError) as ctx:
            embed_chunks(chunks, FakeEmbedder(), expected_dim=1024)
        self.assertIn("chunks.embedding", str(ctx.exception))

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(embed_chunks([], FakeEmbedder()), [])


class TestLocalStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = LocalStore(Path(self.tmp.name) / "out")
        self.chunks = sample_chunks()
        self.vectors = embed_chunks(self.chunks, FakeEmbedder(), expected_dim=5)
        self.document = DocumentRecord(
            course_id="c1",
            doc_key="doc-x",
            filename="ch01.pptx",
            doc_type="slide",
            trust_level=1,
            title="第 1 章",
            status="loaded",
            chunk_count=len(self.chunks),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def load(self) -> int:
        records = [
            ChunkRecord.from_chunk(chunk, trust_level=1, embedding=vector)
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        return self.store.replace_chunks(self.document, records)

    def test_replace_chunks_is_idempotent(self) -> None:
        self.assertEqual(self.load(), len(self.chunks))
        self.assertEqual(self.load(), len(self.chunks))
        stats = self.store.stats("c1")
        self.assertEqual(stats["chunks"], len(self.chunks))
        self.assertEqual(stats["documents"], 1)

    def test_search_returns_ranked_hits_with_pages(self) -> None:
        self.load()
        query = FakeEmbedder().embed(["节点 KCL 电流定律"])[0]
        hits = self.store.search("c1", query, limit=5, max_trust=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0].page_no, 1)
        self.assertGreater(hits[0].score, 0.95)
        self.assertGreater(hits[0].score, hits[-1].score)

    def test_search_respects_trust_level(self) -> None:
        records = [
            ChunkRecord.from_chunk(chunk, trust_level=3, embedding=vector)
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        self.store.replace_chunks(self.document, records)
        self.assertEqual(self.store.search("c1", self.vectors[0], max_trust=2), [])
        self.assertTrue(self.store.search("c1", self.vectors[0], max_trust=3))

    def test_search_on_other_course_returns_nothing(self) -> None:
        self.load()
        self.assertEqual(self.store.search("other", self.vectors[0]), [])

    def test_gap_is_counted_when_repeated(self) -> None:
        self.store.record_gap("c1", "什么是回转器？")
        self.store.record_gap("c1", "什么是回转器？")
        import json

        rows = [
            json.loads(line)
            for line in self.store.gaps_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(rows[0]["hit_count"], 2)

    def test_parse_error_and_qa_log_are_appended(self) -> None:
        self.store.record_parse_error(
            course_id="c1",
            doc_key="doc-x",
            page_no=5,
            error_type="formula_lost",
            detail="麦克斯韦方程组整段丢失",
        )
        self.store.log_qa({"course_id": "c1", "question": "q", "mode": "kb"})
        self.assertTrue((self.store.out_dir / "parse_errors.jsonl").is_file())
        self.assertTrue((self.store.out_dir / "qa_logs.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
