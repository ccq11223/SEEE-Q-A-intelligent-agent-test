from __future__ import annotations

import unittest

from pipeline.chunker import ChunkConfig, chunk_document
from pipeline.textutils import page_marker


def build_markdown() -> str:
    body_a = "节点电流定律：在任一节点上，流入电流之和等于流出电流之和。" * 6
    body_b = "戴维南定理说明，含源二端网络可以等效为一个电压源与一个电阻串联。" * 6
    body_c = "正弦稳态下相量法把时域微分方程化为代数方程。" * 6
    return "\n\n".join(
        [
            page_marker(1),
            "# 第 1 章 电路基本定律",
            "## 1.1 基尔霍夫定律",
            body_a,
            page_marker(2),
            "## 1.2 戴维南定理",
            body_b,
            page_marker(3),
            "### 1.2.1 相量法",
            body_c,
        ]
    )


class TestChunking(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ChunkConfig(
            target_chars=200, max_chars=320, min_chars=80, overlap_chars=0
        )
        self.chunks = chunk_document(build_markdown(), doc_key="doc-a", config=self.config)

    def test_produces_multiple_chunks(self) -> None:
        self.assertGreaterEqual(len(self.chunks), 3)

    def test_every_chunk_has_page_number(self) -> None:
        for chunk in self.chunks:
            self.assertIsNotNone(chunk.page_no, f"{chunk.chunk_key} 缺页码")

    def test_every_chunk_has_section_title(self) -> None:
        for chunk in self.chunks:
            self.assertTrue(chunk.section_title, f"{chunk.chunk_key} 缺章节")

    def test_chunk_keys_are_unique_and_ordered(self) -> None:
        keys = [chunk.chunk_key for chunk in self.chunks]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(keys[0], "doc-a#0001")
        self.assertEqual(keys, sorted(keys))

    def test_pages_are_increasing(self) -> None:
        pages = [chunk.page_no for chunk in self.chunks]
        self.assertEqual(pages, sorted(pages))

    def test_content_is_not_lost(self) -> None:
        joined = "".join(chunk.content for chunk in self.chunks)
        for token in ("节点电流定律", "戴维南定理", "相量法"):
            self.assertIn(token, joined)

    def test_chapter_is_first_level_heading(self) -> None:
        self.assertTrue(all(chunk.chapter == "第 1 章 电路基本定律" for chunk in self.chunks))

    def test_token_count_is_positive(self) -> None:
        for chunk in self.chunks:
            self.assertGreater(chunk.token_count, 0)

    def test_char_count_matches_content(self) -> None:
        for chunk in self.chunks:
            self.assertEqual(chunk.char_count, len(chunk.content))


class TestLongParagraphSplit(unittest.TestCase):
    def test_single_huge_paragraph_is_split(self) -> None:
        paragraph = "这是一段很长的正文内容。" * 60
        markdown = f"{page_marker(1)}\n# 标题\n{paragraph}"
        chunks = chunk_document(
            markdown,
            doc_key="doc-b",
            config=ChunkConfig(target_chars=200, max_chars=250, min_chars=80, overlap_chars=0),
        )
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk.char_count, 300)


class TestOverlap(unittest.TestCase):
    def test_overlap_keeps_all_content(self) -> None:
        paragraph = "第一句内容。第二句内容。第三句内容。" * 20
        markdown = f"{page_marker(1)}\n# 标题\n{paragraph}"
        config = ChunkConfig(target_chars=150, max_chars=200, min_chars=60, overlap_chars=30)
        chunks = chunk_document(markdown, doc_key="doc-c", config=config)
        self.assertGreater(len(chunks), 1)
        joined = "".join(chunk.content for chunk in chunks)
        self.assertIn("第一句内容。", joined)


if __name__ == "__main__":
    unittest.main()
