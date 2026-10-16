"""端到端离线回归：真实 PDF → 渲染 → 解析（假模型）→ 切块 → 向量化 → 入库 → 检索。

这是最接近真实使用的一次演练，全程不联网：
唯一的假件是「视觉模型」和「向量模型」，其余（渲染、切块、存储、检索）都走真实代码。
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from pipeline.manifest import load_manifest
from pipeline.runner import (
    RunOptions,
    build_combined_markdown,
    run_document,
)
from pipeline.settings import Settings
from pipeline.store import LocalStore
from tests.support import FakeEmbedder, FakeModelConfig

CONFIG_FILES = ("pipeline.yaml", "models.yaml", "retrieval.yaml", "prompts.yaml")

PAGE_MARKDOWN = {
    1: "# 第 1 章 电路基本定律\n\n## 1.1 基尔霍夫定律\n\n节点电流定律：$\\sum i = 0$。\n\n[图 第1页-1：电路节点示意]",
    2: "# 第 1 章 电路基本定律\n\n## 1.2 戴维南定理\n\n含源二端网络可等效为电压源与电阻串联。",
    3: "# 第 1 章 电路基本定律\n\n## 1.3 相量法\n\n正弦稳态下可用相量把微分方程化为代数方程。",
}


class FakeVision:
    """假视觉模型：按页码返回预设 Markdown，并记录调用次数。"""

    def __init__(self) -> None:
        self.config = FakeModelConfig(name="fake-vision")
        self.calls: list[int] = []

    def chat_with_images(self, *, system, user, image_paths, max_tokens=None) -> str:
        # user 模板里带「第 N 页」，从里面取页码，便于断言缓存行为
        page_no = 1
        for token in user.replace("，", " ").split():
            digits = "".join(ch for ch in token if ch.isdigit())
            if digits:
                page_no = int(digits)
                break
        self.calls.append(page_no)
        return PAGE_MARKDOWN.get(page_no, "空页")

    def chat(self, *, system, user, max_tokens=None) -> str:  # pragma: no cover
        raise AssertionError("vision 客户端在本测试里不应该走纯文本分支")

    def close(self) -> None:
        pass


def build_project(root: Path) -> Path:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    for name in CONFIG_FILES:
        shutil.copy(project_root / "config" / name, config_dir / name)

    # 假向量模型是 5 维，把配置里的维度改成一致，避免维度校验（这是它的职责）误报
    models_path = config_dir / "models.yaml"
    models_path.write_text(
        models_path.read_text(encoding="utf-8").replace("dim: 1024", "dim: 5"),
        encoding="utf-8",
    )

    (config_dir / "manifest.csv").write_text(
        "doc_key,course_id,doc_type,trust_level,filename,title,term,priority,notes\n"
        "demo,circuit-theory-1,slide,1,demo.pdf,演示课件,2026秋,1,离线回归\n",
        encoding="utf-8",
    )

    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_pdf(raw_dir / "demo.pdf", len(PAGE_MARKDOWN))
    return root


def write_pdf(path: Path, pages: int) -> None:
    try:
        import pymupdf
    except ImportError:  # pragma: no cover
        import fitz as pymupdf  # type: ignore

    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 100), f"Page {index + 1} of demo", fontsize=16)
    document.save(str(path))
    document.close()


class TestRunnerOffline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = build_project(Path(self.tmp.name))
        self.settings = Settings.load(self.root)
        self.entry = load_manifest(self.root / "config" / "manifest.csv")[0]
        self.store = LocalStore(self.settings.paths.out_dir)
        self.vision = FakeVision()
        self.embedder = FakeEmbedder()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_full(self, options: RunOptions | None = None):
        return run_document(
            self.entry,
            self.settings,
            self.store,
            vision_client=self.vision,
            embed_client=self.embedder,
            options=options or RunOptions(),
            log=lambda message: None,
        )

    def test_full_pipeline_loads_chunks(self) -> None:
        summary = self.run_full()
        self.assertEqual(summary.page_count, 3)
        self.assertEqual(summary.parsed_pages, 3)
        self.assertEqual(summary.failed_pages, 0)
        self.assertGreater(summary.chunk_count, 0)
        self.assertEqual(summary.loaded, summary.chunk_count)
        self.assertEqual(summary.skipped_stages, [])

    def test_combined_markdown_keeps_page_markers(self) -> None:
        self.run_full()
        markdown = self.settings.paths.doc_markdown("demo").read_text(encoding="utf-8")
        for page_no in (1, 2, 3):
            self.assertIn(f"<!-- page: {page_no} -->", markdown)
        self.assertIn("\\sum i = 0", markdown)
        self.assertIn("[图 第1页-1：电路节点示意]", markdown)

    def test_chunks_carry_page_and_trust_level(self) -> None:
        self.run_full()
        stats = self.store.stats("circuit-theory-1")
        self.assertGreater(stats["chunks"], 0)
        self.assertIn("1", stats["chunks_by_trust"])
        hits = self.store.search(
            "circuit-theory-1", self.embedder.embed(["节点 基尔霍夫定律"])[0], limit=3
        )
        self.assertTrue(hits)
        self.assertIsNotNone(hits[0].page_no)

    def test_second_run_hits_cache_and_does_not_call_model(self) -> None:
        self.run_full()
        first_calls = len(self.vision.calls)
        self.assertEqual(first_calls, 3)
        self.run_full()
        self.assertEqual(len(self.vision.calls), first_calls)

    def test_no_cache_forces_reparse(self) -> None:
        self.run_full()
        self.run_full(RunOptions(use_cache=False))
        self.assertEqual(len(self.vision.calls), 6)

    def test_partial_parse_skips_loading(self) -> None:
        summary = self.run_full(RunOptions(pages={1}))
        self.assertEqual(summary.parsed_pages, 1)
        self.assertIn("chunk/embed/load", summary.skipped_stages)
        self.assertEqual(self.store.stats("circuit-theory-1").get("chunks", 0), 0)
        _, missing = build_combined_markdown(self.settings, "demo", page_count=3)
        self.assertEqual(missing, [2, 3])

    def test_partial_parse_allowed_loads_partial_content(self) -> None:
        summary = self.run_full(RunOptions(pages={1}, allow_partial=True))
        self.assertEqual(summary.parsed_pages, 1)
        self.assertGreater(summary.loaded, 0)

    def test_dry_run_writes_files_but_not_database(self) -> None:
        summary = self.run_full(RunOptions(dry_run=True))
        self.assertIn("load", summary.skipped_stages)
        self.assertEqual(summary.loaded, 0)
        self.assertEqual(self.store.stats("circuit-theory-1").get("chunks", 0), 0)
        self.assertTrue(self.settings.paths.doc_chunks("demo").is_file())

    def test_rendered_page_images_are_jpeg(self) -> None:
        self.run_full()
        pages_dir = self.settings.paths.doc_pages("demo")
        images = sorted(pages_dir.glob("demo_page_*.jpg"))
        self.assertEqual(len(images), 3)
        for image in images:
            self.assertGreater(image.stat().st_size, 500)


if __name__ == "__main__":
    unittest.main()
