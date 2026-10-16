"""失败原因必须能被人看到。

这组测试的由来：第一次真实运行时 5 页全部 403 失败，而 `run` 只打印了
「第 N 页失败」，没说为什么——使用者只能去翻中间文件。这类「错误被吞掉」
的缺陷不写出测试就会复发。
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Sequence

import httpx

from pipeline import cli
from pipeline.ai_parser import parse_document
from pipeline.convert import PageAsset, RenderedDoc
from pipeline.llm import LLMError, ModelConfig, OpenAICompatClient
from pipeline.manifest import ManifestEntry
from pipeline.runner import RunOptions, run_document
from pipeline.settings import Settings
from pipeline.store import LocalStore
from tests.support import FakeModelConfig, copy_config_files

DASHSCOPE_403 = {
    "error": {
        "message": "Access denied. For details, see: https://help.aliyun.com/zh/model-studio/error-code",
        "code": "AccessDenied",
    }
}


class TestErrorMessageExtraction(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = os.environ.get("DASHSCOPE_API_KEY")
        os.environ["DASHSCOPE_API_KEY"] = "sk-test-key"

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop("DASHSCOPE_API_KEY", None)
        else:
            os.environ["DASHSCOPE_API_KEY"] = self._saved

    def detail(self, response: httpx.Response) -> str:
        return OpenAICompatClient._error_detail(response)

    def test_extracts_nested_error_message(self) -> None:
        response = httpx.Response(403, json=DASHSCOPE_403)
        self.assertIn("Access denied", self.detail(response))

    def test_extracts_plain_string_error(self) -> None:
        response = httpx.Response(400, json={"error": "bad request"})
        self.assertEqual(self.detail(response), "bad request")

    def test_falls_back_to_raw_text(self) -> None:
        response = httpx.Response(500, text="<html>gateway error</html>")
        self.assertIn("gateway error", self.detail(response))

    def test_handles_empty_body(self) -> None:
        response = httpx.Response(503, text="")
        self.assertTrue(self.detail(response))

    def test_client_raises_with_hint_for_rejection(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json=DASHSCOPE_403)

        client = OpenAICompatClient(
            ModelConfig(
                role="vision",
                name="qwen-vl-max-latest",
                base_url="https://example.test/v1",
                api_key_env="DASHSCOPE_API_KEY",
            ),
            client=httpx.Client(transport=httpx.MockTransport(handler), timeout=5),
        )
        with self.assertRaises(LLMError) as ctx:
            client.chat(system="s", user="u")
        message = str(ctx.exception)
        self.assertIn("Access denied", message)
        self.assertIn("pipeline keys", message)
        self.assertIn("DASHSCOPE_API_KEY", message)


class FailingVision:
    """模拟百炼返回 403 的视觉客户端。"""

    def __init__(self, message: str = "vision 调用被拒绝（HTTP 403）：Access denied.") -> None:
        self.config = FakeModelConfig(name="fake-vision")
        self.message = message
        self.calls = 0

    def chat_with_images(self, *, system, user, image_paths, max_tokens=None) -> str:
        self.calls += 1
        raise LLMError(self.message)

    def chat(self, *, system, user, max_tokens=None) -> str:
        raise LLMError(self.message)

    def close(self) -> None:
        pass


class TestParseErrorsArePersisted(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        copy_config_files(self.root, embedding_dim=5)
        self.settings = Settings.load(self.root)
        self.settings.parsing["retry"] = 1
        self.entry = ManifestEntry(
            doc_key="doc-x",
            course_id="circuit-theory-1",
            doc_type="slide",
            trust_level=1,
            filename="doc.pdf",
            title="测试课件",
        )
        pages_dir = self.settings.paths.doc_pages("doc-x")
        pages_dir.mkdir(parents=True, exist_ok=True)
        images = []
        for page_no in (1, 2):
            image = pages_dir / f"doc-x_page_{page_no:04d}.jpg"
            image.write_bytes(b"\xff\xd8\xff\xd9")
            images.append(
                PageAsset(page_no=page_no, image_path=image, has_embedded_image=True)
            )
        self.rendered = RenderedDoc(
            doc_key="doc-x", source_path=Path("x.pdf"), converter="test", pages=images
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_full_error_is_written_to_parse_errors_json(self) -> None:
        long_message = "vision 调用被拒绝（HTTP 403）：" + "Access denied. " * 40
        report, _ = parse_document(
            self.rendered,
            self.entry,
            self.settings,
            FailingVision(long_message),
            use_cache=False,
        )
        self.assertEqual(len(report.failures), 2)

        errors_path = self.settings.paths.doc_work("doc-x") / "parse_errors.json"
        rows = json.loads(errors_path.read_text(encoding="utf-8"))
        self.assertEqual([row["page_no"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["error"], long_message)
        self.assertIn("403", rows[0]["error"])

    def test_page_marker_keeps_enough_of_the_message(self) -> None:
        parse_document(
            self.rendered,
            self.entry,
            self.settings,
            FailingVision(),
            use_cache=False,
        )
        marker = (
            self.settings.paths.doc_pages("doc-x") / "doc-x_page_0001.md"
        ).read_text(encoding="utf-8")
        self.assertIn("HTTP 403", marker)
        self.assertIn("Access denied", marker)

    def test_run_summary_carries_failure_details(self) -> None:
        write_pdf(self.settings.paths.raw_dir / "doc.pdf", pages=2)
        store = LocalStore(self.settings.paths.out_dir)
        summary = run_document(
            self.entry,
            self.settings,
            store,
            vision_client=FailingVision(),
            embed_client=object(),  # 质量闸门会拦住，不该走到向量化
            options=RunOptions(),
            log=lambda message: None,
        )
        self.assertEqual(summary.failed_pages, 2)
        self.assertEqual(len(summary.failure_details), 2)
        self.assertIn("403", summary.failure_details[0][1])
        # 关键：全部页失败时不许把「解析失败」标记切块入库
        self.assertIn("chunk/embed/load", summary.skipped_stages)
        self.assertEqual(store.stats("circuit-theory-1").get("chunks", 0), 0)

    def test_allow_partial_overrides_the_gate(self) -> None:
        write_pdf(self.settings.paths.raw_dir / "doc.pdf", pages=2)
        store = LocalStore(self.settings.paths.out_dir)
        summary = run_document(
            self.entry,
            self.settings,
            store,
            vision_client=FailingVision(),
            embed_client=FailingVision(),
            options=RunOptions(allow_partial=True, skip_embed=True),
            log=lambda message: None,
        )
        self.assertGreater(summary.chunk_count, 0)


def write_pdf(path: Path, pages: int) -> None:
    try:
        import pymupdf
    except ImportError:  # pragma: no cover
        import fitz as pymupdf  # type: ignore

    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 100), f"page {index + 1}", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path))
    document.close()


class TestFailureSummaryOutput(unittest.TestCase):
    def capture(self, details: Sequence[tuple[int, str]]) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(buffer):
            cli.summarize_failures(list(details))
        return buffer.getvalue()

    def test_groups_identical_reasons(self) -> None:
        output = self.capture([(n, "同样的错") for n in (1, 2, 3)])
        self.assertIn("1 种原因", output)
        self.assertEqual(output.count("同样的错"), 1)
        self.assertIn("[1, 2, 3]", output)

    def test_lists_distinct_reasons_separately(self) -> None:
        output = self.capture([(1, "原因A"), (2, "原因B")])
        self.assertIn("2 种原因", output)
        self.assertIn("原因A", output)
        self.assertIn("原因B", output)

    def test_truncates_when_too_many_reasons(self) -> None:
        output = self.capture([(n, f"原因{n}") for n in range(1, 7)], )
        self.assertIn("另有", output)
        self.assertIn("parse_errors.json", output)

    def test_no_output_when_no_failures(self) -> None:
        self.assertEqual(self.capture([]), "")


if __name__ == "__main__":
    unittest.main()
