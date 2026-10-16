"""密钥体检的测试。全程用 httpx.MockTransport，不联网。"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import httpx

from pipeline import cli, keycheck
from pipeline.keycheck import _model_ids, find_model, probe_key
from pipeline.keycheck import suggest_models
from pipeline.settings import Settings
from tests.support import copy_config_files


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5)


def ok_handler(ids: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": [{"id": i} for i in ids]})

    return handler


class TestFindModel(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertEqual(find_model(["a", "bge-m3"], "bge-m3"), "bge-m3")

    def test_latest_suffix_falls_back_to_base_name(self) -> None:
        self.assertEqual(find_model(["qwen-vl-max"], "qwen-vl-max-latest"), "qwen-vl-max")

    def test_prefix_match(self) -> None:
        self.assertEqual(
            find_model(["Qwen/Qwen2.5-VL-72B-Instruct"], "Qwen/Qwen2.5-VL"), "Qwen/Qwen2.5-VL-72B-Instruct"
        )

    def test_no_match_returns_none(self) -> None:
        self.assertIsNone(find_model(["a", "b"], "unrelated-model"))


class TestSuggestModels(unittest.TestCase):
    """模型名对不上时，必须给出可以照抄的候选名。"""

    def test_relaxes_prefix_step_by_step(self) -> None:
        ids = ["qwen-vl-max", "qwen-vl-plus", "qwen-turbo"]
        self.assertEqual(
            suggest_models(ids, "qwen-vl-max-latest", "vision"),
            ["qwen-vl-max", "qwen-vl-plus"],
        )

    def test_falls_back_to_role_keywords(self) -> None:
        ids = ["gpt-4o-vision", "llama-3", "embed-english"]
        self.assertEqual(
            suggest_models(ids, "unrelated-name-here", "vision"), ["gpt-4o-vision"]
        )

    def test_finds_deepseek_family(self) -> None:
        ids = ["deepseek-v4-pro", "deepseek-flash"]
        self.assertEqual(
            suggest_models(ids, "deepseek-chat", "generate"),
            ["deepseek-v4-pro", "deepseek-flash"],
        )

    def test_returns_empty_when_nothing_similar(self) -> None:
        self.assertEqual(suggest_models(["a", "b"], "zzz-yyy-xxx", "vision"), [])

    def test_respects_limit(self) -> None:
        ids = [f"vl-{n}" for n in range(50)]
        self.assertEqual(len(suggest_models(ids, "unrelated-xx", "vision", limit=5)), 5)


class TestModelIds(unittest.TestCase):
    def test_reads_id_field(self) -> None:
        self.assertEqual(_model_ids({"data": [{"id": "x"}]}), ["x"])

    def test_reads_models_field_and_plain_strings(self) -> None:
        self.assertEqual(_model_ids({"models": ["a", {"name": "b"}]}), ["a", "b"])

    def test_tolerates_junk(self) -> None:
        self.assertEqual(_model_ids(None), [])
        self.assertEqual(_model_ids({"data": [{"nothing": 1}]}), [])


class TestProbeKey(unittest.TestCase):
    def probe(self, handler, **kwargs):
        defaults = dict(
            role="vision",
            key_env="DASHSCOPE_API_KEY",
            api_key="sk-test",
            base_url="https://example.test/v1",
            model="qwen-vl-max-latest",
        )
        defaults.update(kwargs)
        return probe_key(client=make_client(handler), **defaults)

    def test_missing_key(self) -> None:
        probe = self.probe(ok_handler([]), api_key=None)
        self.assertEqual(probe.status, "missing")
        self.assertFalse(probe.healthy)
        self.assertIn("DASHSCOPE_API_KEY", probe.detail)

    def test_healthy_key_and_model(self) -> None:
        probe = self.probe(ok_handler(["qwen-vl-max-latest", "qwen-vl-plus"]))
        self.assertTrue(probe.healthy)
        self.assertEqual(probe.model_found, "qwen-vl-max-latest")
        self.assertEqual(probe.model_count, 2)
        self.assertEqual(probe.notes, [])

    def test_fuzzy_model_match_is_reported_as_a_note(self) -> None:
        probe = self.probe(ok_handler(["qwen-vl-max"]))
        self.assertTrue(probe.healthy)
        self.assertEqual(probe.model_found, "qwen-vl-max")
        self.assertTrue(any("略有出入" in note for note in probe.notes))

    def test_key_ok_but_model_missing(self) -> None:
        probe = self.probe(ok_handler(["text-embedding-v3"]))
        self.assertEqual(probe.status, "ok")
        self.assertIsNone(probe.model_found)
        self.assertFalse(probe.healthy)
        self.assertTrue(probe.model_missing)
        self.assertTrue(any("没有" in note for note in probe.notes))

    def test_model_missing_produces_copyable_suggestions(self) -> None:
        probe = self.probe(ok_handler(["embed-v1", "embed-v2"]), role="embedding")
        self.assertEqual(probe.suggestions, ["embed-v1", "embed-v2"])
        self.assertIn("embed-v1", " ".join(probe.notes))

    def test_unauthorized_points_at_wrong_slot(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "invalid api key"})

        probe = self.probe(handler)
        self.assertEqual(probe.status, "unauthorized")
        self.assertFalse(probe.healthy)
        joined = " ".join(probe.notes)
        self.assertIn("DASHSCOPE_API_KEY", joined)
        self.assertIn("别家", joined)

    def test_404_means_wrong_base_url(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="not found")

        probe = self.probe(handler)
        self.assertEqual(probe.status, "not_found")
        self.assertIn("base_url", " ".join(probe.notes))

    def test_server_error_is_reported(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        probe = self.probe(handler)
        self.assertEqual(probe.status, "error")
        self.assertIn("500", probe.detail)

    def test_network_failure_is_reported(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        probe = self.probe(handler)
        self.assertEqual(probe.status, "network")
        self.assertTrue(any("出网" in note for note in probe.notes))

    def test_non_json_response_is_reported(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>proxy login</html>")

        probe = self.probe(handler)
        self.assertEqual(probe.status, "error")

    def test_sends_bearer_token_and_hits_models_endpoint(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization", "")
            return httpx.Response(200, json={"data": [{"id": "qwen-vl-max"}]})

        self.probe(handler)
        self.assertEqual(seen["url"], "https://example.test/v1/models")
        self.assertEqual(seen["auth"], "Bearer sk-test")


class TestKeysCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        copy_config_files(self.root, embedding_dim=1024)
        self.settings = Settings.load(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_cli(self, argv: list[str], probes: list[keycheck.KeyProbe]) -> tuple[int, str]:
        original = keycheck.probe_key
        queue = list(probes)
        keycheck.probe_key = lambda **kwargs: queue.pop(0)  # type: ignore[assignment]
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                code = cli.main(["--root", str(self.root), *argv])
        finally:
            keycheck.probe_key = original  # type: ignore[assignment]
        return code, buffer.getvalue()

    def make_probe(self, role: str, status: str) -> keycheck.KeyProbe:
        return keycheck.KeyProbe(
            role=role,
            key_env=f"{role.upper()}_KEY",
            model="m",
            base_url="https://example.test/v1",
            status=status,
            model_found="m" if status == "ok" else None,
            detail="d",
        )

    def test_all_healthy_returns_zero(self) -> None:
        probes = [self.make_probe(role, "ok") for role in ("vision", "embedding", "generate")]
        code, output = self.run_cli(["keys"], probes)
        self.assertEqual(code, 0)
        self.assertIn("可以开始建库", output)
        self.assertIn("模型名也都对得上", output)

    def test_missing_key_returns_one(self) -> None:
        probes = [
            self.make_probe("vision", "ok"),
            self.make_probe("embedding", "missing"),
            self.make_probe("generate", "ok"),
        ]
        code, output = self.run_cli(["keys"], probes)
        self.assertEqual(code, 1)
        self.assertIn("未配置", output)

    def test_role_filter_probes_only_that_role(self) -> None:
        probes = [self.make_probe("embedding", "ok")]
        code, output = self.run_cli(["keys", "--role", "embedding"], probes)
        self.assertEqual(code, 0)
        self.assertIn("embedding", output)
        self.assertNotIn("generate", output)

    def test_model_name_problem_is_not_reported_as_all_good(self) -> None:
        probe = self.make_probe("vision", "ok")
        probe.model_found = None
        probe.suggestions = ["qwen-vl-max", "qwen-vl-plus"]
        code, output = self.run_cli(["keys", "--role", "vision"], [probe])
        self.assertEqual(code, 1)
        self.assertIn("问题只在模型名", output)
        self.assertIn("qwen-vl-max", output)
        self.assertNotIn("可以开始建库", output)

    def test_list_flag_dumps_available_models(self) -> None:
        probe = self.make_probe("vision", "ok")
        probe.all_models = ["qwen-vl-max", "qwen-turbo"]
        code, output = self.run_cli(["keys", "--role", "vision", "--list"], [probe])
        self.assertEqual(code, 0)
        self.assertIn("qwen-turbo", output)

    def test_grep_filters_the_list(self) -> None:
        probe = self.make_probe("vision", "ok")
        probe.all_models = ["qwen3-vl-plus", "qwen-turbo", "qwen-vl-ocr"]
        code, output = self.run_cli(
            ["keys", "--role", "vision", "--list", "--grep", "vl"], [probe]
        )
        self.assertEqual(code, 0)
        self.assertIn("qwen3-vl-plus", output)
        self.assertIn("qwen-vl-ocr", output)
        self.assertNotIn("qwen-turbo", output)

    def test_grep_without_match_says_so(self) -> None:
        probe = self.make_probe("vision", "ok")
        probe.all_models = ["qwen-turbo"]
        code, output = self.run_cli(
            ["keys", "--role", "vision", "--list", "--grep", "vl"], [probe]
        )
        self.assertEqual(code, 0)
        self.assertIn("没有名字含", output)


class TestOutputEncoding(unittest.TestCase):
    """中文输出必须能在 GBK 控制台和管道两种情况下都不乱码。"""

    def test_configure_stdio_survives_without_console(self) -> None:
        cli.configure_stdio()  # 不应抛异常

    def test_marks_fall_back_to_ascii_on_narrow_encoding(self) -> None:
        original = sys.stdout

        class Narrow:
            encoding = "gbk"

        sys.stdout = Narrow()  # type: ignore[assignment]
        try:
            ok, no = cli.marks()
        finally:
            sys.stdout = original
        self.assertEqual((ok, no), ("OK", "!!"))

    def test_marks_use_symbols_on_utf8(self) -> None:
        original = sys.stdout

        class Wide:
            encoding = "utf-8"

        sys.stdout = Wide()  # type: ignore[assignment]
        try:
            ok, no = cli.marks()
        finally:
            sys.stdout = original
        self.assertEqual((ok, no), ("✓", "✗"))


if __name__ == "__main__":
    unittest.main()
