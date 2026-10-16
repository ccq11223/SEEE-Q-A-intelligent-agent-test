"""推理模型的空答案要能被自动救回来。

实测：deepseek-flash / deepseek-v4-pro 是推理模型，先输出思考过程再输出答案。
额度被思考过程用光时，content 是空字符串、finish_reason 是 length。
这种事在真实数据里随机出现（同一批页里大部分正常、个别页中招），
所以必须自动加大额度重试，而不是把报错抛给使用者。
"""

from __future__ import annotations

import os
import unittest

import httpx

from pipeline.llm import (
    LLMError,
    ModelConfig,
    OpenAICompatClient,
    OutputTruncated,
)


def truncated_response() -> dict:
    return {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "我们需要先理解题意……",
                },
            }
        ]
    }


def ok_response(text: str = "答案（第 12 页）。") -> dict:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ]
    }


class TestExtractText(unittest.TestCase):
    def test_empty_content_with_reasoning_raises_output_truncated(self) -> None:
        with self.assertRaises(OutputTruncated) as ctx:
            OpenAICompatClient._extract_text(truncated_response())
        self.assertIn("思考过程", str(ctx.exception))

    def test_empty_content_without_reasoning_is_plain_error(self) -> None:
        payload = {"choices": [{"message": {"content": ""}}]}
        with self.assertRaises(LLMError) as ctx:
            OpenAICompatClient._extract_text(payload)
        self.assertNotIsInstance(ctx.exception, OutputTruncated)

    def test_normal_response_passes_through(self) -> None:
        self.assertEqual(
            OpenAICompatClient._extract_text(ok_response("第一句话。")), "第一句话。"
        )


class TestEscalation(unittest.TestCase):
    def setUp(self) -> None:
        # ModelConfig.api_key 会去读环境变量，测试里给它一个假值
        os.environ.setdefault("K", "sk-test")

    def build(self, handler) -> OpenAICompatClient:
        return OpenAICompatClient(
            ModelConfig(
                role="generate",
                name="deepseek-flash",
                base_url="https://example.test/v1",
                api_key_env="K",
                max_tokens=1000,
            ),
            client=httpx.Client(transport=httpx.MockTransport(handler), timeout=5),
        )

    def test_retries_with_double_cap_and_succeeds(self) -> None:
        caps: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            caps.append(json.loads(request.content)["max_tokens"])
            if len(caps) == 1:
                return httpx.Response(200, json=truncated_response())
            return httpx.Response(200, json=ok_response())

        client = self.build(handler)
        self.assertEqual(client.chat(system="s", user="u"), "答案（第 12 页）。")
        self.assertEqual(caps, [1000, 2000])

    def test_gives_up_after_three_attempts_with_actionable_message(self) -> None:
        caps: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            caps.append(json.loads(request.content)["max_tokens"])
            return httpx.Response(200, json=truncated_response())

        client = self.build(handler)
        with self.assertRaises(LLMError) as ctx:
            client.chat(system="s", user="u")
        self.assertEqual(len(caps), 3)
        self.assertEqual(caps, [1000, 2000, 4000])
        self.assertIn("max_tokens", str(ctx.exception))

    def test_cap_is_not_doubled_forever(self) -> None:
        caps: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            cap = json.loads(request.content)["max_tokens"]
            caps.append(cap)
            if cap >= 20000:
                return httpx.Response(200, json=ok_response())
            return httpx.Response(200, json=truncated_response())

        client = OpenAICompatClient(
            ModelConfig(
                role="generate",
                name="m",
                base_url="https://example.test/v1",
                api_key_env="K",
                max_tokens=20000,
            ),
            client=httpx.Client(transport=httpx.MockTransport(handler), timeout=5),
        )
        self.assertEqual(client.chat(system="s", user="u"), "答案（第 12 页）。")
        self.assertEqual(caps, [20000])

    def test_vision_calls_also_escalate(self) -> None:
        import tempfile
        from pathlib import Path

        caps: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            caps.append(json.loads(request.content)["max_tokens"])
            if len(caps) == 1:
                return httpx.Response(200, json=truncated_response())
            return httpx.Response(200, json=ok_response("# 标题"))

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "page.jpg"
            image.write_bytes(b"\xff\xd8\xff\xd9")
            client = self.build(handler)
            text = client.chat_with_images(
                system="s", user="u", image_paths=[image]
            )
        self.assertEqual(text, "# 标题")
        self.assertEqual(caps, [1000, 2000])


if __name__ == "__main__":
    unittest.main()
