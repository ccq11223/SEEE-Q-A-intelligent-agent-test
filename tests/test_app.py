"""在线服务接口测试。

用假模型客户端注入，不联网、不花钱；测的是「接口行为」——
参数校验、限流、转人工标记、错误处理。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import AppConfig
from app.main import create_app
from app.ratelimit import HourlyLimiter
from app.services import Services
from pipeline.store import Hit
from tests.support import FakeEmbedder, FakeGenerator, FakeModelConfig, FakeStore, make_settings


class FakeReranker:
    def __init__(self, score: float) -> None:
        self.config = FakeModelConfig(name="fake-rerank")
        self.score = score

    def rerank(self, *, query, documents, top_n=None):
        return [(index, self.score) for index in range(len(documents))]

    def close(self) -> None:
        pass


def make_hit(page_no: int = 12, score: float = 0.9) -> Hit:
    return Hit(
        chunk_key="slide-07#0001",
        page_no=page_no,
        page_end=None,
        chapter="第4章",
        section_title="1.1 基尔霍夫定律",
        content="节点电流定律：流入等于流出。",
        trust_level=1,
        score=score,
    )


class AppTestCase(unittest.TestCase):
    def build(
        self,
        *,
        hits=None,
        reply="节点电流定律说明流入等于流出（第 12 页）。",
        rerank_score=0.95,
        limit=30,
        settings_overrides=None,
    ) -> TestClient:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        settings = make_settings(Path(self.tmp.name), **(settings_overrides or {}))
        self.generator = FakeGenerator([reply])
        services = Services(
            settings=settings,
            store=FakeStore(hits if hits is not None else [make_hit()]),
            embed=FakeEmbedder(),
            generate=self.generator,
            rerank=FakeReranker(rerank_score),
        )
        config = AppConfig(
            title="测试课程答疑",
            intro="介绍",
            disclaimer="免责声明",
            placeholder="问吧",
            max_question_chars=20,
            asks_per_hour_per_ip=limit,
        )
        app = create_app(services=services, config=config)
        client = TestClient(app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def ask(self, client: TestClient, question: str, anon: str = "xh-test"):
        return client.post("/api/ask", json={"anon_code": anon, "question": question})


class TestMetaEndpoints(AppTestCase):
    def test_config(self) -> None:
        client = self.build()
        data = client.get("/api/config").json()
        self.assertEqual(data["title"], "测试课程答疑")
        self.assertEqual(data["max_question_chars"], 20)
        self.assertEqual(data["course"], "电路理论（上）")

    def test_health(self) -> None:
        client = self.build()
        data = client.get("/api/health").json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["rerank_enabled"])
        self.assertEqual(data["backend"], "fake")

    def test_index_page_is_served(self) -> None:
        client = self.build()
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("匿名代号", resp.text)


class TestAsk(AppTestCase):
    def test_answer_from_knowledge_base(self) -> None:
        client = self.build()
        resp = self.ask(client, "节点电流定律是什么？")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["mode"], "kb")
        self.assertFalse(data["handoff"])
        self.assertIn("第 12 页", data["answer"])
        self.assertEqual(data["citations"], [12])
        # 课件作答要带回原页图片，前端据此「图文并茂」
        self.assertEqual(len(data["images"]), 1)
        self.assertEqual(data["images"][0]["doc"], "slide-07")
        self.assertEqual(data["images"][0]["page"], 12)
        self.assertEqual(data["images"][0]["url"], "/api/page-image/slide-07/12")

    def test_page_image_endpoint_missing_and_bad_doc(self) -> None:
        client = self.build()
        self.assertEqual(client.get("/api/page-image/slide-07/12").status_code, 404)
        self.assertEqual(client.get("/api/page-image/bad%20doc/1").status_code, 400)

    def test_followup_history_reaches_generation(self) -> None:
        client = self.build()
        resp = client.post(
            "/api/ask",
            json={
                "anon_code": "xh-test",
                "question": "不是，是那种通用的结点方程",
                "history": [{"role": "user", "content": "怎么列写节点支路方程？"}],
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("之前的对话", self.generator.prompts[0])
        self.assertIn("怎么列写节点支路方程", self.generator.prompts[0])

    def test_no_evidence_answers_from_general_knowledge(self) -> None:
        client = self.build(
            hits=[make_hit(score=0.01)],
            rerank_score=0.05,
            reply="替代定理适用于线性电路，具体条件如下……",
        )
        resp = self.ask(client, "回转器怎么用？")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["handoff"])
        self.assertEqual(data["mode"], "general")
        self.assertEqual(data["source_level"], "general")
        self.assertIn("课件未覆盖", data["answer"].splitlines()[0])

    def test_handoff_only_when_fallbacks_are_disabled(self) -> None:
        client = self.build(
            hits=[make_hit(score=0.01)],
            rerank_score=0.05,
            settings_overrides={"retrieval": {"allow_general": False}},
        )
        data = self.ask(client, "回转器怎么用？").json()
        self.assertTrue(data["handoff"])
        self.assertEqual(data["mode"], "handoff")

    def test_empty_question_is_rejected(self) -> None:
        client = self.build()
        resp = self.ask(client, "   ")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("不能为空", resp.json()["detail"])

    def test_too_long_question_is_rejected(self) -> None:
        client = self.build()
        resp = self.ask(client, "问" * 50)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("太长", resp.json()["detail"])

    def test_bad_anon_code_is_rejected(self) -> None:
        client = self.build()
        for bad in ("", "a", "学号20230001", "has space", "x" * 20):
            resp = self.ask(client, "节点电流定律是什么？", anon=bad)
            self.assertEqual(resp.status_code, 400, bad)
        self.assertIn("匿名代号", resp.json()["detail"])

    def test_rate_limit(self) -> None:
        client = self.build(limit=2)
        self.assertEqual(self.ask(client, "第一个问题").status_code, 200)
        self.assertEqual(self.ask(client, "第二个问题").status_code, 200)
        resp = self.ask(client, "第三个问题")
        self.assertEqual(resp.status_code, 429)
        self.assertIn("频繁", resp.json()["detail"])

    def test_llm_failure_returns_503_without_leaking_details(self) -> None:
        client = self.build(hits=[make_hit()], reply="", rerank_score=0.95)
        # FakeGenerator 返回空串会让引用校验失败转人工，不会抛异常；
        # 这里改用会抛错的客户端验证 503 分支
        from pipeline.llm import LLMError

        class BrokenGenerator(FakeGenerator):
            def chat(self, *, system, user, max_tokens=None):
                raise LLMError("vision 调用被拒绝（HTTP 403）：Access denied")

        services = client.app.state.services
        services.generate.close()
        services.generate = BrokenGenerator()
        resp = self.ask(client, "节点电流定律是什么？")
        self.assertEqual(resp.status_code, 503)
        self.assertIn("暂时不可用", resp.json()["detail"])
        self.assertNotIn("403", resp.json()["detail"])


class TestLimiter(unittest.TestCase):
    def test_window_slides(self) -> None:
        limiter = HourlyLimiter(2, window_s=10)
        self.assertTrue(limiter.allow("ip", now=0))
        self.assertTrue(limiter.allow("ip", now=1))
        self.assertFalse(limiter.allow("ip", now=2))
        # 11 秒后第一条过期
        self.assertTrue(limiter.allow("ip", now=11))

    def test_separate_keys(self) -> None:
        limiter = HourlyLimiter(1)
        self.assertTrue(limiter.allow("a", now=0))
        self.assertFalse(limiter.allow("a", now=0))
        self.assertTrue(limiter.allow("b", now=0))

    def test_remaining(self) -> None:
        limiter = HourlyLimiter(3)
        limiter.allow("ip", now=0)
        self.assertEqual(limiter.remaining("ip", now=1), 2)


if __name__ == "__main__":
    unittest.main()
