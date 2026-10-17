"""OpenAI 兼容接口客户端（对话 / 视觉 / 向量）。

刻意只依赖 httpx：三家服务（百炼、硅基流动、DeepSeek）都提供 OpenAI 兼容的
HTTP 接口，手写这一层比绑定各家 SDK 更好换、也更容易看明白发生了什么。
"""

from __future__ import annotations

import base64
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx

from .env import get_env


class LLMError(RuntimeError):
    """模型调用失败。消息里必须带上服务商返回的原文，方便对症下药。"""


class OutputTruncated(LLMError):
    """推理模型的思考过程把 max_tokens 用光了，答案一个字都没写出来。

    单独定义成一个异常，是为了能被自动重试捕获——换更大的额度再试一次，
    而不是直接把失败抛给使用者。实测推理模型的思考长度不可预测：
    同一个提示词，大部分页够用，个别页 4096 都不够。
    """


@dataclass
class ModelConfig:
    role: str
    name: str
    base_url: str
    api_key_env: str
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout_s: float = 120.0
    dim: int | None = None

    @classmethod
    def from_dict(cls, role: str, raw: dict[str, Any]) -> "ModelConfig":
        return cls(
            role=role,
            name=str(raw["name"]),
            base_url=str(raw["base_url"]).rstrip("/"),
            api_key_env=str(raw["api_key_env"]),
            max_tokens=int(raw.get("max_tokens", 2048)),
            temperature=float(raw.get("temperature", 0.0)),
            timeout_s=float(raw.get("timeout_s", 120)),
            dim=int(raw["dim"]) if raw.get("dim") else None,
        )

    @property
    def api_key(self) -> str:
        key = get_env(self.api_key_env)
        if not key:
            raise LLMError(
                f"缺少 {self.api_key_env}。请在 .env 里填好该服务的 API Key"
                f"（模型：{self.name}，服务地址：{self.base_url}）。"
            )
        return key


def image_to_data_url(path: Path | str) -> str:
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


class OpenAICompatClient:
    """所有模型调用的唯一出口。"""

    def __init__(
        self,
        config: ModelConfig,
        *,
        retry: int = 3,
        backoff_s: float = 2.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self.retry = max(1, retry)
        self.backoff_s = backoff_s
        self._client = client or httpx.Client(timeout=config.timeout_s)
        self._owns_client = client is None

    # ---------- 生命周期 ----------
    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OpenAICompatClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---------- 底层请求 ----------
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _error_detail(response: httpx.Response, limit: int = 600) -> str:
        """尽量抽出服务商返回里真正有用的那句话。

        OpenAI 兼容接口的报错通常长这样：
            {"error": {"message": "Access denied. ...", "code": "..."}}
        直接打印整段 JSON 会让人抓不到重点，而且关键信息常被截断在末尾。
        """
        raw = (response.text or "").strip()
        try:
            payload = response.json()
        except ValueError:
            return raw[:limit] or "(服务端没有返回内容)"

        message: object = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message") or error.get("code")
            elif isinstance(error, str):
                message = error
            if not message:
                message = payload.get("message") or payload.get("code")

        if isinstance(message, str) and message.strip():
            return message.strip()[:limit]
        return raw[:limit] or "(服务端没有返回可读的错误信息)"

    def _rejection_hint(self) -> str:
        return (
            f"排查顺序：\n"
            f"  1) 运行 python -m pipeline keys。它能区分「密钥无效」和「密钥有效但模型名不可用」——"
            f"后者同样返回 403，报错看起来像权限问题，实际是模型名对不上。\n"
            f"  2) 确认 {self.config.api_key_env} 是在国内站创建、状态为「启用」"
            f"（国际站的密钥不能用于国内站点）；\n"
            f"  3) 确认账号已开通 name={self.config.name!r} 这个模型、且没有欠费；\n"
            f"  4) 核对 base_url={self.config.base_url!r} 与上面的密钥是否配套。"
        )

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}{endpoint}"
        last_error: Exception | None = None

        for attempt in range(1, self.retry + 1):
            try:
                response = self._client.post(url, headers=self._headers(), json=payload)
            except httpx.HTTPError as exc:  # 超时、连接中断等
                last_error = exc
            else:
                if response.status_code < 400:
                    return response.json()
                if response.status_code in (408, 409, 429) or response.status_code >= 500:
                    last_error = LLMError(
                        f"{self.config.role} 调用失败（HTTP {response.status_code}）："
                        f"{response.text[:400]}"
                    )
                else:
                    raise LLMError(
                        f"{self.config.role} 调用被拒绝（HTTP {response.status_code}）："
                        f"{self._error_detail(response)}\n{self._rejection_hint()}"
                    )

            if attempt < self.retry:
                time.sleep(self.backoff_s * attempt)

        raise LLMError(
            f"{self.config.role} 连续 {self.retry} 次调用失败，最后错误：{last_error}"
        )

    # ---------- 高层能力 ----------
    def _chat_payload(
        self,
        system: str,
        user: str,
        *,
        images: Sequence[Path] = (),
        max_tokens: int,
    ) -> dict[str, Any]:
        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": user}]
            for path in images:
                content.append(
                    {"type": "image_url", "image_url": {"url": image_to_data_url(path)}}
                )
            user_content: Any = content
        else:
            user_content = user
        return {
            "model": self.config.name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": self.config.temperature,
        }

    def _chat_with_escalation(
        self,
        system: str,
        user: str,
        *,
        images: Sequence[Path] = (),
        max_tokens: int | None = None,
    ) -> str:
        """输出被推理过程挤空时，自动加大额度重试。

        实测推理模型的思考长度不可预测：同一批页面里大部分额度够用，
        个别页照样被吃光。自动翻倍重试比让人盯着报错去改配置靠谱。
        """
        cap = max_tokens or self.config.max_tokens
        last_error: Exception | None = None
        for _ in range(3):
            payload = self._chat_payload(system, user, images=images, max_tokens=cap)
            try:
                return self._extract_text(self._post("/chat/completions", payload))
            except OutputTruncated as exc:
                last_error = exc
                cap = min(cap * 2, 32768)
        raise LLMError(
            f"连续 3 次都因为思考过程占满额度而没写出答案，最后试到 {cap}。\n"
            f"可以把 config/models.yaml 里该模型的 max_tokens 调得更大，"
            f"或换一个非推理模型。\n{last_error}"
        )

    def chat(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        return self._chat_with_escalation(system, user, max_tokens=max_tokens)

    def chat_with_images(
        self,
        *,
        system: str,
        user: str,
        image_paths: Sequence[Path],
        max_tokens: int | None = None,
    ) -> str:
        if not image_paths:
            return self.chat(system=system, user=user, max_tokens=max_tokens)
        return self._chat_with_escalation(
            system, user, images=image_paths, max_tokens=max_tokens
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.config.name, "input": list(texts)}
        data = self._post("/embeddings", payload)
        items = data.get("data") or []
        if len(items) != len(texts):
            raise LLMError(
                f"向量化返回条数不符：请求 {len(texts)} 条，返回 {len(items)} 条。"
                f"服务商：{self.config.base_url}"
            )
        items = sorted(items, key=lambda item: item.get("index", 0))
        return [list(item["embedding"]) for item in items]

    def rerank(
        self, *, query: str, documents: Sequence[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        """给「问题—段落」两两打分，返回 (原始下标, 相关度分)。

        交叉编码器把问题和段落拼在一起过一遍模型，直接判断「这段话能不能回答这个问题」。
        这和向量余弦是两回事：向量只衡量语义接近，实测会出现「话题相邻但没答案」
        被判成有依据的情况（例如问二端口网络的 Z 参数，召回了一堆电容的内容）。
        """
        if not documents:
            return []
        payload: dict[str, Any] = {
            "model": self.config.name,
            "query": query,
            "documents": list(documents),
            "return_documents": False,
        }
        if top_n:
            payload["top_n"] = min(top_n, len(documents))

        data = self._post("/rerank", payload)
        results = data.get("results") or []
        scored: list[tuple[int, float]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index", 0))
                score = float(item.get("relevance_score", 0.0))
            except (TypeError, ValueError):
                continue
            scored.append((index, score))
        if not scored:
            raise LLMError(f"重排返回里没有 results：{str(data)[:300]}")
        return scored

    # ---------- 解析响应 ----------
    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"模型返回里没有 choices：{str(data)[:400]}")
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content")
        if isinstance(text, list):  # 部分服务商返回分段结构
            text = "".join(
                part.get("text", "") for part in text if isinstance(part, dict)
            )
        if not text:
            # 推理模型（实测 deepseek-flash、deepseek-v4-pro）会把思考过程放在
            # reasoning_content 里。如果 content 为空而 finish_reason 是 length，
            # 说明输出额度被思考过程吃光了——答案一个字都还没写就被截断。
            # 这时必须把话说清楚，否则使用者只看到「模型返回内容为空」，无从下手。
            reasoning = message.get("reasoning_content")
            finish = choice.get("finish_reason")
            if reasoning:
                raise OutputTruncated(
                    f"模型只输出了思考过程、没有输出答案（finish_reason={finish}）："
                    f"max_tokens 被思考过程用光了。思考过程开头：{str(reasoning)[:120]}"
                )
        if not text:
            raise LLMError(f"模型返回内容为空：{str(data)[:400]}")
        return str(text)

    # ---------- 批量向量化（自动分批） ----------
    def embed_batched(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> Iterable[list[float]]:
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            yield from self.embed(batch)
