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
                        f"{response.text[:400]}\n"
                        f"请核对 config/models.yaml 里的 name/base_url 与 {self.config.api_key_env} 是否匹配。"
                    )

            if attempt < self.retry:
                time.sleep(self.backoff_s * attempt)

        raise LLMError(
            f"{self.config.role} 连续 {self.retry} 次调用失败，最后错误：{last_error}"
        )

    # ---------- 高层能力 ----------
    def chat(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        payload = {
            "model": self.config.name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        return self._extract_text(self._post("/chat/completions", payload))

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

        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for path in image_paths:
            content.append(
                {"type": "image_url", "image_url": {"url": image_to_data_url(path)}}
            )

        payload = {
            "model": self.config.name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        return self._extract_text(self._post("/chat/completions", payload))

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

    # ---------- 解析响应 ----------
    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"模型返回里没有 choices：{str(data)[:400]}")
        message = choices[0].get("message") or {}
        text = message.get("content")
        if isinstance(text, list):  # 部分服务商返回分段结构
            text = "".join(
                part.get("text", "") for part in text if isinstance(part, dict)
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
