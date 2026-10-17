"""课件没覆盖时的联网检索。

支持三个服务，按环境变量自动挑选，谁配了用谁：

* Tavily：``TAVILY_API_KEY``（英文/技术资料较好，自带一段摘要）
* 博查 Bocha：``BOCHA_API_KEY``（中文资料较好）
* SearXNG：``SEARXNG_URL``（自建或公共实例，需开启 JSON 输出）

一个都没配时抛 :class:`WebSearchUnavailable`，上层自动回落为「通用知识作答」，
**不再直接拒答**。检索结果只作为提示词里的参考材料，由生成模型组织成一段
连贯的回答；同时把 URL 原样带给前端，学生自己能点开核对。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from .env import get_env

# 刻意只用 GBK 也能显示的字符：这份文案会出现在学生看到的回答里，
# 在中文 Windows 控制台或旧终端里，emoji 只会变成问号。
WEB_BANNER = (
    "> **注意**：以下内容来自网络搜索，未经审核，仅供参考；"
    "与课件不一致时一律以课件为准。\n\n"
)

#: 提示词里给模型的材料上限，避免一次塞进几万字
MAX_CONTEXT_CHARS = 6000


@dataclass(frozen=True)
class WebHit:
    title: str = ""
    url: str = ""
    content: str = ""


@dataclass
class WebResult:
    hits: list[WebHit] = field(default_factory=list)
    answer: str = ""
    provider: str = ""

    @property
    def sources(self) -> list[str]:
        return [hit.url for hit in self.hits if hit.url]

    @property
    def context(self) -> str:
        """拼给生成模型的材料（不带抬头，抬头在最终回答里由模型输出）。"""
        pieces: list[str] = []
        if self.answer:
            pieces.append(self.answer.strip())
        for index, hit in enumerate(self.hits, start=1):
            body = hit.content.strip()
            if not body:
                continue
            title = hit.title.strip() or f"搜索结果 {index}"
            source = f"（{hit.url}）" if hit.url else ""
            pieces.append(f"[{index}] {title}{source}\n{body}")
        text = "\n\n".join(pieces).strip()
        return text[:MAX_CONTEXT_CHARS]

    @property
    def text(self) -> str:
        """带「网络来源」抬头的完整文本，给不需要再过模型的场景用。"""
        return WEB_BANNER + self.context


class WebSearchUnavailable(RuntimeError):
    """没配搜索 Key 或搜索失败——上层据此回落为通用知识作答。"""


def configured_providers() -> list[str]:
    """按优先级列出已配置的搜索服务。"""
    providers: list[str] = []
    if get_env("TAVILY_API_KEY"):
        providers.append("tavily")
    if get_env("BOCHA_API_KEY"):
        providers.append("bocha")
    if get_env("SEARXNG_URL"):
        providers.append("searxng")
    return providers


def configured_provider() -> str:
    """当前会优先用的搜索服务名；没配置返回空串。"""
    providers = configured_providers()
    return providers[0] if providers else ""


def _clean(value: object) -> str:
    return str(value or "").strip()


def _tavily(question: str, key: str, limit: int, timeout: float) -> WebResult:
    response = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": key,
            "query": question,
            "max_results": limit,
            "include_answer": True,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    hits = [
        WebHit(
            title=_clean(item.get("title")),
            url=_clean(item.get("url")),
            content=_clean(item.get("content")),
        )
        for item in payload.get("results") or []
    ]
    return WebResult(hits=hits, answer=_clean(payload.get("answer")), provider="tavily")


def _bocha(question: str, key: str, limit: int, timeout: float) -> WebResult:
    response = httpx.post(
        "https://api.bochaai.com/v1/web-search",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "query": question,
            "summary": True,
            "count": limit,
            "freshness": "noLimit",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or {}
    pages = (data.get("webPages") or {}).get("value") or []
    hits = [
        WebHit(
            title=_clean(item.get("name")),
            url=_clean(item.get("url")),
            content=_clean(item.get("summary") or item.get("snippet")),
        )
        for item in pages
    ]
    return WebResult(hits=hits, provider="bocha")


def _searxng(question: str, base_url: str, limit: int, timeout: float) -> WebResult:
    response = httpx.get(
        f"{base_url.rstrip('/')}/search",
        params={"q": question, "format": "json", "language": "zh-CN"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    hits = [
        WebHit(
            title=_clean(item.get("title")),
            url=_clean(item.get("url")),
            content=_clean(item.get("content")),
        )
        for item in (payload.get("results") or [])[:limit]
    ]
    return WebResult(hits=hits, provider="searxng")


def search(question: str, *, max_results: int = 5, timeout_s: float = 20.0) -> WebResult:
    providers = configured_providers()
    if not providers:
        raise WebSearchUnavailable(
            "未配置联网搜索（TAVILY_API_KEY / BOCHA_API_KEY / SEARXNG_URL 任一即可）"
        )

    errors: list[str] = []
    for provider in providers:
        try:
            if provider == "tavily":
                result = _tavily(
                    question, get_env("TAVILY_API_KEY"), max_results, timeout_s
                )
            elif provider == "bocha":
                result = _bocha(question, get_env("BOCHA_API_KEY"), max_results, timeout_s)
            else:
                result = _searxng(question, get_env("SEARXNG_URL"), max_results, timeout_s)
        except httpx.HTTPError as exc:
            errors.append(f"{provider}：{exc}")
            continue
        except ValueError as exc:  # JSON 解析失败
            errors.append(f"{provider}：返回格式异常（{exc}）")
            continue
        if result.context:
            return result
        errors.append(f"{provider}：没有返回可用内容")

    raise WebSearchUnavailable("联网搜索失败（" + "；".join(errors) + "）")
