"""无依据时的联网兜底（方案 §4.2 的 ⑥ 分支）。

设计上刻意保守：
  - 默认关闭（config/retrieval.yaml 的 allow_web=false），关着的时候一律转人工；
  - 打开后只做「概念题/简单题」的兜底，回答必须带「来自网络、未经审核」的抬头；
  - 无论成败都记进知识缺口清单，绝不让网络内容静默进入知识库。
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .env import get_env

WEB_BANNER = "> ⚠️ 以下内容来自网络搜索，未经审核，仅供参考；与课件不一致时以课件为准。\n\n"


@dataclass
class WebResult:
    text: str
    sources: list[str]


class WebSearchUnavailable(RuntimeError):
    """没配搜索 Key 或搜索失败——上层据此回落为转人工。"""


def search(question: str, *, max_results: int = 5, timeout_s: float = 20.0) -> WebResult:
    api_key = get_env("TAVILY_API_KEY")
    if not api_key:
        raise WebSearchUnavailable(
            "未配置 TAVILY_API_KEY，联网兜底不可用（这是默认状态，会直接转人工）。"
        )

    try:
        response = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": question,
                "max_results": max_results,
                "include_answer": True,
            },
            timeout=timeout_s,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise WebSearchUnavailable(f"联网搜索失败：{exc}") from exc

    payload = response.json()
    answer = (payload.get("answer") or "").strip()
    sources = [item.get("url", "") for item in payload.get("results") or []]
    pieces = [answer] if answer else []
    for item in (payload.get("results") or [])[:max_results]:
        title = (item.get("title") or "").strip()
        content = (item.get("content") or "").strip()
        if content:
            pieces.append(f"- {title}：{content}")

    if not pieces:
        raise WebSearchUnavailable("联网搜索没有返回可用内容。")

    return WebResult(text=WEB_BANNER + "\n".join(pieces), sources=sources)
