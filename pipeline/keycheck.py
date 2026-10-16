"""密钥体检：确认每个密钥有效、填在了正确的变量上、且能访问配置里的模型。

只调用各家的 ``/models`` 列表接口——这是一个**免费、不消耗额度、不产生内容**的
只读接口，用来回答三个问题：

1. 这个密钥能不能通过鉴权（填错了吗、复制少了吗）；
2. 它是不是这一家的密钥（把 DeepSeek 的密钥填到百炼那一行，这里会立刻暴露）；
3. 配置里的模型名，这个账号用得上吗（名字打错、账号没开通，这里能提前发现）。

刻意**不**做「拿 A 家的密钥去试 B 家」的交叉验证：那等于把密钥发给第三方，
不划算也不礼貌。密钥鉴权失败时直接提示「确认没把别家的填到这一行」就够了。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class KeyProbe:
    role: str
    key_env: str
    model: str
    base_url: str
    status: str = "unknown"  # ok / missing / unauthorized / not_found / network / error
    http_status: int | None = None
    model_found: str | None = None
    model_count: int = 0
    detail: str = ""
    notes: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    all_models: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        """密钥能用**且**模型名对得上，才算真的可用。

        只判 status 会掩盖一种很坑的情况：密钥有效、模型名写错，此时建库会在
        解析阶段整份失败（HTTP 403），使用者却拿着「三个密钥都能用」的结论。
        """
        return self.status == "ok" and self.model_found is not None

    @property
    def model_missing(self) -> bool:
        return self.status == "ok" and self.model_found is None


#: 模型名对不上时，按用途猜使用者真正想要的是哪些
ROLE_KEYWORDS = {
    "vision": ("vl", "vision", "ocr"),
    "embedding": ("embed", "bge", "gte", "rerank"),
    "generate": ("chat", "instruct", "deepseek", "plus", "max"),
}


def suggest_models(
    candidates: list[str], wanted: str, role: str, *, limit: int = 12
) -> list[str]:
    """从可用列表里挑出「可能是你想写的那个名字」。

    先按名字前缀逐级放宽（qwen-vl-max-latest → qwen-vl-max → qwen-vl），
    前缀都对不上时按用途关键词兜底（视觉找含 vl 的、向量找含 bge 的）。

    逐级放宽时**累积**各层的结果，而不是取第一层就停：
    这个函数只在「一个都没匹配上」时才会被调用，
    此时多列几个同族模型（qwen-vl-max 和 qwen-vl-plus）比只列一个更有用。
    """
    parts = wanted.lower().replace("_", "-").split("-")
    ordered: list[str] = []
    for cut in range(len(parts), 0, -1):
        prefix = "-".join(parts[:cut])
        if len(prefix) < 5:
            continue
        for item in candidates:
            if item.lower().startswith(prefix) and item not in ordered:
                ordered.append(item)
        if len(ordered) >= limit:
            return ordered[:limit]

    if ordered:
        return ordered[:limit]

    keywords = ROLE_KEYWORDS.get(role, ())
    hits = [item for item in candidates if any(word in item.lower() for word in keywords)]
    return hits[:limit]


def find_model(candidates: list[str], wanted: str) -> str | None:
    """在可用模型列表里找配置的模型名。

    允许一点模糊：百炼把 ``qwen-vl-max`` 和 ``qwen-vl-max-latest`` 分开列，
    名字带 ``-latest`` 后缀时允许退一步匹配。
    """
    if wanted in candidates:
        return wanted

    if wanted.endswith("-latest"):
        base = wanted[: -len("-latest")]
        if base in candidates:
            return base

    for candidate in candidates:
        if candidate.startswith(wanted) or wanted.startswith(candidate):
            return candidate
    return None


def _model_ids(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    items = payload.get("data") or payload.get("models") or []
    ids: list[str] = []
    for item in items:
        if isinstance(item, dict):
            for key in ("id", "model", "name"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    ids.append(value)
                    break
        elif isinstance(item, str):
            ids.append(item)
    return ids


def probe_key(
    *,
    role: str,
    key_env: str,
    api_key: str | None,
    base_url: str,
    model: str,
    timeout_s: float = 25.0,
    client: httpx.Client | None = None,
) -> KeyProbe:
    probe = KeyProbe(role=role, key_env=key_env, model=model, base_url=base_url)

    if not api_key:
        probe.status = "missing"
        probe.detail = f".env 里 {key_env} 是空的（或整行没写）"
        return probe

    url = f"{base_url.rstrip('/')}/models"
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout_s)
    try:
        response = http.get(url, headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError as exc:
        probe.status = "network"
        probe.detail = f"请求不通：{type(exc).__name__}: {exc}"
        probe.notes.append("能上网页不代表命令行能出网；检查代理、公司网络或防火墙。")
        return probe
    finally:
        if owns_client:
            http.close()

    probe.http_status = response.status_code
    if response.status_code in (401, 403):
        probe.status = "unauthorized"
        body = (response.text or "").strip()
        try:
            payload = response.json()
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                body = str(error.get("message") or error.get("code") or body)
            elif isinstance(error, str):
                body = error
            elif isinstance(payload, dict) and payload.get("message"):
                body = str(payload["message"])
        except ValueError:
            pass
        probe.detail = f"鉴权被拒绝（HTTP {response.status_code}）：{body[:200]}"
        probe.notes.append(
            f"要么这个密钥不属于 {base_url}，要么已失效/被删。"
            f"确认没把别家的密钥填到 {key_env} 这一行。"
        )
        probe.notes.append(
            "若这是阿里云百炼：再确认密钥在国内站创建且状态为「启用」、账号已开通该模型、且没有欠费。"
        )
        return probe
    if response.status_code == 404:
        probe.status = "not_found"
        probe.detail = "服务地址不对（HTTP 404）"
        probe.notes.append(f"核对 config/models.yaml 里的 base_url：{base_url}")
        return probe
    if response.status_code >= 400:
        probe.status = "error"
        probe.detail = f"HTTP {response.status_code}：{response.text[:200]}"
        return probe

    try:
        ids = _model_ids(response.json())
    except ValueError:
        probe.status = "error"
        probe.detail = "返回内容不是 JSON，可能被网关或代理改写"
        return probe

    probe.model_count = len(ids)
    probe.all_models = ids
    probe.model_found = find_model(ids, model)
    if probe.model_found is None:
        probe.status = "ok"
        probe.detail = f"密钥有效，但可用模型列表（{len(ids)} 个）里没有 {model}"
        probe.suggestions = suggest_models(ids, model, role)
        if probe.suggestions:
            probe.notes.append(
                "这个账号里名字相近、可用的模型有："
                + "、".join(probe.suggestions)
                + f"。把 config/models.yaml 里 {role}.name 换成其中之一即可。"
            )
        else:
            probe.notes.append(
                f"列表里连相近的 {role} 模型都没有：可能是这个账号没开通该类模型，"
                f"也可能该地址的 /models 不列出这类模型。"
            )
        return probe

    probe.status = "ok"
    probe.detail = f"密钥有效，模型可用（列表 {len(ids)} 个）"
    if probe.model_found != model:
        probe.notes.append(f"实际匹配到的是 {probe.model_found}，与配置的 {model} 略有出入。")
    return probe
