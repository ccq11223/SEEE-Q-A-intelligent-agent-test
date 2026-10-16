"""数据库连接串体检。

存在的理由：口令里的特殊字符会**悄悄**破坏连接串，而且失败时的报错完全不指向真因。
实测（psycopg / libpq 的真实解析器，2026-10-16）：

    口令            libpq 解析结果                            表现
    Pa-ss_Word      原样保留                                  ✓ 没问题
    Pa@ss           password='Pa'  host='ss@主机名'            ✗ 静默错位，连不上且报错指向错误
    Pa/ss           password=None  host='用户名'  dbname 错乱   ✗ 整串崩坏
    Pa%ss           直接抛 invalid percent-encoded token      ✗ 硬报错
    "Pa ss"         直接抛 unexpected spaces found            ✗ 硬报错

所以只有 4 个字符需要处理：``/ @ % 空格``，其余（含 ``- _ . ~``）可原样使用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import quote

#: 在 URI 里必须百分号转义的字符 → 转义写法
MUST_ESCAPE = {
    "/": "%2F",
    "@": "%40",
    "%": "%25",
    " ": "%20",
}

_BAD_PERCENT_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")


@dataclass
class DsnReport:
    ok: bool = True
    fields: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def host(self) -> str:
        return self.fields.get("host", "")

    @property
    def dbname(self) -> str:
        return self.fields.get("dbname", "")

    @property
    def user(self) -> str:
        return self.fields.get("user", "")

    @property
    def password(self) -> str:
        return self.fields.get("password", "")


def escape_password(password: str) -> str:
    """把口令转成可以安全放进 URI 的写法。

    ``urllib.parse.quote`` 会保留 RFC 3986 的 unreserved 字符
    （字母、数字、``-``、``_``、``.``、``~``），所以 ``-`` 和 ``_`` 原样不动。
    """
    return quote(password, safe="")


def bad_characters(password: str) -> list[str]:
    """列出口令里需要转义的字符（按出现顺序、去重）。"""
    seen: list[str] = []
    for char in password:
        if char in MUST_ESCAPE and char not in seen:
            seen.append(char)
    return seen


def inspect_dsn(dsn: str) -> DsnReport:
    """解析并体检连接串。不联网、不建立连接。"""
    report = DsnReport()
    if not dsn or not dsn.strip():
        report.ok = False
        report.problems.append("连接串是空的。")
        return report

    dsn = dsn.strip()
    is_uri = dsn.startswith(("postgres://", "postgresql://"))

    # ---- 1. URI 结构检查：这些坑解析器不报错，只会把主机名/库名悄悄改掉 ----
    # 正确形态：<scheme>://<用户>:<口令>@<主机>:<端口>/<库名>，
    # 所以 "://" 之后到第一个 "/" 之前的那一段（authority）里必须恰好有一个 @。
    if is_uri:
        authority = dsn.split("://", 1)[1].split("/", 1)[0]
        at_count = authority.count("@")

        if at_count == 0:
            report.ok = False
            report.problems.append(
                "连接串的「用户@主机」这一段里找不到 @。最常见的原因是口令里含 / ——"
                "它会被当成路径分隔符，把主机名整段截掉。"
            )
            report.hints.append("口令里的 / 必须写成 %2F。")
        elif at_count > 1:
            report.ok = False
            report.problems.append(
                f"「用户@主机」这一段里出现了 {at_count} 个 @，正常只应该有 1 个"
                f"（主机名前面那个）。多出来的应该是口令里的 @。"
            )
            report.hints.append("口令里的 @ 必须写成 %40，例如 Pa@ss 要写成 Pa%40ss。")

        if " " in dsn:
            report.ok = False
            report.problems.append("连接串里有空格。")
            report.hints.append("空格必须写成 %20；如果只是想分隔，说明多打了空格。")

        bad_pct = _BAD_PERCENT_RE.search(dsn)
        if bad_pct:
            report.ok = False
            report.problems.append(
                f"位置 {bad_pct.start() + 1} 处的 % 不是合法的百分号转义（应为 %XX）。"
            )
            report.hints.append("口令里的 % 必须写成 %25。")

    # ---- 2. 用 PostgreSQL 自己的解析器读一遍（权威口径）----
    try:
        from psycopg.conninfo import conninfo_to_dict
    except ImportError:
        report.hints.append(
            '未安装 psycopg，跳过深入体检（pip install "psycopg[binary]"）。'
        )
        return report

    try:
        report.fields = dict(conninfo_to_dict(dsn))
    except Exception as exc:  # psycopg.ProgrammingError 等
        report.ok = False
        report.problems.append(f"连接串无法解析：{exc}")
        if not report.hints:
            report.hints.append(
                "最常见的原因是口令里有 % 或空格：把 % 写成 %25、空格写成 %20 再试。"
            )
        return report

    # ---- 3. 必需字段与结构合理性 ----
    if not report.fields.get("host"):
        report.ok = False
        report.problems.append("解析不到主机名（host）。")
    dbname = report.fields.get("dbname") or ""
    if not dbname:
        report.ok = False
        report.problems.append("解析不到数据库名（dbname）。")
    elif any(char in dbname for char in "@/:"):
        report.ok = False
        report.problems.append(
            f"解析出的数据库名是 {dbname!r}，明显不对——正常的库名不会含 @ 或冒号。"
            f"通常是口令里有未转义的 @ 或 /，把连接串的结构撑坏了。"
        )
        report.hints.append("先把口令里需要转义的字符（/ @ % 空格）转义，再重新拼连接串。")

    # ---- 4. 口令本身是否有需要转义的字符 ----
    password = report.fields.get("password") or ""
    flagged = bad_characters(password)
    if flagged:
        shown = "、".join(f"{char}→{MUST_ESCAPE[char]}" for char in flagged)
        report.hints.append(
            f"口令里有需要转义的字符（{shown}）。"
            f"若连接失败优先怀疑这里；转义后应为 {escape_password(password)}"
        )

    return report
