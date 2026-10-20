"""探测 Supabase 的 IPv4 连接池区域。

Supabase 直连地址 db.<ref>.supabase.co 可能只有 IPv6；
没有 IPv6 的服务器要改用 aws-0-<region>.pooler.supabase.com。
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote, urlparse

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.env import get_env, load_env_file  # noqa: E402


def main() -> int:
    load_env_file(ROOT / ".env")
    raw = get_env("SUPABASE_DB_URL") or ""
    url = urlparse(raw)
    host = url.hostname or ""
    password = quote(url.password or "", safe="")
    project = host.split(".")[1] if host.startswith("db.") else ""
    regions = [
        "ap-southeast-1",
        "ap-northeast-1",
        "ap-northeast-2",
        "ap-south-1",
        "us-east-1",
        "us-west-1",
        "eu-central-1",
        "eu-west-2",
    ]

    print(f"project={project} direct_host={host}")
    for region in regions:
        pooler = f"aws-0-{region}.pooler.supabase.com"
        dsn = (
            f"postgresql://postgres.{project}:{password}"
            f"@{pooler}:5432/postgres"
        )
        try:
            conn = psycopg.connect(dsn, connect_timeout=8)
            conn.execute("select 1")
            conn.close()
            print(f"OK {pooler}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"-- {pooler}: {str(exc).splitlines()[0][:90]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
