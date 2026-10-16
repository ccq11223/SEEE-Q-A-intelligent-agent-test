"""云数据库体检：连接能不能通、建表脚本跑了没有。

在把知识库写入云端之前跑一次，避免出现「向量化跑了几分钟、
最后一步才发现表不存在」这种白干。

用法：
    python scripts/check_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.cli import configure_stdio  # noqa: E402
from pipeline.dsn import inspect_dsn  # noqa: E402
from pipeline.env import load_env_file, get_env  # noqa: E402

EXPECTED_TABLES = [
    "courses",
    "documents",
    "chunks",
    "questions",
    "diagnostic_sets",
    "diagnostic_items",
    "student_sessions",
    "diagnostics",
    "qa_logs",
    "gaps",
    "error_ledger",
    "parse_errors",
]


def main() -> int:
    configure_stdio()
    root = Path(__file__).resolve().parents[1]
    load_env_file(root / ".env")

    dsn = get_env("SUPABASE_DB_URL")
    if not dsn:
        print("✗ .env 里没有 SUPABASE_DB_URL")
        return 1

    report = inspect_dsn(dsn)
    print(f"连接串：主机 {report.host}｜库 {report.dbname}｜用户 {report.user}")
    if not report.ok:
        print("✗ 连接串体检未通过：")
        for problem in report.problems:
            print(f"  · {problem}")
        return 1

    try:
        import psycopg
    except ImportError:
        print("✗ 未安装 psycopg，请执行 pip install \"psycopg[binary]\"")
        return 1

    try:
        with psycopg.connect(dsn, connect_timeout=20) as conn:
            with conn.cursor() as cur:
                cur.execute("select extname from pg_extension order by 1")
                extensions = [row[0] for row in cur.fetchall()]
                print("✓ 连接成功")
                has_vector = "vector" in extensions
                print(f"{'✓' if has_vector else '✗'} pgvector 扩展：{'已安装' if has_vector else '未安装'}")

                cur.execute(
                    "select table_name from information_schema.tables"
                    " where table_schema = 'public' order by 1"
                )
                tables = [row[0] for row in cur.fetchall()]
                missing = [name for name in EXPECTED_TABLES if name not in tables]
                print(
                    f"{'✓' if not missing else '✗'} 数据表："
                    f"已建 {len(tables)} 张，缺 {len(missing)} 张"
                )
                if tables:
                    print(f"    实际存在：{', '.join(tables)}")
                if missing:
                    print(f"    缺少：{', '.join(missing)}")
                    print()
                    print("下一步：照 db/README.md 把 db/migrations/*.sql 在 Supabase 的")
                    print("SQL Editor 里依次执行。注意顺序：001 → 002 → 003。")
                    return 1

                # 顺手看一眼课程种子数据在不在
                cur.execute("select count(*) from courses")
                course_count = cur.fetchone()[0]
                if course_count:
                    print(f"✓ courses 表有 {course_count} 门课程（002_seed.sql 已执行）")
                else:
                    print("✗ courses 表是空的，请执行 db/migrations/002_seed.sql")
                    return 1

                cur.execute("select count(*) from chunks")
                print(f"   chunks 表当前 {cur.fetchone()[0]} 块")
    except Exception as exc:  # noqa: BLE001 - 这里就是要如实报出任何连接错误
        print(f"✗ 连不上数据库：{type(exc).__name__}: {exc}")
        print()
        print("常见原因：")
        print("  · 用的是直连地址（db.xxx.supabase.co），Supabase 给它的常常只有 IPv6。")
        print("    国内很多网络没有 IPv6，改用在 Connect 面板里的 Session pooler 地址即可。")
        print("  · 数据库口令里有 / @ % 空格没转义（python -m pipeline doctor 会指出）。")
        print("  · 项目已暂停（免费层长期不用会休眠），去 Supabase 控制台恢复。")
        return 1

    print()
    print("云数据库已就绪，可以把 config/pipeline.yaml 的 storage.backend 改成 postgres，")
    print("然后执行：python -m pipeline load --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
