"""连接串体检的测试。

期望值全部来自 psycopg（libpq）真实解析器的实测行为，不是猜的。
"""

from __future__ import annotations

import unittest

from pipeline.dsn import (
    MUST_ESCAPE,
    bad_characters,
    escape_password,
    inspect_dsn,
)

try:
    import psycopg  # noqa: F401

    HAS_PSYCOPG = True
except ImportError:  # pragma: no cover
    HAS_PSYCOPG = False

GOOD = "postgresql://postgres.abc123:Pa-ss_Word-123_x@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"


class TestEscapePassword(unittest.TestCase):
    def test_dash_underscore_dot_tilde_untouched(self) -> None:
        self.assertEqual(escape_password("Pa-ss_Word.1~2"), "Pa-ss_Word.1~2")

    def test_alphanumeric_untouched(self) -> None:
        self.assertEqual(escape_password("Abc123"), "Abc123")

    def test_at_sign_escaped(self) -> None:
        self.assertEqual(escape_password("Pa@ss"), "Pa%40ss")

    def test_slash_escaped(self) -> None:
        self.assertEqual(escape_password("Pa/ss"), "Pa%2Fss")

    def test_percent_escaped(self) -> None:
        self.assertEqual(escape_password("Pa%ss"), "Pa%25ss")

    def test_space_escaped(self) -> None:
        self.assertEqual(escape_password("Pa ss"), "Pa%20ss")

    def test_every_escaped_form_is_accepted_by_parser(self) -> None:
        if not HAS_PSYCOPG:
            self.skipTest("未安装 psycopg")
        for raw in ("Pa@ss", "Pa/ss", "Pa%ss", "Pa ss", "Pa:ss", "Pa#ss"):
            encoded = escape_password(raw)
            report = inspect_dsn(f"postgresql://u:{encoded}@h.example:5432/db")
            self.assertTrue(report.ok, f"{raw!r} → {encoded!r} 应当通过体检")
            self.assertEqual(report.password, raw)


class TestBadCharacters(unittest.TestCase):
    def test_lists_only_must_escape_chars(self) -> None:
        self.assertEqual(bad_characters("Pa-ss_Word"), [])
        self.assertEqual(bad_characters("Pa@ss"), ["@"])
        self.assertEqual(bad_characters("a@b/c%d e"), ["@", "/", "%", " "])

    def test_deduplicates_in_order_of_appearance(self) -> None:
        self.assertEqual(bad_characters("/a@b/c"), ["/", "@"])

    def test_table_covers_the_four_real_problem_chars(self) -> None:
        self.assertEqual(set(MUST_ESCAPE), {"/", "@", "%", " "})


class TestInspectGoodDsn(unittest.TestCase):
    def setUp(self) -> None:
        if not HAS_PSYCOPG:
            self.skipTest("未安装 psycopg")

    def test_dash_and_underscore_password_is_fine(self) -> None:
        report = inspect_dsn(GOOD)
        self.assertTrue(report.ok, report.problems)
        self.assertEqual(report.password, "Pa-ss_Word-123_x")
        self.assertEqual(report.host, "aws-0-ap-southeast-1.pooler.supabase.com")
        self.assertEqual(report.dbname, "postgres")
        self.assertEqual(report.user, "postgres.abc123")
        self.assertEqual(report.hints, [])

    def test_colon_and_hash_in_password_are_tolerated(self) -> None:
        # libpq 实测：口令里的 : 和 # 会被原样保留，不必转义
        report = inspect_dsn("postgresql://u:Pa:ss#x@h.example:5432/db")
        self.assertTrue(report.ok, report.problems)
        self.assertEqual(report.password, "Pa:ss#x")


class TestInspectBadDsn(unittest.TestCase):
    def setUp(self) -> None:
        if not HAS_PSYCOPG:
            self.skipTest("未安装 psycopg")

    def test_empty_dsn(self) -> None:
        report = inspect_dsn("   ")
        self.assertFalse(report.ok)
        self.assertIn("空", " ".join(report.problems))

    def test_unencoded_at_sign_is_caught(self) -> None:
        report = inspect_dsn(
            "postgresql://postgres.abc:Pa@ss@aws-0.pooler.supabase.com:5432/postgres"
        )
        self.assertFalse(report.ok)
        self.assertIn("@", " ".join(report.problems))
        self.assertIn("%40", " ".join(report.hints))

    def test_space_is_caught(self) -> None:
        report = inspect_dsn("postgresql://u:Pa ss@h.example:5432/db")
        self.assertFalse(report.ok)
        self.assertIn("空格", " ".join(report.problems))

    def test_bad_percent_escape_is_caught(self) -> None:
        report = inspect_dsn("postgresql://u:Pa%ss@h.example:5432/db")
        self.assertFalse(report.ok)
        self.assertIn("%", " ".join(report.problems))

    def test_slash_in_password_is_caught(self) -> None:
        report = inspect_dsn("postgresql://u:Pa/ss@h.example:5432/db")
        self.assertFalse(report.ok)

    def test_missing_host_and_dbname(self) -> None:
        report = inspect_dsn("postgresql://u:pw@")
        self.assertFalse(report.ok)
        joined = " ".join(report.problems)
        self.assertIn("主机名", joined)
        self.assertIn("数据库名", joined)

    def test_hint_shows_escaped_password(self) -> None:
        report = inspect_dsn("postgresql://u:Pa@ss@h.example:5432/db")
        self.assertTrue(any("%40" in hint for hint in report.hints))

    def test_keyword_form_is_accepted(self) -> None:
        report = inspect_dsn("host=h.example port=5432 dbname=postgres user=u password=pw")
        self.assertTrue(report.ok, report.problems)
        self.assertEqual(report.host, "h.example")


if __name__ == "__main__":
    unittest.main()
