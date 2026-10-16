from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from pipeline import cli
from pipeline.manifest import load_manifest
from pipeline.register import (
    AUTO_NOTE,
    append_rows,
    build_drafts,
    guess_doc_type,
    guess_trust_level,
    next_doc_key,
    scan_files,
)
from pipeline.settings import Settings
from tests.support import copy_config_files


class TestGuessing(unittest.TestCase):
    def test_slide_keywords(self) -> None:
        self.assertEqual(guess_doc_type("电路理论串讲.pptx")[0], "slide")
        self.assertEqual(guess_doc_type("第3章 复习.pdf")[0], "slide")

    def test_exam_scope_keywords(self) -> None:
        self.assertEqual(guess_doc_type("2026秋考点范围.docx")[0], "exam_scope")

    def test_past_exam_keywords(self) -> None:
        self.assertEqual(guess_doc_type("2023年期末试题.pdf")[0], "past_exam")
        self.assertEqual(guess_doc_type("往年真题汇编.pdf")[0], "past_exam")

    def test_solution_wins_over_exam(self) -> None:
        self.assertEqual(guess_doc_type("2023试题解析.pdf")[0], "past_exam_solution")

    def test_example_keywords(self) -> None:
        self.assertEqual(guess_doc_type("典型例题精选.pdf")[0], "example")

    def test_textbook_keywords(self) -> None:
        self.assertEqual(guess_doc_type("教材重点第1章.pdf")[0], "textbook")

    def test_extension_fallback_is_flagged_as_guess(self) -> None:
        doc_type, is_guess = guess_doc_type("a1b2c3.pdf")
        self.assertEqual(doc_type, "slide")
        self.assertTrue(is_guess)

    def test_keyword_hit_is_not_a_guess(self) -> None:
        _, is_guess = guess_doc_type("串讲课件.pptx")
        self.assertFalse(is_guess)

    def test_trust_level_defaults_match_plan(self) -> None:
        self.assertEqual(guess_trust_level("slide"), 1)
        self.assertEqual(guess_trust_level("exam_scope"), 1)
        self.assertEqual(guess_trust_level("past_exam"), 2)
        self.assertEqual(guess_trust_level("web"), 3)


class TestScan(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def touch(self, name: str) -> Path:
        path = self.dir / name
        path.write_text("x", encoding="utf-8")
        return path

    def test_finds_supported_formats_only(self) -> None:
        self.touch("a.pptx")
        self.touch("b.pdf")
        self.touch("c.zip")
        self.touch("~$d.pptx")
        names = [path.name for path in scan_files(self.dir)]
        self.assertEqual(names, ["a.pptx", "b.pdf"])

    def test_recursive_option(self) -> None:
        nested = self.dir / "sub"
        nested.mkdir()
        (nested / "c.pdf").write_text("x", encoding="utf-8")
        self.assertEqual(scan_files(self.dir), [])
        self.assertEqual([p.name for p in scan_files(self.dir, recursive=True)], ["c.pdf"])

    def test_missing_directory_returns_empty(self) -> None:
        self.assertEqual(scan_files(self.dir / "nope"), [])


class TestDocKeys(unittest.TestCase):
    def test_sequence_starts_at_one(self) -> None:
        self.assertEqual(next_doc_key("slide", set()), "slide-01")

    def test_sequence_skips_taken_keys(self) -> None:
        taken = {"slide-01", "slide-02"}
        self.assertEqual(next_doc_key("slide", taken), "slide-03")
        self.assertIn("slide-03", taken)

    def test_underscore_type_becomes_hyphen(self) -> None:
        self.assertEqual(next_doc_key("exam_scope", set()), "exam-scope-01")


class TestBuildDrafts(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def touch(self, name: str) -> Path:
        path = self.dir / name
        path.write_text("x", encoding="utf-8")
        return path

    def test_generates_one_row_per_file(self) -> None:
        files = [self.touch("电路串讲.pptx"), self.touch("2023试题.pdf")]
        drafts = build_drafts(files, course_id="circuit-theory-1", term="2026秋")
        self.assertEqual(len(drafts), 2)
        self.assertEqual({draft.doc_type for draft in drafts}, {"slide", "past_exam"})
        self.assertTrue(all(draft.course_id == "circuit-theory-1" for draft in drafts))
        self.assertTrue(all(draft.term == "2026秋" for draft in drafts))

    def test_auto_note_marks_guessed_type(self) -> None:
        files = [self.touch("a1b2c3.pdf"), self.touch("串讲课件.pptx")]
        drafts = build_drafts(files, course_id="c1")
        notes = {draft.filename: draft.notes for draft in drafts}
        self.assertEqual(notes["a1b2c3.pdf"], AUTO_NOTE)
        self.assertEqual(notes["串讲课件.pptx"], "")

    def test_overrides_apply(self) -> None:
        files = [self.touch("随便.pdf")]
        drafts = build_drafts(
            files, course_id="c1", doc_type="exam_scope", trust_level=1
        )
        self.assertEqual(drafts[0].doc_type, "exam_scope")
        self.assertEqual(drafts[0].trust_level, 1)
        self.assertEqual(drafts[0].notes, "")

    def test_skips_already_registered_files(self) -> None:
        from pipeline.manifest import ManifestEntry

        files = [self.touch("已登记.pptx"), self.touch("新的.pptx")]
        existing = [
            ManifestEntry(
                doc_key="slide-01",
                course_id="c1",
                doc_type="slide",
                trust_level=1,
                filename="已登记.pptx",
                title="已登记",
            )
        ]
        drafts = build_drafts(files, course_id="c1", existing=existing)
        self.assertEqual([draft.filename for draft in drafts], ["新的.pptx"])
        self.assertEqual(drafts[0].doc_key, "slide-02")


class TestAppendRows(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "manifest.csv"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_writes_header_when_file_missing(self) -> None:
        files = [
            path
            for path in [
                (Path(self.tmp.name) / "串讲.pptx"),
                (Path(self.tmp.name) / "考点.docx"),
            ]
        ]
        for path in files:
            path.write_text("x", encoding="utf-8")
        drafts = build_drafts(files, course_id="c1", term="2026秋")
        self.assertEqual(append_rows(self.path, drafts), 2)

        entries = load_manifest(self.path)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].doc_key, drafts[0].doc_key)

    def test_preserves_existing_rows(self) -> None:
        self.path.write_text(
            "doc_key,course_id,doc_type,trust_level,filename,title,term,priority,notes\n"
            "keep,c1,slide,1,keep.pptx,保留,2026秋,1,\n",
            encoding="utf-8",
        )
        source = Path(self.tmp.name) / "串讲.pptx"
        source.write_text("x", encoding="utf-8")
        drafts = build_drafts([source], course_id="c1")
        append_rows(self.path, drafts)
        entries = load_manifest(self.path)
        self.assertEqual([entry.doc_key for entry in entries], ["keep", drafts[0].doc_key])

    def test_no_rows_is_noop(self) -> None:
        self.assertEqual(append_rows(self.path, []), 0)
        self.assertFalse(self.path.exists())


class TestRegisterCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        copy_config_files(self.root)
        self.settings = Settings.load(self.root)
        self.settings.paths.ensure()
        for name in ("电路理论串讲.pptx", "2026秋考点范围.docx"):
            (self.settings.paths.raw_dir / name).write_text("x", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_cli(self, argv: list[str]) -> tuple[int, str]:
        buffer = io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(buffer):
            code = cli.main(["--root", str(self.root), *argv])
        return code, buffer.getvalue()

    def test_dry_run_does_not_write(self) -> None:
        code, output = self.run_cli(["register", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("待登记", output)
        self.assertFalse((self.root / "config" / "manifest.csv").exists())

    def test_register_then_manifest_lists_rows(self) -> None:
        code, _ = self.run_cli(["register"])
        self.assertEqual(code, 0)
        manifest_path = self.root / "config" / "manifest.csv"
        self.assertTrue(manifest_path.is_file())
        self.assertEqual(len(load_manifest(manifest_path)), 2)

        code, output = self.run_cli(["manifest"])
        self.assertEqual(code, 0)
        self.assertIn("电路理论串讲", output)

    def test_register_is_idempotent(self) -> None:
        self.run_cli(["register"])
        code, output = self.run_cli(["register"])
        self.assertEqual(code, 0)
        self.assertIn("2 个已登记", output)
        self.assertEqual(len(load_manifest(self.root / "config" / "manifest.csv")), 2)

    def test_bad_doc_type_is_rejected(self) -> None:
        code, output = self.run_cli(["register", "--doc-type", "homework"])
        self.assertEqual(code, 2)
        self.assertIn("doc-type", output)

    def test_empty_directory_is_reported(self) -> None:
        empty = self.root / "empty"
        empty.mkdir()
        code, output = self.run_cli(["register", "--dir", str(empty)])
        self.assertEqual(code, 0)
        self.assertIn("没有找到", output)


if __name__ == "__main__":
    unittest.main()
