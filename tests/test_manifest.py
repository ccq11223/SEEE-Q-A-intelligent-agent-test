from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.manifest import ManifestError, load_manifest, select

HEADER = "doc_key,course_id,doc_type,trust_level,filename,title,term,priority,notes\n"


class TestManifest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "manifest.csv"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, body: str) -> None:
        self.path.write_text(HEADER + body, encoding="utf-8")

    def test_loads_valid_rows(self) -> None:
        self.write("a,circuit-theory-1,slide,1,ch01.pptx,第01章,2026秋,1,备注\n")
        entries = load_manifest(self.path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].doc_key, "a")
        self.assertEqual(entries[0].trust_level, 1)
        self.assertEqual(entries[0].doc_type_label, "串讲 PPT")

    def test_missing_file_raises_with_guidance(self) -> None:
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(Path(self.tmp.name) / "nope.csv")
        self.assertIn("manifest.example.csv", str(ctx.exception))

    def test_missing_column_is_reported(self) -> None:
        self.path.write_text("doc_key,course_id\n", encoding="utf-8")
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(self.path)
        self.assertIn("缺少列", str(ctx.exception))

    def test_duplicate_doc_key_is_reported(self) -> None:
        self.write(
            "a,circuit-theory-1,slide,1,f.pptx,t,2026秋,1,\n"
            "a,circuit-theory-1,slide,1,g.pptx,t2,2026秋,2,\n"
        )
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(self.path)
        self.assertIn("重复", str(ctx.exception))

    def test_bad_trust_level_is_reported(self) -> None:
        self.write("a,circuit-theory-1,slide,9,f.pptx,t,2026秋,1,\n")
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(self.path)
        self.assertIn("trust_level", str(ctx.exception))

    def test_unknown_doc_type_is_reported(self) -> None:
        self.write("a,circuit-theory-1,homework,1,f.pptx,t,2026秋,1,\n")
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(self.path)
        self.assertIn("doc_type", str(ctx.exception))

    def test_blank_lines_are_skipped(self) -> None:
        self.write("a,circuit-theory-1,slide,1,f.pptx,t,2026秋,1,\n,,,,,,,,\n")
        self.assertEqual(len(load_manifest(self.path)), 1)

    def test_select_sorts_by_priority(self) -> None:
        self.write(
            "later,circuit-theory-1,slide,1,f1.pptx,后做,2026秋,9,\n"
            "first,circuit-theory-1,textbook,1,f2.pdf,先做,2026秋,1,\n"
            "second,circuit-theory-1,web,3,f3.html,网络,2026秋,5,\n"
        )
        entries = load_manifest(self.path)
        ordered = [entry.doc_key for entry in select(entries, course_id="circuit-theory-1")]
        self.assertEqual(ordered, ["first", "second", "later"])

    def test_select_filters_by_trust_level(self) -> None:
        self.write(
            "a,circuit-theory-1,slide,1,f1.pptx,A,2026秋,1,\n"
            "b,circuit-theory-1,web,3,f2.html,B,2026秋,2,\n"
        )
        entries = load_manifest(self.path)
        limited = select(entries, max_trust=1)
        self.assertEqual([entry.doc_key for entry in limited], ["a"])

    def test_select_rejects_unknown_doc_key(self) -> None:
        self.write("a,circuit-theory-1,slide,1,f.pptx,t,2026秋,1,\n")
        entries = load_manifest(self.path)
        with self.assertRaises(ManifestError):
            select(entries, doc_keys=["missing"])

    def test_local_path_prefers_original_filename_then_doc_key(self) -> None:
        from tests.support import make_settings

        self.write("key1,circuit-theory-1,slide,1,原文件.pptx,标题,2026秋,1,\n")
        entry = load_manifest(self.path)[0]
        with tempfile.TemporaryDirectory() as tmp:
            settings = make_settings(Path(tmp))
            (settings.paths.raw_dir / "key1.pptx").write_text("x", encoding="utf-8")
            self.assertEqual(entry.local_path(settings.paths).name, "key1.pptx")


if __name__ == "__main__":
    unittest.main()
