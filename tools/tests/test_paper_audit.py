from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pymupdf

from tools.paper_audit import build_audit, extract_document


def make_pdf(path: Path, highlighted: bool) -> None:
    document = pymupdf.open()
    page = document.new_page()
    if highlighted:
        page.draw_rect(pymupdf.Rect(68, 67, 146, 85), color=None, fill=(1, 1, 0))
    page.insert_text((72, 80), "highlighted words", fontsize=11)
    page.insert_text((72, 110), "ordinary words", fontsize=11)
    document.save(path)
    document.close()


class PaperAuditTests(unittest.TestCase):
    def test_extracts_only_words_over_flattened_yellow_fill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "highlighted.pdf"
            make_pdf(path, highlighted=True)
            result = extract_document(path)
        lines = result["pages"][0]["highlighted_lines"]
        self.assertEqual([line["selected_text"] for line in lines], ["highlighted words"])
        self.assertEqual([line["text"] for line in lines], ["highlighted words"])
        self.assertEqual(result["annotation_count"], 0)

    def test_audit_is_json_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_path = root / "old.pdf"
            new_path = root / "new.pdf"
            make_pdf(old_path, highlighted=False)
            make_pdf(new_path, highlighted=True)
            first = json.dumps(build_audit(old_path, new_path), sort_keys=True)
            second = json.dumps(build_audit(old_path, new_path), sort_keys=True)
        self.assertEqual(first, second)
        result = json.loads(first)
        self.assertEqual(result["new"]["page_count"], 1)
        self.assertEqual(result["text_differences"], [])


if __name__ == "__main__":
    unittest.main()
