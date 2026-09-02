"""Extract deterministic text and flattened yellow highlights from two PDFs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import pymupdf


WHITESPACE = re.compile(r"\s+")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_text(text: str) -> str:
    return WHITESPACE.sub(" ", text.replace("\u00ad", "").replace("\ufb01", "fi")).strip()


def is_yellow(color: tuple[float, ...] | None) -> bool:
    if color is None or len(color) < 3:
        return False
    red, green, blue = color[:3]
    return red >= 0.70 and green >= 0.55 and blue <= 0.45


def _word_selected(word_rect: pymupdf.Rect, yellow_rects: Iterable[pymupdf.Rect]) -> bool:
    area = max(word_rect.get_area(), 1e-9)
    for yellow_rect in yellow_rects:
        overlap = word_rect & yellow_rect
        if not overlap.is_empty and overlap.get_area() / area >= 0.15:
            return True
        center = pymupdf.Point((word_rect.x0 + word_rect.x1) / 2,
                       (word_rect.y0 + word_rect.y1) / 2)
        if yellow_rect.contains(center):
            return True
    return False


def extract_highlighted_lines(page: pymupdf.Page) -> list[dict[str, Any]]:
    yellow_rects = [drawing["rect"] for drawing in page.get_drawings()
                    if is_yellow(drawing.get("fill"))]
    lines: dict[tuple[int, int], list[tuple[int, str, pymupdf.Rect, bool]]] = {}
    for item in page.get_text("words", sort=True):
        x0, y0, x1, y1, text, block, line, word = item
        rect = pymupdf.Rect(x0, y0, x1, y1)
        lines.setdefault((block, line), []).append(
            (word, text, rect, _word_selected(rect, yellow_rects)))

    output = []
    for (block, line), words in sorted(lines.items()):
        if not any(item[3] for item in words):
            continue
        words.sort(key=lambda item: item[0])
        rect = pymupdf.Rect(words[0][2])
        for _, _, word_rect, _ in words[1:]:
            rect.include_rect(word_rect)
        output.append({
            "block": block,
            "line": line,
            "bbox": [round(value, 3) for value in rect],
            "selected_text": normalize_text(" ".join(item[1] for item in words if item[3])),
            "text": normalize_text(" ".join(item[1] for item in words)),
        })
    return output


def extract_document(path: Path) -> dict[str, Any]:
    with pymupdf.open(path) as document:
        pages = []
        annotation_count = 0
        for number, page in enumerate(document, start=1):
            annotations = list(page.annots() or [])
            annotation_count += len(annotations)
            text = normalize_text(page.get_text("text", sort=True))
            pages.append({
                "page": number,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "highlighted_lines": extract_highlighted_lines(page),
            })
        return {
            "filename": path.name,
            "sha256": sha256(path),
            "page_count": len(document),
            "annotation_count": annotation_count,
            "metadata": {key: value or "" for key, value in sorted(document.metadata.items())},
            "pages": pages,
        }


def compare_pages(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    old_hashes = [page["text_sha256"] for page in old["pages"]]
    new_hashes = [page["text_sha256"] for page in new["pages"]]
    matcher = SequenceMatcher(a=old_hashes, b=new_hashes, autojunk=False)
    return [{"operation": tag, "old_pages": [i1 + 1, i2], "new_pages": [j1 + 1, j2]}
            for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]


def _document_tokens(document: dict[str, Any]) -> list[tuple[str, int]]:
    return [(token, page["page"]) for page in document["pages"]
            for token in page["text"].split()]


def compare_text(old: dict[str, Any], new: dict[str, Any], context: int = 12) -> list[dict[str, Any]]:
    old_tokens = _document_tokens(old)
    new_tokens = _document_tokens(new)
    matcher = SequenceMatcher(a=[item[0] for item in old_tokens],
                              b=[item[0] for item in new_tokens])
    differences = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_slice = old_tokens[max(0, i1 - context):min(len(old_tokens), i2 + context)]
        new_slice = new_tokens[max(0, j1 - context):min(len(new_tokens), j2 + context)]
        differences.append({
            "operation": tag,
            "old_pages": sorted({item[1] for item in old_tokens[i1:i2]}),
            "new_pages": sorted({item[1] for item in new_tokens[j1:j2]}),
            "old_context": " ".join(item[0] for item in old_slice),
            "new_context": " ".join(item[0] for item in new_slice),
        })
    return differences


def build_audit(old_path: Path, new_path: Path) -> dict[str, Any]:
    old = extract_document(old_path)
    new = extract_document(new_path)
    return {
        "schema_version": "1.0",
        "extractor": {"name": "PyMuPDF", "version": pymupdf.__version__},
        "old": old,
        "new": new,
        "page_differences": compare_pages(old, new),
        "text_differences": compare_text(old, new),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_pdf", type=Path)
    parser.add_argument("new_pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit(args.old_pdf.resolve(), args.new_pdf.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}: {audit['old']['page_count']} -> "
          f"{audit['new']['page_count']} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
