"""Render and inspect the encoder chemistry deck and its vector figure exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DPI = 180


def inspect_document(pdf: Path, preview: Path, prefix: str) -> tuple[dict, list[Path], list[str]]:
    """Retain text and geometry diagnostics; rasterize only inspection previews."""
    preview.mkdir(parents=True, exist_ok=True)
    pages, images, text = [], [], []
    fonts: dict[int, dict] = {}
    edge_words, outside_words = [], []
    with pymupdf.open(pdf) as document:
        for index, page in enumerate(document):
            number = index + 1
            text.append(f"--- {prefix} {number} ---\n{page.get_text()}")
            image_path = preview / f"{prefix}-{number:02d}.png"
            page.get_pixmap(dpi=DPI, alpha=False).save(image_path)
            images.append(image_path)
            for x0, y0, x1, y1, word, *_ in page.get_text("words"):
                issue = {"page": number, "text": word, "bounds": [x0, y0, x1, y1]}
                if x0 < 12 or x1 > page.rect.width - 12 or y0 < 0 or y1 > page.rect.height:
                    edge_words.append(issue)
                if x0 < -.5 or x1 > page.rect.width + .5 or y0 < -.5 or y1 > page.rect.height + .5:
                    outside_words.append(issue)
            for xref, extension, kind, name, *_ in page.get_fonts(full=True):
                if xref not in fonts:
                    _, _, _, data = document.extract_font(xref)
                    fonts[xref] = {"xref": xref, "name": name, "type": kind,
                                   "extension": extension, "embedded": bool(data)}
            spans = [span for block in page.get_text("dict")["blocks"] if block["type"] == 0
                     for line in block["lines"] for span in line["spans"] if span["text"].strip()]
            pages.append({
                "page": number, "size_points": [page.rect.width, page.rect.height],
                "size_mm": [page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72],
                "raster_image_count": len(page.get_images(full=True)),
                "vector_path_count": len(page.get_drawings()),
                "minimum_font_points": min((float(span["size"]) for span in spans), default=None),
                "text_span_count": len(spans),
            })
    missing_fonts = [font for font in fonts.values() if not font["embedded"]]
    type3_fonts = [font for font in fonts.values() if font["type"].replace(" ", "").lower() == "type3"]
    raster_count = sum(page["raster_image_count"] for page in pages)
    return ({
        "pdf": str(pdf), "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "page_count": len(pages), "render_dpi": DPI, "pages": pages,
        "fonts": list(fonts.values()), "all_fonts_embedded": not missing_fonts,
        "type3_fonts": type3_fonts, "raster_image_count": raster_count,
        "vector_only_artwork": raster_count == 0,
        "near_edge_words": edge_words, "outside_page_words": outside_words,
        "passed": bool(pages) and not (outside_words or missing_fonts or type3_fonts or raster_count),
    }, images, text)


def contact_sheets(images: list[Path], output: Path, prefix: str, labels: list[str]) -> list[str]:
    columns, per_sheet = 3, 12
    cell_width, cell_height = 640, 384
    contacts = []
    for start in range(0, len(images), per_sheet):
        selected = images[start:start + per_sheet]
        canvas = Image.new("RGB", (columns * cell_width, math.ceil(len(selected) / columns) * cell_height), "#d9dcdf")
        draw = ImageDraw.Draw(canvas)
        for offset, path in enumerate(selected):
            with Image.open(path) as image:
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((cell_width - 8, cell_height - 28))
            x, y = (offset % columns) * cell_width, (offset // columns) * cell_height
            canvas.paste(thumbnail, (x + 4, y + 24))
            draw.text((x + 8, y + 6), labels[start + offset], fill="#0c0c0c")
        path = output / f"{prefix}-{start // per_sheet + 1}.jpg"
        canvas.save(path, quality=94)
        contacts.append(str(path))
    return contacts


def inspect(pdf_path: Path, output: Path, figures: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    preview = output / "preview"
    audit, images, text = inspect_document(pdf_path, preview, "slide")
    audit["contact_sheets"] = contact_sheets(images, preview, "contact", [f"Slide {i + 1:02d}" for i in range(len(images))])
    log_path = output / "encoder_chemistry.log"
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    warnings = [line for line in log_text.splitlines()
                if re.search(r"Overfull|Underfull|Missing character|LaTeX Warning|Package .* Warning", line)]
    hard_warnings = [line for line in warnings if "Overfull" in line or "Missing character" in line]
    audit.update(latex_log_present=log_path.is_file(), latex_warnings=warnings)

    figure_audits, figure_images, figure_labels = [], [], []
    figure_preview = preview / "figures"
    for path in sorted(figures.glob("*.pdf")):
        checked, rendered, extracted = inspect_document(path, figure_preview, path.stem)
        figure_audits.append(checked)
        figure_images.extend(rendered)
        figure_labels.extend(f"{path.stem} · page {i + 1}" for i in range(len(rendered)))
        text.extend(extracted)
    audit["standalone_figures"] = figure_audits
    audit["figure_contact_sheets"] = (contact_sheets(figure_images, figure_preview, "contact", figure_labels)
                                      if figure_images else [])
    audit["visual_review_limit"] = "Page bounds do not detect internal overlaps. Inspect contact sheets and full-resolution slides, including diagram connector endpoints and source/takeaway clearance."
    audit["passed"] = (audit["passed"] and log_path.is_file() and not hard_warnings
                       and all(figure["passed"] for figure in figure_audits))
    (output / "extracted_text.txt").write_text("\n\n".join(text))
    (output / "layout-audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "encoder_chemistry.pdf")
    parser.add_argument("--out", type=Path, default=ROOT / "build")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    audit = inspect(args.pdf.resolve(), args.out.resolve(), args.figures.resolve())
    print(json.dumps({key: audit[key] for key in ["pdf", "page_count", "render_dpi", "all_fonts_embedded",
                                                  "type3_fonts", "vector_only_artwork", "latex_log_present",
                                                  "contact_sheets", "figure_contact_sheets", "passed"]}, indent=2))
    raise SystemExit(0 if audit["passed"] else 1)


if __name__ == "__main__":
    main()
