"""Render and audit the retrieval failure-analysis deck."""

import argparse
import json
import math
import re
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent


def inspect(pdf_path: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    preview = output / "preview"
    preview.mkdir(exist_ok=True)
    pages: list[dict[str, object]] = []
    edge_words: list[dict[str, object]] = []
    outside_words: list[dict[str, object]] = []
    fonts: dict[int, dict[str, object]] = {}
    all_text: list[str] = []
    images: list[Path] = []

    with pymupdf.open(pdf_path) as document:
        for index, page in enumerate(document):
            number = index + 1
            all_text.append(f"--- Slide {number} ---\n{page.get_text()}")
            image_path = preview / f"slide-{number:02d}.png"
            page.get_pixmap(dpi=180, alpha=False).save(image_path)
            images.append(image_path)
            for x0, y0, x1, y1, word, *_ in page.get_text("words"):
                issue = {"page": number, "text": word, "bounds": [x0, y0, x1, y1]}
                if x0 < 12 or x1 > page.rect.width - 12 or y0 < 0 or y1 > page.rect.height:
                    edge_words.append(issue)
                if x0 < -0.5 or x1 > page.rect.width + 0.5 or y0 < -0.5 or y1 > page.rect.height + 0.5:
                    outside_words.append(issue)
            for font in page.get_fonts(full=True):
                xref, extension, kind, name, *_ = font
                if xref not in fonts:
                    _, _, _, data = document.extract_font(xref)
                    fonts[xref] = {
                        "xref": xref,
                        "name": name,
                        "type": kind,
                        "extension": extension,
                        "embedded": bool(data),
                    }
            spans = [
                span
                for block in page.get_text("dict")["blocks"]
                if block["type"] == 0
                for line in block["lines"]
                for span in line["spans"]
                if span["text"].strip()
            ]
            pages.append({
                "page": number,
                "size_points": [page.rect.width, page.rect.height],
                "raster_image_count": len(page.get_images(full=True)),
                "vector_path_count": len(page.get_drawings()),
                "minimum_font_points": min((float(span["size"]) for span in spans), default=None),
            })

    contacts: list[str] = []
    columns, per_sheet = 3, 12
    cell_width, cell_height = 640, 384
    for start in range(0, len(images), per_sheet):
        sheet_images = images[start:start + per_sheet]
        rows = math.ceil(len(sheet_images) / columns)
        canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "#d9dcdf")
        draw = ImageDraw.Draw(canvas)
        for offset, image_path in enumerate(sheet_images):
            with Image.open(image_path) as image:
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((cell_width - 8, cell_height - 28))
            x, y = (offset % columns) * cell_width, (offset // columns) * cell_height
            canvas.paste(thumbnail, (x + 4, y + 24))
            draw.text((x + 8, y + 6), f"Slide {start + offset + 1:02d}", fill="#0c0c0c")
        contact_path = preview / f"contact-{start // per_sheet + 1}.jpg"
        canvas.save(contact_path, quality=94)
        contacts.append(str(contact_path))

    log_path = output / "retrieval_failures.log"
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    warnings = [
        line for line in log_text.splitlines()
        if re.search(r"Overfull|Underfull|Missing character|LaTeX Warning|Package .* Warning", line)
    ]
    hard_warnings = [line for line in warnings if "Overfull" in line or "Missing character" in line]
    missing_fonts = [font for font in fonts.values() if not font["embedded"]]
    raster_count = sum(int(page["raster_image_count"]) for page in pages)
    audit: dict[str, object] = {
        "pdf": str(pdf_path),
        "page_count": len(pages),
        "render_dpi": 180,
        "pages": pages,
        "fonts": list(fonts.values()),
        "all_fonts_embedded": not missing_fonts,
        "raster_image_count": raster_count,
        "vector_only_artwork": raster_count == 0,
        "near_edge_words": edge_words,
        "outside_page_words": outside_words,
        "latex_log_present": log_path.exists(),
        "latex_warnings": warnings,
        "contact_sheets": contacts,
        "passed": not (outside_words or missing_fonts or hard_warnings or raster_count),
    }
    (output / "extracted_text.txt").write_text("\n\n".join(all_text))
    (output / "layout-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "retrieval_failures.pdf")
    parser.add_argument("--out", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    audit = inspect(args.pdf.resolve(), args.out.resolve())
    print(json.dumps(audit, indent=2))
    raise SystemExit(0 if audit["passed"] else 1)


if __name__ == "__main__":
    main()
