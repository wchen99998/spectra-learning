"""Export selected slide diagrams as editable, Nature-style vector PDFs.

Run from the repository root with ``.venv/bin/python presentations/export_figures.py``.
The source slides remain untouched. Every output and diagnostic stays beneath
presentations/. The shared preamble is separately editable and is not rewritten.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess

import pymupdf as fitz
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
BUILD = ROOT / "build" / "figures"
SOURCE = ROOT / "novogaia_spectra.tex"
SCALE = 1.14375
WIDTH_MM = 183.0
HEIGHT_MM = 60.046875
FIGURE_SLIDES = {
    2: "inverse-problem",
    3: "two-routes",
    4: "graph-diffusion",
    5: "autoregressive-smiles",
    6: "decoder-comparison",
    7: "raw-spectra-scale",
    8: "acquisition-metadata",
    9: "two-stage-pretraining",
    10: "multiscale-mae",
    11: "frozen-teacher-jepa",
    12: "representation-coordinates",
    13: "predictable-signal",
    14: "encoder-gradient",
    16: "representation-diagnostics",
    17: "peak-pair-representations",
    18: "triangle-interactions",
    19: "pair-distogram",
    20: "retrieval-encoders",
    21: "local-compatibility",
    22: "retrieval-training",
    23: "retrieval-result",
    24: "research-directions",
    26: "gradient-flow-metric",
    27: "linear-factorization",
    28: "auxiliary-readout",
    30: "retrieval-details",
}


def extract_diagrams(source: str) -> list[dict[str, str | int]]:
    """Retain only scientific TikZ content, normalizing label and stroke styles."""
    frames = re.findall(
        r"\\begin\{frame\}(?:\[[^\]]*\])?\s*(.*?)\\end\{frame\}",
        source,
        flags=re.DOTALL,
    )
    if len(frames) < max(FIGURE_SLIDES):
        raise ValueError(f"Expected at least {max(FIGURE_SLIDES)} frames, found {len(frames)}")
    diagrams = []
    for index, (slide, slug) in enumerate(FIGURE_SLIDES.items(), start=1):
        frame = frames[slide - 1]
        body = re.sub(r"^\\(?:header|source|takeaway)\{.*$", "", frame, flags=re.MULTILINE)
        body = re.sub(r"^\\begin\{tikzpicture\}(?:\[[^\n]*\])?\s*$", "", body, flags=re.MULTILINE)
        body = re.sub(r"^\\end\{tikzpicture\}\s*$", "", body, flags=re.MULTILINE)
        if any(token in body for token in (r"\header", r"\source", r"\takeaway", r"\includegraphics")):
            raise ValueError(f"Slide {slide} still contains a page-level macro or external image")
        body = re.sub(r"\\fontsize\{[^{}]*\}\{[^{}]*\}\\selectfont", r"\\figurefont", body)
        body = body.replace(r"\figurefont\figurefont", r"\figurefont")
        body = re.sub(r"line width\s*=\s*[\d.]+\s*pt", "line width=.55pt", body)
        body = body.strip()
        # Figure 09 is reserved for the independently generated dynamics plot.
        number = index if index < 9 else index + 1
        diagrams.append({
            "slide": slide,
            "figure": number,
            "stem": f"slide-{slide:02d}-{slug}",
            "body": body,
        })
    return diagrams


def write_source(diagram: dict[str, str | int]) -> Path:
    stem = str(diagram["stem"])
    path = FIGURES / f"{stem}.tex"
    content = (
        "% Editable standalone vector figure extracted from novogaia_spectra.tex.\n"
        f"% Source slide {diagram['slide']}; figure {diagram['figure']:02d}; native embedded fonts.\n"
        "% Compile with LuaLaTeX from this directory, or run ../export_figures.py.\n"
        r"\documentclass[tikz,border=0pt]{standalone}" "\n"
        r"\input{nature-figure-preamble.tex}" "\n"
        r"\begin{document}" "\n"
        r"\figurefont" "\n"
        f"\\begin{{tikzpicture}}[x=1cm,y=1cm,scale={SCALE},transform shape]\n"
        f"{diagram['body']}\n"
        "% Fixed canvas: 183 x 60.046875 mm. No clipping: overflow is audited.\n"
        r"\pgfresetboundingbox" "\n"
        r"\path[use as bounding box] (0,1.6) rectangle (16,6.85);" "\n"
        r"\end{tikzpicture}" "\n"
        r"\end{document}" "\n"
    )
    path.write_text(content)
    return path


def compile_figure(path: Path) -> Path:
    command = [
        "lualatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
        f"-output-directory={BUILD}", path.name,
    ]
    result = subprocess.run(command, cwd=FIGURES, text=True, capture_output=True)
    log = BUILD / f"{path.stem}.compile.txt"
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"LuaLaTeX failed for {path.name}; inspect {log}")
    pdf = FIGURES / f"{path.stem}.pdf"
    shutil.copy2(BUILD / pdf.name, pdf)
    return pdf


def audit_pdf(pdf: Path) -> dict:
    """Check actual PDF text sizes, embedded fonts, vectors, and page boundaries."""
    with fitz.open(pdf) as document:
        if len(document) != 1:
            raise ValueError(f"Expected one page in {pdf}")
        page = document[0]
        bounds = page.rect
        issues = []
        width_mm, height_mm = bounds.width * 25.4 / 72, bounds.height * 25.4 / 72
        if abs(width_mm - WIDTH_MM) > .03 or abs(height_mm - HEIGHT_MM) > .03:
            issues.append({"kind": "page_size", "width_mm": width_mm, "height_mm": height_mm})
        spans = [
            span
            for block in page.get_text("dict")["blocks"] if "lines" in block
            for line in block["lines"] for span in line["spans"] if span["text"].strip()
        ]
        sizes = sorted({round(span["size"], 4) for span in spans})
        for span in spans:
            if not 4.95 <= span["size"] <= 7.05:
                issues.append({"kind": "font_size", "text": span["text"], "size_pt": span["size"]})
            if not (bounds + (-.5, -.5, .5, .5)).contains(fitz.Rect(span["bbox"])):
                issues.append({"kind": "text_bounds", "text": span["text"], "bbox": span["bbox"]})
        fonts = []
        for xref, extension, font_type, base_name, resource, encoding, *rest in page.get_fonts(full=True):
            _, _, _, font_data = document.extract_font(xref)
            fonts.append({
                "name": base_name, "type": font_type, "extension": extension,
                "embedded": bool(font_data), "bytes": len(font_data),
            })
            if not font_data:
                issues.append({"kind": "unembedded_font", "name": base_name})
        drawings = page.get_drawings()
        widths = []
        for drawing in drawings:
            if drawing["type"] in {"s", "fs"}:
                width = drawing["width"]
                widths.append(width)
                if not .24 <= width <= 1.01:
                    issues.append({"kind": "stroke_width", "width_pt": width})
            if not (bounds + (-.6, -.6, .6, .6)).contains(drawing["rect"]):
                issues.append({"kind": "vector_bounds", "bbox": list(drawing["rect"])})
        images = page.get_images(full=True)
        if images:
            issues.append({"kind": "raster_images", "count": len(images)})
        if not spans:
            issues.append({"kind": "missing_live_text"})
        preview = BUILD / f"{pdf.stem}.png"
        page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False).save(preview)
        return {
            "file": str(pdf.relative_to(ROOT)),
            "size_mm": [round(width_mm, 4), round(height_mm, 4)],
            "font_sizes_pt": sizes,
            "font_size_range_pt": [min(sizes), max(sizes)] if sizes else [],
            "embedded_fonts": fonts,
            "text_spans": len(spans),
            "vector_paths": len(drawings),
            "stroke_widths_pt": sorted({round(width, 4) for width in widths}),
            "raster_images": len(images),
            "issues": issues,
        }


def make_contact_sheets(diagrams: list[dict[str, str | int]]) -> list[str]:
    """Create readable overview sheets; high-resolution per-figure PNGs also remain."""
    output = []
    tile_w, tile_h = 1300, 430
    for start in range(0, len(diagrams), 6):
        batch = diagrams[start:start + 6]
        rows = (len(batch) + 1) // 2
        canvas = Image.new("RGB", (tile_w * 2, tile_h * rows), "#e8e8e8")
        draw = ImageDraw.Draw(canvas)
        for offset, diagram in enumerate(batch):
            image_path = BUILD / f"{diagram['stem']}.png"
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((tile_w - 16, tile_h - 34))
                x, y = (offset % 2) * tile_w, (offset // 2) * tile_h
                canvas.paste(image, (x + 8, y + 26))
                draw.text((x + 10, y + 7), f"Figure {diagram['figure']:02d} / Slide {diagram['slide']:02d} / {diagram['stem']}", fill="black")
        path = BUILD / f"contact-{start // 6 + 1:02d}.png"
        canvas.save(path)
        output.append(str(path.relative_to(ROOT)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-only", action="store_true", help="Extract editable sources without compiling")
    parser.add_argument("--workers", type=int, default=4, help="Parallel LuaLaTeX processes (default: 4)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    FIGURES.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    rebuild = BUILD / "rebuild.sh"
    rebuild.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'cd -- "$(dirname -- "$0")/../../.."\n'
        'exec .venv/bin/python presentations/export_figures.py "$@"\n'
    )
    rebuild.chmod(0o755)
    source = SOURCE.read_text()
    diagrams = extract_diagrams(source)
    sources = [write_source(diagram) for diagram in diagrams]
    manifest = {
        "style": "Nature-style standalone vector diagrams; not a journal-acceptance claim",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(source.encode()).hexdigest(),
        "preamble": "figures/nature-figure-preamble.tex",
        "preamble_sha256": sha256((FIGURES / "nature-figure-preamble.tex").read_bytes()).hexdigest(),
        "width_mm": WIDTH_MM, "height_mm": HEIGHT_MM,
        "coordinate_scale": SCALE,
        "body_font_before_scale_tex_pt": 5.9,
        "math_font_before_scale_tex_pt": 6,
        "script_font_before_scale_tex_pt": 5,
        "reserved_figure_09": "Independent analytic dynamics plot, maintained separately",
        "figures": [{key: value for key, value in diagram.items() if key != "body"} for diagram in diagrams],
    }
    (FIGURES / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.sources_only:
        print(f"Extracted {len(sources)} editable figure sources into {FIGURES}")
        return
    for font in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf"):
        if not (ROOT / "assets" / font).is_file():
            raise FileNotFoundError(f"Missing publication font: {ROOT / 'assets' / font}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pdfs = list(pool.map(compile_figure, sources))
    audits = [audit_pdf(pdf) for pdf in pdfs]
    contacts = make_contact_sheets(diagrams)
    report = {
        "figures": audits,
        "contact_sheets": contacts,
        "total_issues": sum(len(audit["issues"]) for audit in audits),
        "checks": ["183 mm width", "5–7 pt live text", "embedded fonts", "0.25–1 pt strokes", "page boundaries", "no raster images"],
    }
    (BUILD / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"exported": len(pdfs), "total_issues": report["total_issues"], "contacts": contacts}, indent=2))
    if report["total_issues"]:
        raise SystemExit(f"Figure audit found {report['total_issues']} issues; inspect {BUILD / 'audit.json'}")


if __name__ == "__main__":
    main()
