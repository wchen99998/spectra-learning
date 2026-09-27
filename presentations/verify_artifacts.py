"""Audit the delivered deck and standalone figures; assemble the figure PDF."""
from hashlib import sha256
import json
from pathlib import Path
import re

import pymupdf as fitz

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / 'figures'


def audit_figure(path: Path) -> dict:
    with fitz.open(path) as doc:
        assert len(doc) == 1, path
        page = doc[0]
        width, height = [v * 25.4 / 72 for v in (page.rect.width,page.rect.height)]
        spans = [span for block in page.get_text('dict')['blocks']
                 if 'lines' in block for line in block['lines'] for span in line['spans']
                 if span['text'].strip()]
        sizes = [span['size'] for span in spans]
        assert min(sizes) >= 5 and max(sizes) <= 7.01, (path, min(sizes), max(sizes))
        assert min(abs(width - 89), abs(width - 183)) < .01 and height < 170, path
        assert not page.get_images(full=True), path
        for font in page.get_fonts(full=True):
            assert font[2] != 'Type3' and doc.extract_font(font[0])[3], (path, font)
        widths = sorted({round(path['width'], 4) for path in page.get_drawings()
                         if path['color'] is not None and path['width'] > 0})
        assert min(widths) >= .25 and max(widths) <= 1.001, (path, widths)
        return {'file':path.name, 'dimensions_mm':[round(width,4),round(height,4)],
                'font_size_range_pt':[round(min(sizes),4),round(max(sizes),4)],
                'stroke_widths_pt':widths, 'raster_images':0,
                'fonts_embedded':True, 'text_outlined':False,
                'sha256':sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    deck = ROOT / 'novogaia_spectra.pdf'
    with fitz.open(deck) as doc:
        assert len(doc) == 33
        for number,page in enumerate(doc,1):
            assert 'Private & Confidential' not in page.get_text(), number
            assert not page.get_images(full=True), number
        assert all(font[2] != 'Type3' and doc.extract_font(font[0])[3]
                   for page in doc for font in page.get_fonts(full=True))
    log=(ROOT/'build/novogaia_spectra.log').read_text()
    assert not re.search(r'Overfull|Missing character|! LaTeX Error|Warning:',log)
    assert not json.loads((ROOT/'build/layout-audit.json').read_text())['near_edge_words']
    manifest=json.loads((FIGURES/'manifest.json').read_text())
    assert manifest['source_sha256'] == sha256((ROOT/'novogaia_spectra.tex').read_bytes()).hexdigest()
    entries=[(item['figure'],FIGURES/(item['stem']+'.pdf')) for item in manifest['figures']]
    entries.append((9,FIGURES/'fig-09-dynamics.pdf'))
    entries.sort()
    audit=[audit_figure(path) for _,path in entries]
    assert len(audit)==len(manifest["figures"])+1
    combined=fitz.open()
    for _,path in entries:
        with fitz.open(path) as source:combined.insert_pdf(source)
    combined.set_metadata({'title':'Novogaia — standalone scientific figures','author':'Novogaia',
                           'subject':f'{len(audit)} vector figures prepared to Nature artwork specifications'})
    combined.save(FIGURES/'novogaia_figures.pdf',garbage=4,deflate=True)
    combined.close()
    report={'deck':{'pages':33,'main_slides':24,'appendix_slides':9,'confidentiality_footers':0,
                    'raster_images':0,'fonts_embedded':True,'page_boundary_issues':0,'latex_warnings':0},
            'figures':audit,'standalone_figure_count':len(audit),
            'source_sha256':manifest['source_sha256'],
            'theme_sha256':sha256((ROOT/'novogaia-theme.sty').read_bytes()).hexdigest(),
            'deck_sha256':sha256(deck.read_bytes()).hexdigest(),
            'standard':'Nature artwork technical checks; final editorial layout is determined by the journal'}
    (FIGURES/'production_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'deck_pages':33,'figures':len(audit),'technical_issues':0}))


if __name__=='__main__':main()
