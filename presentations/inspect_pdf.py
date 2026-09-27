"""Render the deck and create contact sheets; audit extracted text bounds."""
from pathlib import Path
import json
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
build = root / 'build'
preview = build / 'preview'
preview.mkdir(parents=True, exist_ok=True)
pdf = root / 'novogaia_spectra.pdf'
subprocess.run(['pdftoppm', '-r', '180', '-png', str(pdf), str(preview / 'slide')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
subprocess.run(['pdftotext', '-bbox-layout', str(pdf), str(build / 'text-bounds.html')], check=True)
xml = ET.parse(build / 'text-bounds.html')
ns = {'x': 'http://www.w3.org/1999/xhtml'}
issues = []
for page_num, page in enumerate(xml.findall('.//x:page', ns), 1):
    width, height = float(page.attrib['width']), float(page.attrib['height'])
    for word in page.findall('.//x:word', ns):
        x0, x1 = float(word.attrib['xMin']), float(word.attrib['xMax'])
        y0, y1 = float(word.attrib['yMin']), float(word.attrib['yMax'])
        if x0 < 12 or x1 > width-12 or y0 < 0 or y1 > height:
            issues.append({'page': page_num, 'text': word.text, 'bounds': [x0,y0,x1,y1]})
files = sorted(preview.glob('slide-*.png'))
for start in range(0, len(files), 9):
    canvas = Image.new('RGB', (1920, 1143), '#d9dcdf')
    draw = ImageDraw.Draw(canvas)
    for offset, file in enumerate(files[start:start+9]):
        im = Image.open(file).convert('RGB')
        im.thumbnail((636, 358))
        x, y = (offset % 3) * 640, (offset // 3) * 381
        canvas.paste(im, (x+2, y+21))
        draw.text((x+7, y+4), f'Slide {start+offset+1:02}', fill='black')
    canvas.save(preview / f'contact-{start//9+1}.jpg', quality=92)
(build / 'layout-audit.json').write_text(json.dumps({'pages':len(files),'near_edge_words':issues}, indent=2))
print(json.dumps({'pages':len(files),'near_edge_words':issues}, indent=2))
