"""
보고 문서 PDF 변환기 — mermaid 도식을 PNG로 렌더링해 MD를 PDF로 만든다.

파이프라인 (외부 설치 없이 이 맥의 도구 재사용):
  ① mermaid-cli(npx) — MD 안의 ```mermaid 블록을 모두 PNG로 렌더 (시스템 Chrome 사용, 한글 OK)
  ② markdown-it-py — MD → HTML (표·한글)
  ③ Chrome headless — HTML → PDF (--print-to-pdf)

대상: 5_도식도.md, 6_보고서.md → 각각 PDF + assets/에 PNG.
실행:  ~/rnd-env/bin/python make_pdf.py
"""
import base64
import json
import os
import re
import subprocess
from pathlib import Path

from markdown_it import MarkdownIt

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DOCS = ["5_도식도", "6_보고서"]

PC = ASSETS / ".puppeteer.json"          # mmdc가 시스템 Chrome을 쓰도록 (크로미움 재다운로드 방지)
PC.write_text(json.dumps({"executablePath": CHROME, "args": ["--no-sandbox"]}), encoding="utf-8")

CSS = """
@page { margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: 'Apple SD Gothic Neo','Noto Sans KR',sans-serif; font-size: 11pt;
       line-height: 1.6; color: #1a1a1a; max-width: 100%; }
h1 { font-size: 19pt; border-bottom: 2px solid #333; padding-bottom: .3em; }
h2 { font-size: 15pt; margin-top: 1.4em; border-bottom: 1px solid #ccc; padding-bottom: .2em; }
h3 { font-size: 12.5pt; margin-top: 1em; }
table { border-collapse: collapse; width: 100%; margin: .8em 0; font-size: 10pt; }
th,td { border: 1px solid #bbb; padding: 5px 8px; text-align: left; }
th { background: #f0f2f5; }
img { max-width: 100%; height: auto; }
pre { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 5px; padding: 10px;
      font-size: 9pt; white-space: pre-wrap; }
code { font-family: 'SF Mono',Menlo,monospace; }
blockquote { color: #555; border-left: 3px solid #ccc; padding-left: 1em; margin-left: 0; }
"""
TEMPLATE = "<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"


def render(name):
    src = HERE / f"{name}.md"
    out_md = ASSETS / f"{name}.rendered.md"        # mmdc: mermaid → PNG + 이미지 참조 MD
    subprocess.run(
        ["npx", "-y", "@mermaid-js/mermaid-cli", "-i", str(src), "-o", str(out_md),
         "-p", str(PC), "-b", "white", "-e", "png", "-s", "2"],   # -e png: PNG 출력(SVG 아님) / -s 2: 2배 해상도
        env={**os.environ, "PUPPETEER_SKIP_DOWNLOAD": "true"}, check=True)

    md_text = out_md.read_text(encoding="utf-8")
    # 이미지 참조를 base64 data URI로 치환 (Chrome의 로컬 파일 접근 제약 회피 — 자립 HTML)
    def embed(m):
        p = (out_md.parent / m.group(1)).resolve()
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'<p style="text-align:center"><img src="data:image/png;base64,{b64}"></p>'
    md_text = re.sub(r"!\[[^\]]*\]\(([^)]+)\)", embed, md_text)

    body = MarkdownIt("commonmark", {"html": True}).enable("table").render(md_text)
    html_file = ASSETS / f"{name}.html"
    html_file.write_text(TEMPLATE.format(css=CSS, body=body), encoding="utf-8")

    pdf = HERE / f"{name}.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", str(html_file)],
                   check=True, capture_output=True)
    n_png = len(list(ASSETS.glob(f"{name}.rendered-*.png")))
    print(f"{name}.pdf 생성 (도식 {n_png}개 PNG 포함) → {pdf}")


if __name__ == "__main__":
    for d in DOCS:
        render(d)
    print("완료. assets/에 PNG·중간 HTML 보관.")
