"""Render the README graphics (docs/src/*.dc.html) to PNG with a local headless Edge/Chrome.

Usage: python tools/render_docs.py

The .dc.html files are Claude Design canvas artboards; this strips the canvas runtime tags, fills the
accent colour, and screenshots each at its native size into docs/.
"""
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]
ACCENT = "#F2B24C"


def to_plain_html(src: str) -> tuple[str, int, int]:
    w, h = map(int, re.search(r'"\$preview":\{"width":(\d+),"height":(\d+)\}', src).groups())
    helmet = re.search(r"<helmet>(.*?)</helmet>", src, re.S).group(1)
    body = re.search(r"</helmet>(.*?)</x-dc>", src, re.S).group(1).replace("{{accent}}", ACCENT)
    title = re.search(r"<title>(.*?)</title>", src).group(1)
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>{helmet}'
            f"</head><body>{body}</body></html>")
    return html, w, h


def main():
    browser = next(b for b in BROWSERS if os.path.exists(b))
    for f in sorted((DOCS / "src").glob("*.dc.html")):
        html, w, h = to_plain_html(f.read_text(encoding="utf-8"))
        name = f.name.replace(".dc.html", "")
        page = DOCS / "src" / f"{name}.html"
        page.write_text(html, encoding="utf-8")
        out = DOCS / f"{name}.png"
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                        f"--window-size={w},{h}", "--virtual-time-budget=8000", f"--screenshot={out}", page.as_uri()],
                       check=True, capture_output=True)
        print(f"{out.relative_to(ROOT)}  {w}x{h}")


if __name__ == "__main__":
    main()
