"""Render a report HTML (with the generated charts) to PDF.

    python scripts/make_report_pdf.py                       # docs/REPORT_FOR_PPT.pdf
    python scripts/make_report_pdf.py --source docs/study_guide_source.html                                       --output docs/STUDY_GUIDE.pdf

Run ``scripts/make_charts.py`` first so ``artifacts/charts/*.png`` are current.
Uses a locally installed Chrome or Edge in headless mode; no other dependency.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPORT_HTML = Path("docs/report_source.html")
PRINT_HTML = Path("artifacts/report_print.html")
OUTPUT_PDF = Path("docs/REPORT_FOR_PPT.pdf")
CHARTS = Path("artifacts/charts")

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
]

PRINT_CSS = """
<style>
  @page { size: A4; margin: 16mm 14mm 18mm 14mm; }
  html, :root { color-scheme: light !important; }
  body { padding: 0 !important; background: #ffffff !important; font-size: 10.5pt; line-height: 1.45; }
  .shell { display: block !important; max-width: none !important; }
  .index { display: none !important; }
  main { padding-top: 0 !important; gap: 22pt !important; }
  section { page-break-before: always; break-before: page; gap: 12pt !important; }
  section#lead { page-break-before: auto; break-before: auto; }
  .masthead { padding-bottom: 12pt !important; }
  h1 { font-size: 24pt !important; }
  h2 { font-size: 17pt !important; }
  h3 { font-size: 12.5pt !important; margin-top: 6pt; }
  figure { box-shadow: none !important; page-break-inside: avoid; break-inside: avoid; padding: 6pt !important; }
  figure img { max-height: 118mm; width: auto; max-width: 100%; margin: 0 auto; }
  .fig-pair { grid-template-columns: 1fr !important; }
  .fig-pair figure img { max-height: 100mm; }
  .kpis { grid-template-columns: repeat(5, 1fr) !important; gap: 6pt !important; }
  .kpi { padding: 7pt 8pt !important; page-break-inside: avoid; }
  .kpi .n { font-size: 17pt !important; }
  .kpi .l { font-size: 8.5pt !important; }
  .kpi .s { font-size: 7.5pt !important; }
  table { font-size: 8.8pt !important; }
  th, td { padding: 4pt 6pt !important; }
  .table-wrap { page-break-inside: avoid; break-inside: avoid; overflow: visible !important; }
  tr { page-break-inside: avoid; }
  .message, .qa details, .dont div { page-break-inside: avoid; }
  .qa details { padding: 6pt 10pt !important; }
  .qa details summary { list-style: none; }
  .qa details summary::-webkit-details-marker { display: none; }
  a { color: inherit; text-decoration: none; }
  .foot { page-break-before: avoid; }
</style>
"""


def find_browser() -> str:
    for candidate in BROWSERS:
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise SystemExit("no Chrome or Edge found; install one or add its path to BROWSERS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPORT_HTML)
    parser.add_argument("--output", type=Path, default=OUTPUT_PDF)
    args = parser.parse_args()
    source, output = args.source, args.output
    print_html = PRINT_HTML.with_name(output.stem.lower() + "_print.html")

    if not source.exists():
        raise SystemExit(f"{source} is missing")
    if not any(CHARTS.glob("*.png")):
        raise SystemExit("no charts found; run scripts/make_charts.py first")

    html = source.read_text(encoding="utf-8")
    html = html.replace('src="charts/', f'src="file:///{CHARTS.resolve().as_posix()}/')
    html = html.replace("</style>", "</style>" + PRINT_CSS, 1)
    html = html.replace("<details>", "<details open>")
    print_html.parent.mkdir(parents=True, exist_ok=True)
    print_html.write_text(html, encoding="utf-8")

    browser = find_browser()
    command = [
        browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={output.resolve()}",
        "--virtual-time-budget=15000",
        print_html.resolve().as_uri(),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if not output.exists():
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit("PDF was not produced")
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
