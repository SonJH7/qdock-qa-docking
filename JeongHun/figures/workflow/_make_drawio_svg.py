"""
Post-process matplotlib-generated SVGs into draw.io-friendly variants.

draw.io / app.diagrams.net often fails to open raw matplotlib SVGs because:
  (1) DOCTYPE declaration is non-standard for inline SVG,
  (2) RDF/Dublin-Core metadata block confuses the parser,
  (3) viewBox uses 'pt' units instead of unitless pixels,
  (4) large file size from base64-embedded PNGs.

This script produces *_drawio.svg next to each source SVG with these fixes
applied non-destructively (the originals remain untouched).
"""
import os
import re
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCES = [
    os.path.join(THIS_DIR, "qdock_workflow_overview.svg"),
    os.path.join(THIS_DIR, "fam_pipeline.svg"),
]


def clean(src_path: str, dst_path: str) -> None:
    with open(src_path, "r", encoding="utf-8") as f:
        s = f.read()

    # 1) Strip XML processing instruction + DOCTYPE.
    s = re.sub(r"<\?xml[^>]*\?>\s*", "", s)
    s = re.sub(r"<!DOCTYPE[^>]*>\s*", "", s)

    # 2) Strip metadata block (RDF/Dublin Core) — draw.io rejects it.
    s = re.sub(r"<metadata>.*?</metadata>\s*", "", s, flags=re.DOTALL)

    # 3) Convert pt-based width/height attributes to unitless px.
    #    matplotlib emits e.g. width="1087.2pt"; draw.io is happier with no unit.
    s = re.sub(r'(width|height)="([\d.]+)pt"', r'\1="\2"', s)

    # 4) Drop standalone="no" attribute (uncommon but draw.io occasionally rejects).
    s = re.sub(r'standalone="no"\s*', "", s)

    # 5) Re-add a minimal XML prologue.
    out = '<?xml version="1.0" encoding="UTF-8"?>\n' + s.lstrip()

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(out)


def main():
    for src in SOURCES:
        if not os.path.exists(src):
            print(f"[skip] missing: {src}", file=sys.stderr)
            continue
        base, _ = os.path.splitext(src)
        dst = f"{base}_drawio.svg"
        clean(src, dst)
        size_kb = os.path.getsize(dst) / 1024
        print(f"[OK] {dst}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
