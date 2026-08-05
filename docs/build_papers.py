#!/usr/bin/env python3
"""Build paper PDFs from the markdown sources.

The two published papers were produced with pandoc -> WeasyPrint, but neither the
stylesheet nor the command was committed, so they could not be regenerated. This
script plus ``paper.css`` is that missing recipe.

The markdown sources are left untouched and stay readable on GitHub: rather than
requiring YAML front matter, the title block is parsed out of the existing
convention

    # Title
    ### Subtitle
    **Author** · Date
    ---

and re-emitted as a styled title block, so the same file serves both renderers.

    python docs/build_papers.py                  # all papers
    python docs/build_papers.py paper4-*.md      # one
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DOCS = Path(__file__).parent
CSS = DOCS / "paper.css"


def weasyprint_env() -> dict:
    """WeasyPrint needs native pango/gobject, which macOS does not ship.

    Homebrew installs them outside the default loader path, so the CLI fails with
    ``cannot load library 'libgobject-2.0-0'`` unless DYLD_FALLBACK_LIBRARY_PATH
    points at the Homebrew prefix. Setting it here keeps the build working
    regardless of the caller's shell.
    """
    env = dict(os.environ)
    brew = shutil.which("brew")
    if brew:
        prefix = subprocess.run(
            [brew, "--prefix"], capture_output=True, text=True
        ).stdout.strip()
        if prefix:
            existing = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
            lib = f"{prefix}/lib"
            env["DYLD_FALLBACK_LIBRARY_PATH"] = (
                f"{lib}:{existing}" if existing else lib
            )
    return env

# Drafts only. The two published PDFs are historical artifacts and are not
# rebuilt -- their markdown is not the source the published PDF came from.
PAPERS = [
    "paper-measurement-validity.md",
    "paper3-retrieval-reconstruction-governance.md",
    "paper4-worked-examples.md",
]

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>$title$</title>
</head>
<body>
$body$
</body>
</html>
"""


def parse_title_block(md: str) -> tuple[str, str, str, str]:
    """Split the leading title block from the body.

    Returns (title, subtitle, author_line, remaining_markdown).
    """
    lines = md.split("\n")
    title = subtitle = author = ""
    idx = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if s.startswith("# ") and not title:
            title = s[2:].strip()
        elif s.startswith("### ") and not subtitle:
            subtitle = s[4:].strip()
        elif s.startswith("**") and not author:
            author = re.sub(r"\*\*", "", s).strip()
        elif s == "---" and title:
            idx = i + 1
            break
    return title, subtitle, author, "\n".join(lines[idx:])


def build(md_path: Path) -> Path:
    md = md_path.read_text()
    title, subtitle, author, body = parse_title_block(md)
    if not title:
        raise SystemExit(f"{md_path.name}: no '# Title' found")

    pdf_path = md_path.with_suffix(".pdf")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "body.md").write_text(body)
        (tmp / "tpl.html").write_text(TEMPLATE)

        # gfm for the table and fenced-code syntax the sources actually use.
        html = subprocess.run(
            [
                "pandoc", str(tmp / "body.md"),
                "--from", "gfm",
                "--to", "html5",
                "--template", str(tmp / "tpl.html"),
                "--metadata", f"title={title}",
                "--wrap", "none",
            ],
            capture_output=True, text=True, check=True,
        ).stdout

        # Title block injected after conversion so the markdown source does not
        # need front matter it would then have to display on GitHub.
        block = [f'<h1 class="title">{title}</h1>']
        if subtitle:
            block.append(f'<p class="subtitle">{subtitle}</p>')
        if author:
            block.append(f'<p class="author">{author}</p>')
        html = html.replace("<body>", "<body>\n" + "\n".join(block), 1)

        # Wrap the abstract so the WHOLE section is indented. CSS can only reach
        # one adjacent sibling, which left later abstract paragraphs full-width.
        html = re.sub(
            r'(<h2[^>]*>Abstract</h2>)(.*?)(?=<h[12])',
            lambda m: f'{m.group(1)}<div class="abstract">{m.group(2)}</div>',
            html,
            count=1,
            flags=re.S,
        )

        (tmp / "out.html").write_text(html)
        result = subprocess.run(
            ["weasyprint", "-s", str(CSS), str(tmp / "out.html"), str(pdf_path)],
            capture_output=True, text=True, env=weasyprint_env(),
        )
        if result.returncode != 0:
            tail = (result.stderr or "").strip().splitlines()[-3:]
            raise SystemExit(
                f"weasyprint failed for {md_path.name}:\n  "
                + "\n  ".join(tail)
                + "\n\nOn macOS this usually means the native deps are missing:"
                  "\n  brew install pango gdk-pixbuf libffi"
            )

    return pdf_path


def main() -> None:
    targets = sys.argv[1:] or PAPERS
    for name in targets:
        p = Path(name)
        if not p.is_absolute() and not p.exists():
            p = DOCS / Path(name).name
        pdf = build(p)
        size = pdf.stat().st_size // 1024
        pages = ""
        try:
            info = subprocess.run(
                ["pdfinfo", str(pdf)], capture_output=True, text=True
            ).stdout
            m = re.search(r"Pages:\s+(\d+)", info)
            if m:
                pages = f", {m.group(1)} pages"

        except FileNotFoundError:
            pass
        print(f"  {pdf.name}  ({size} KB{pages})")


if __name__ == "__main__":
    main()
