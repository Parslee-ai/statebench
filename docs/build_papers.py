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

# Keywords are editorial, so they live here rather than being scraped out of
# the prose. Every paper also gets BASE_KEYWORDS.
BASE_KEYWORDS = ["LLM agents", "agent memory", "StateBench", "Parslee"]
KEYWORDS = {
    "paper-measurement-validity.md": [
        "benchmark validity", "measurement error", "phrase-list scoring",
        "LLM-as-judge", "construct validity",
    ],
    "paper3-retrieval-reconstruction-governance.md": [
        "retrieval-augmented generation", "governance", "access control",
        "state resolution", "counterfactual evaluation",
    ],
    "paper4-worked-examples.md": [
        "skill distillation", "worked examples", "in-context learning",
        "procedure synthesis", "evaluation variance",
    ],
    "memgine-deterministic-memory-engine.md": [
        "deterministic memory", "context assembly", "supersession tracking",
        "memory engine", "state correctness",
    ],
    "state-based-context-architecture.md": [
        "context architecture", "state-based memory", "enterprise AI",
        "supersession", "conversation transcripts",
    ],
}

# Historical PDFs. Their rendering is the published record, so these are never
# re-rendered -- only their document metadata is corrected in place. Before
# this, `memgine` carried its filename slug as the PDF Title and the 2025
# paper carried no Title at all and an author of "Un-named", which is what
# search engines and reference managers read.
PUBLISHED = {
    "memgine-deterministic-memory-engine.pdf": {
        "title": "Memgine: A Deterministic Memory Engine for Stateful AI Agents",
        "author": "Matt Liotta",
        "date": "2026-02-01",
        "keywords_from": "memgine-deterministic-memory-engine.md",
    },
    "state-based-context-architecture.pdf": {
        "title": ("Beyond Conversation: A State-Based Context Architecture "
                  "for Enterprise AI Agents"),
        "author": "Matt Liotta",
        "date": "2025-12-01",
        "keywords_from": "state-based-context-architecture.md",
        # This paper has no markdown source anywhere -- it was produced in
        # LibreOffice and only the PDF survives -- so the abstract cannot be
        # read back out of a source file. Transcribed from page 1 of the PDF.
        "subject": (
            "Current approaches to LLM agent memory treat conversation history "
            "as the source of truth, replaying message transcripts to establish "
            "context. This paradigm fails at scale: transcripts grow unbounded, "
            "old context pollutes current reasoning, and mining conversations "
            "for facts is computationally expensive. We propose a fundamental "
            "reframing: context is structured state assembled fresh on every "
            "turn, not a transcript replay."
        ),
    },
}

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}

# WeasyPrint maps these to the PDF document information dictionary: title,
# author, subject, keywords, creator, and the creation/modification dates.
# Verified against `pdfinfo` output -- without them a reader, a reference
# manager, or Google Scholar has only the filename to go on.
TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>$title$</title>
$if(author-meta)$<meta name="author" content="$author-meta$">
$endif$$if(subject-meta)$<meta name="description" content="$subject-meta$">
$endif$$if(keywords-meta)$<meta name="keywords" content="$keywords-meta$">
$endif$$if(created-meta)$<meta name="dcterms.created" content="$created-meta$">
<meta name="dcterms.modified" content="$created-meta$">
$endif$<meta name="generator" content="statebench docs/build_papers.py">
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


def split_author_line(author_line: str) -> tuple[str, str]:
    """Split "Matt Liotta · August 2026" into (name, ISO date).

    The date half is optional; papers that predate the convention carry only a
    name. Returns ("", "") for anything unparseable rather than guessing, so a
    malformed byline yields no metadata instead of wrong metadata.
    """
    if not author_line:
        return "", ""
    name, _, date_part = author_line.partition("·")
    name = name.strip()
    m = re.search(r"([A-Z][a-z]+)\s+(\d{4})", date_part)
    if not m or m.group(1) not in MONTHS:
        return name, ""
    return name, f"{m.group(2)}-{MONTHS[m.group(1)]:02d}-01"


def extract_subject(body: str) -> str:
    """First sentences of the abstract, for the PDF Subject field.

    This is what a reference manager shows under the title and what Scholar
    reads as the description, so it has to stand alone as one line.
    """
    m = re.search(r"##\s+Abstract\s*\n+(.+?)(?=\n##\s|\Z)", body, re.S)
    if not m:
        return ""
    text = re.sub(r"[*_`]", "", m.group(1)).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= 500:
        return text
    cut = text[:500].rsplit(". ", 1)[0]
    return (cut + ".") if cut else text[:500]


def build(md_path: Path) -> Path:
    md = md_path.read_text()
    title, subtitle, author, body = parse_title_block(md)
    if not title:
        raise SystemExit(f"{md_path.name}: no '# Title' found")

    author_name, created = split_author_line(author)
    subject = extract_subject(body)
    keywords = ", ".join(KEYWORDS.get(md_path.name, []) + BASE_KEYWORDS)

    pdf_path = md_path.with_suffix(".pdf")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "body.md").write_text(body)
        (tmp / "tpl.html").write_text(TEMPLATE)

        # gfm for the table and fenced-code syntax the sources actually use.
        # Metadata goes through pandoc rather than being spliced in afterwards
        # so the values are HTML-escaped for the attribute context.
        html = subprocess.run(
            [
                "pandoc", str(tmp / "body.md"),
                "--from", "gfm",
                "--to", "html5",
                "--template", str(tmp / "tpl.html"),
                "--metadata", f"title={title}",
                "--metadata", f"author-meta={author_name}",
                "--metadata", f"subject-meta={subject}",
                "--metadata", f"keywords-meta={keywords}",
                "--metadata", f"created-meta={created}",
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

        # Wrap sections whose styling must not leak. CSS sibling selectors are
        # the wrong tool here: an adjacent sibling reaches only the first
        # paragraph (which left later abstract paragraphs full-width), and a
        # general sibling reaches everything after it (which gave the appendix a
        # hanging indent meant for references).
        for heading, cls in (("Abstract", "abstract"), ("References", "references")):
            html = re.sub(
                rf'(<h2[^>]*>{heading}</h2>)(.*?)(?=<h[12]|</body>)',
                lambda m, c=cls: f'{m.group(1)}<div class="{c}">{m.group(2)}</div>',
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


def stamp_published(pdf_name: str, spec: dict) -> Path:
    """Correct the metadata on a published PDF without re-rendering it.

    The published rendering is the citable record, and its markdown is not the
    source it was produced from, so regenerating would silently replace a
    published artifact. Rewriting only the document information dictionary
    leaves every page byte-identical.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise SystemExit(
            "stamping published PDFs needs PyMuPDF:  pip install pymupdf\n"
            "(or skip it: python docs/build_papers.py --drafts-only)"
        )

    pdf_path = DOCS / pdf_name
    if not pdf_path.is_file():
        raise SystemExit(f"{pdf_name}: not found")

    md_name = spec["keywords_from"]
    keywords = ", ".join(KEYWORDS.get(md_name, []) + BASE_KEYWORDS)

    # Prefer the abstract in the markdown; fall back to a transcribed one for
    # papers whose source no longer exists.
    subject = spec.get("subject", "")
    md_path = DOCS / md_name
    if md_path.is_file():
        subject = extract_subject(md_path.read_text()) or subject
    if not subject:
        raise SystemExit(f"{pdf_name}: no subject available")

    doc = fitz.open(pdf_path)
    # PDF date strings are D:YYYYMMDDHHmmSS with an offset.
    stamp = f"D:{spec['date'].replace('-', '')}000000Z"
    wanted = {
        "title": spec["title"],
        "author": spec["author"],
        "subject": subject,
        "keywords": keywords,
        "creator": "statebench docs/build_papers.py",
        "producer": doc.metadata.get("producer", ""),
        "creationDate": stamp,
        "modDate": stamp,
    }

    # An incremental save appends a revision, so re-stamping an already-correct
    # PDF on every build would grow the file without changing anything.
    if all(doc.metadata.get(k) == v for k, v in wanted.items()):
        doc.close()
        return pdf_path

    doc.set_metadata(wanted)
    doc.saveIncr()
    doc.close()
    return pdf_path


def describe(pdf: Path) -> str:
    size = pdf.stat().st_size // 1024
    try:
        info = subprocess.run(
            ["pdfinfo", str(pdf)], capture_output=True, text=True
        ).stdout
        m = re.search(r"Pages:\s+(\d+)", info)
        if m:
            return f"{size} KB, {m.group(1)} pages"
    except FileNotFoundError:
        pass
    return f"{size} KB"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    targets = args or PAPERS
    for name in targets:
        p = Path(name)
        if not p.is_absolute() and not p.exists():
            p = DOCS / Path(name).name
        pdf = build(p)
        print(f"  {pdf.name}  ({describe(pdf)})")

    # Published PDFs get their metadata corrected but are never re-rendered.
    if not args and "--drafts-only" not in flags:
        for pdf_name, spec in PUBLISHED.items():
            pdf = stamp_published(pdf_name, spec)
            print(f"  {pdf.name}  (metadata stamped; {describe(pdf)})")


if __name__ == "__main__":
    main()
