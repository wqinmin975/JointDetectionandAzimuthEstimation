#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build ``coverletter.pdf`` from ``cover_letter.md`` (Markdown -> LaTeX -> PDF).

Toolchain (there is no pandoc on this machine):

    cover_letter.md
      -> _make_coverletter.py          (hand-written Markdown parser, this file)
      -> _coverletter_body.tex         (generated \\input fragment)
      -> _coverletter.tex              (generated wrapper: preamble + \\input)
      -> MiKTeX 25.12 pdflatex, 2 passes (-jobname=coverletter)
      -> coverletter.pdf
      -> pdftotext verification (Markdown-residue scan + content cross-check)

Usage::

    python _make_coverletter.py

Only *reads* ``cover_letter.md``; it never modifies the Markdown source.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD_PATH = HERE / "cover_letter.md"
BODY_TEX = HERE / "_coverletter_body.tex"
WRAP_TEX = HERE / "_coverletter.tex"
PDF_PATH = HERE / "coverletter.pdf"
BUILD_LOG = HERE / "_coverletter_build.log"
TXT_PATH = HERE / "_coverletter_text.txt"

ENGINE = "pdflatex"          # MiKTeX 25.12
PASSES = 2
JOB = "coverletter"

MIKTEX_BIN = Path(
    os.environ.get(
        "MIKTEX_BIN",
        r"C:\Users\Administrator\AppData\Local\Programs\MiKTeX\miktex\bin\x64",
    )
)

# --------------------------------------------------------------------------
# LaTeX escaping
# --------------------------------------------------------------------------

# Characters that must be escaped / translated in running text.
_PLAIN_ESC = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "_": r"\_",
    "%": r"\%",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "—": r"\textemdash{}",       # em dash
    "–": r"\textendash{}",       # en dash
    "‘": "`",
    "’": "'",
    "“": "``",
    "”": "''",
    "…": r"\ldots{}",
}

# Characters that must be escaped inside \texttt{} (monospace).
_TT_ESC = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "_": r"\_",
    "%": r"\%",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def esc_plain(text: str) -> str:
    """Escape LaTeX specials in running text (no markup handling)."""
    return "".join(_PLAIN_ESC.get(ch, ch) for ch in text)


def esc_tt(text: str) -> str:
    """Escape LaTeX specials inside a monospace span."""
    out = "".join(_TT_ESC.get(ch, ch) for ch in text)
    # Defuse TeX ligatures so "--mode" survives as two literal hyphens.
    out = out.replace("---", "-{}-{}-").replace("--", "-{}-")
    return out


def esc_escaped(ch: str) -> str:
    """A character that was backslash-escaped in the Markdown source."""
    return _PLAIN_ESC.get(ch, ch)


# --------------------------------------------------------------------------
# Inline Markdown -> LaTeX
# --------------------------------------------------------------------------


def inline_to_latex(src: str) -> str:
    """Convert inline Markdown to LaTeX.

    Handles, in one left-to-right pass:
      ``code``  -> \\texttt{...}       \\*  -> literal *
      $math$    -> $math$ (verbatim)   **b** -> \\textbf{...}
      *i*       -> \\textit{...}
    """
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch == "\\":                                   # backslash escape
            if i + 1 < n:
                out.append(esc_escaped(src[i + 1]))
                i += 2
            else:
                i += 1
        elif ch == "`":                                  # inline code
            j = src.find("`", i + 1)
            if j == -1:
                out.append(r"\`")
                i += 1
            else:
                out.append(r"\texttt{" + esc_tt(src[i + 1:j]) + "}")
                i = j + 1
        elif ch == "$":                                  # inline math
            j = src.find("$", i + 1)
            if j == -1:
                out.append(r"\$")
                i += 1
            else:
                out.append("$" + src[i + 1:j] + "$")
                i = j + 1
        elif ch == "*":                                  # bold / italic
            if src.startswith("**", i):
                j = src.find("**", i + 2)
                if j == -1:
                    out.append(r"\*")
                    i += 1
                else:
                    out.append(r"\textbf{" + inline_to_latex(src[i + 2:j]) + "}")
                    i = j + 2
            else:
                j = src.find("*", i + 1)
                if j == -1:
                    out.append(r"\*")
                    i += 1
                else:
                    out.append(r"\textit{" + inline_to_latex(src[i + 1:j]) + "}")
                    i = j + 1
        else:                                            # plain run
            j = i
            while j < n and src[j] not in "\\`$*":
                j += 1
            out.append(esc_plain(src[i:j]))
            i = j
    return "".join(out)


# --------------------------------------------------------------------------
# Block-level Markdown parsing
# --------------------------------------------------------------------------

META_RE = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$")
OL_RE = re.compile(r"^(\d+)\.\s+(.*)$")
HR_RE = re.compile(r"^-{3,}$")


def parse_blocks(md: str):
    """Split the Markdown source into a list of typed blocks."""
    lines = md.split("\n")
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):                        # ATX heading
            blocks.append(("heading", [line[2:].strip()]))
            i += 1
            continue
        if HR_RE.match(line):                             # horizontal rule
            blocks.append(("hr", []))
            i += 1
            continue
        if OL_RE.match(line):                            # ordered list
            items = []
            while i < n:
                m = OL_RE.match(lines[i].strip())
                if not m:
                    break
                items.append(m.group(2))
                i += 1
            blocks.append(("list", items))
            continue
        # paragraph: consecutive non-blank lines
        para = []
        while i < n:
            cur = lines[i].strip()
            if not cur or cur.startswith("# ") or HR_RE.match(cur) or OL_RE.match(cur):
                break
            para.append(cur)
            i += 1
        blocks.append(("para", para))

    # A run of "**Label:** value" lines is a metadata block: keep one per line.
    out = []
    for kind, payload in blocks:
        if kind == "para" and len(payload) > 1 and all(META_RE.match(p) for p in payload):
            out.append(("meta", payload))
        else:
            out.append((kind, payload))
    return out


# --------------------------------------------------------------------------
# Block -> LaTeX
# --------------------------------------------------------------------------


def render_blocks(blocks) -> str:
    tex: list[str] = []
    last = len(blocks) - 1
    for idx, (kind, payload) in enumerate(blocks):
        if kind == "heading":
            tex.append(r"\begin{center}")
            tex.append(r"{\Large\bfseries " + inline_to_latex(payload[0]) + r"\par}")
            tex.append(r"\end{center}")
        elif kind == "meta":
            for line in payload:
                m = META_RE.match(line)
                label, rest = m.group(1), m.group(2)
                tex.append(
                    r"\noindent\textbf{" + inline_to_latex(label) + ":}"
                    + ((" " + inline_to_latex(rest)) if rest else "")
                    + r"\par"
                )
        elif kind == "hr":
            tex.append(r"\vspace{0.10em}\hrule\vspace{0.42em}")
        elif kind == "para":
            if idx == last:                              # closing block: keep line breaks
                for k, line in enumerate(payload):
                    tail = r"\\" if k < len(payload) - 1 else r"\par"
                    tex.append(r"\noindent " + inline_to_latex(line) + tail)
            else:
                tex.append(inline_to_latex(" ".join(payload)) + r"\par")
        elif kind == "list":
            tex.append(
                r"\begin{enumerate}[label=\arabic*., leftmargin=2.2em,"
                r" itemsep=0.20em, topsep=0.26em, parsep=0pt]"
            )
            for item in payload:
                tex.append(r"\item " + inline_to_latex(item))
            tex.append(r"\end{enumerate}")
    return "\n".join(tex) + "\n"


# --------------------------------------------------------------------------
# LaTeX wrapper
# --------------------------------------------------------------------------

PREAMBLE = r"""% ---------------------------------------------------------------------------
% Generated by _make_coverletter.py from cover_letter.md -- do not edit by hand.
% ---------------------------------------------------------------------------
\documentclass[10.5pt,a4paper]{article}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newtxtext,newtxmath}
\usepackage[a4paper,top=2.4cm,bottom=2.4cm,left=2.6cm,right=2.6cm]{geometry}
\usepackage{setspace}
\usepackage{enumitem}
\usepackage{microtype}

\setstretch{1.25}          % 1.25 line spacing
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.22em}
\pagestyle{empty}          % cover letter: no page numbers
\emergencystretch=2.5em    % avoid overfull lines when justifying

\begin{document}

\input{_coverletter_body.tex}

\end{document}
"""


# --------------------------------------------------------------------------
# Build helpers
# --------------------------------------------------------------------------


def run_engine() -> None:
    env = os.environ.copy()
    env["PATH"] = str(MIKTEX_BIN) + os.pathsep + env.get("PATH", "")
    env.setdefault("MIKTEX_ENABLE_INSTALLER", "t")

    cmd = [
        ENGINE,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-jobname={JOB}",
        WRAP_TEX.name,
    ]
    log: list[str] = []
    for run in range(1, PASSES + 1):
        proc = subprocess.run(
            cmd, cwd=str(HERE), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        log.append(f"===== pass {run} :: exit {proc.returncode} =====\n")
        log.append(proc.stdout or "")
        log.append(proc.stderr or "")
        if proc.returncode != 0:
            BUILD_LOG.write_text("".join(log), encoding="utf-8")
            raise SystemExit(
                f"{ENGINE} failed on pass {run} (exit {proc.returncode}).\n"
                f"--- log tail ---\n{(proc.stdout or '')[-3000:]}\n"
                f"(full log: {BUILD_LOG})"
            )
    BUILD_LOG.write_text("".join(log), encoding="utf-8")


def extract_text() -> str:
    env = os.environ.copy()
    env["PATH"] = str(MIKTEX_BIN) + os.pathsep + env.get("PATH", "")
    cmd = [str(MIKTEX_BIN / "pdftotext"), "-enc", "UTF-8", str(PDF_PATH), str(TXT_PATH)]
    proc = subprocess.run(cmd, cwd=str(HERE), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"pdftotext failed: {proc.stderr}")
    return TXT_PATH.read_text(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

# Forbidden Markdown residue in the rendered PDF text layer.
# NOTE: '_' is deliberately absent -- it belongs to the legitimate file names
# baseline_snr.py / wsu_eval.py / drone_acoustic_joint.py that requirement 6
# demands to be preserved verbatim in monospace.
RESIDUE_PATTERNS = [
    ("double asterisk '**'", r"\*\*"),
    ("backtick '`'", r"`"),
    ("hash '#'", r"#"),
    ("triple hyphen '---'", r"-{3,}"),
    ("raw markdown list bullet", r"(?m)^\s*\*\s"),
    ("raw markdown heading '# '", r"(?m)^#\s"),
    ("raw markdown rule '---'", r"(?m)^-{3,}\s*$"),
    ("raw markdown bold '**x**'", r"\*\*[^*\n]+\*\*"),
]

# Filenames that legitimately contain underscores.
CODE_FILENAMES = ("drone_acoustic_joint.py", "baseline_snr.py", "wsu_eval.py",
                  "plot_wsu.py", "data_snr.csv", "baseline_snr.csv",
                  "table1_repro.csv")

_NORM_MAP = {
    "’": "'", "‘": "'",        # curly single quotes -> ASCII
    "“": '"', "”": '"',
    "∗": "*",                    # math asterisk operator (from $\gamma^*$)
}

_MATH_UNICODE = {
    "\\lambda": "λ", "\\gamma": "γ", "\\alpha": "α", "\\beta": "β",
    "\\mu": "μ", "\\sigma": "σ", "\\pi": "π", "\\theta": "θ",
    "\\Delta": "Δ", "\\Sigma": "Σ", "\\times": "×", "\\cdot": "·",
}


def md_to_plain(text: str) -> str:
    """Reduce a Markdown fragment to the literal text a reader should see.

    Code spans, math spans and backslash escapes are lifted out into
    placeholders first, so the emphasis-stripping regexes can never eat a
    '\\*' inside code, a '$...$' math expression such as $\\gamma^*$, or the
    escaped corresponding-author asterisk.
    """
    stash: list[str] = []

    def hold(value: str) -> str:
        stash.append(value)
        return f"\x00{len(stash) - 1}\x00"

    # code spans -> their literal content
    text = re.sub(r"`([^`]*)`", lambda m: hold(m.group(1)), text)

    # math spans -> the characters the PDF text layer will carry
    def math(m: re.Match) -> str:
        s = m.group(1)
        for k, v in _MATH_UNICODE.items():
            s = s.replace(k, v)
        s = s.replace("{", "").replace("}", "").replace("\\", "")
        s = s.replace("_", "").replace("^", "")   # sub/superscript markers
        return hold(s)

    text = re.sub(r"\$([^$]*)\$", math, text)

    # backslash-escaped characters -> the literal character
    text = re.sub(r"\\(.)", lambda m: hold(m.group(1)), text)

    # drop the remaining markup markers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)

    # restore the stashed code / math / escaped content
    def restore(m: re.Match) -> str:
        return stash[int(m.group(1))]

    return re.sub(r"\x00(\d+)\x00", restore, text)


def normalize(text: str) -> str:
    """Canonicalise a text layer for comparison: NFKC-fold math letters
    (U+1D706 -> lambda), normalise quotes and the math asterisk operator."""
    text = unicodedata.normalize("NFKC", text)
    for k, v in _NORM_MAP.items():
        text = text.replace(k, v)
    return text


def squash(text: str) -> str:
    """Collapse all whitespace: makes the comparison immune to line breaks
    and to TeX hyphenation at a line end."""
    return re.sub(r"\s+", "", text)


def verify(txt: str, md: str) -> int:
    problems = 0
    print("\n===== verification =====")
    print(f"PDF: {PDF_PATH}  ({PDF_PATH.stat().st_size} bytes)")

    # --- page count -------------------------------------------------------
    pages = -1
    try:
        import pypdf
        pages = len(pypdf.PdfReader(str(PDF_PATH)).pages)
    except Exception as exc:                              # pragma: no cover
        print(f"page count unavailable: {exc}")
    print(f"pages: {pages}")

    # --- Markdown residue -------------------------------------------------
    print("\n-- Markdown residue scan over the full extracted text --")
    for name, pat in RESIDUE_PATTERNS:
        hits = re.findall(pat, txt)
        print(f"  {name:30s} : {'OK (absent)' if not hits else 'FAIL ' + str(len(hits)) + ' hit(s)'}")
        if hits:
            problems += 1

    # --- underscores may only occur inside the legitimate file names ------
    # Strip the three known file names (and their trailing punctuation) first,
    # then require that no underscore survives anywhere else.
    scan = txt
    for fn in CODE_FILENAMES:
        scan = scan.replace(fn, "")
    stray_us = sorted({m.group(0) for m in re.finditer(r"\S*_\S*", scan) if "_" in m.group(0)})
    print(f"  {'underscores outside file names':30s} : "
          f"{'OK (none)' if not stray_us else 'FAIL ' + str(stray_us)}")
    if stray_us:
        problems += 1

    # --- asterisk audit (every ASCII '*' must be intentional) -------------
    expect_stars = len(re.findall(r"\\\*", md))
    stars = [(m.start(), txt[max(0, m.start() - 26):m.start() + 26].replace("\n", " "))
             for m in re.finditer(r"\*", txt)]
    print("\n-- asterisk audit --")
    print(f"  expected intentional ASCII '*' in source : {expect_stars}")
    print(f"  ASCII '*' found in PDF text               : {len(stars)}")
    for pos, ctx in stars:
        print(f"      @{pos}: ...{ctx}...")
    if len(stars) != expect_stars:
        print("  FAIL: asterisk count differs from the source")
        problems += 1
    else:
        print("  OK: the only ASCII '*' is the corresponding-author mark")

    # --- required literal strings ----------------------------------------
    print("\n-- required literal strings --")
    for needle in CODE_FILENAMES + ("--mode sim",):
        ok = needle in txt
        print(f"  {needle:28s} : {'OK (present)' if ok else 'FAIL (missing)'}")
        if not ok:
            problems += 1

    # --- title / sign-off -------------------------------------------------
    print("\n-- title & sign-off --")
    new_title = ("Joint Detection and Azimuth Estimation of UAVs Using MEMS Microphone "
                 "Arrays and Multi-Task Deep Learning: From Simulation to Real-World "
                 "Recordings")
    old_title = ("End-to-End Joint Drone Detection and DOA Estimation with MEMS "
                 "Microphone Arrays")
    flat = squash(normalize(txt))
    for label, ok in [
        ("new title present", squash(normalize(new_title)) in flat),
        ("old title absent", squash(normalize(old_title)) not in flat),
        ("'Sincerely,' present", squash(normalize("Sincerely,")) in flat),
        ("sign-off 'Liwei Tian' present", squash(normalize("Liwei Tian")) in flat),
        ("sign-off 'Corresponding author' present",
         squash(normalize("Corresponding author")) in flat),
        ("sign-off email present",
         squash(normalize("tianliwei@gdust.edu.cn")) in flat),
        ("sign-off 'on behalf of all authors' present",
         squash(normalize("on behalf of all authors")) in flat),
        ("'Liwei Tian*' present", squash(normalize("Liwei Tian*")) in flat),
    ]:
        print(f"  {label:24s} : {'OK' if ok else 'FAIL'}")
        if not ok:
            problems += 1

    # --- block-level content cross-check ----------------------------------
    print("\n-- block cross-check: every md block present in the PDF text --")
    missing = 0
    for kind, payload in parse_blocks(md):
        for item in payload:
            plain = md_to_plain(item).strip()
            if len(squash(plain)) < 8:
                continue
            if squash(normalize(plain)) not in flat:
                missing += 1
                print(f"  MISSING [{kind}]: {plain[:90]!r}")
    print(f"  blocks not found: {missing}")
    if missing:
        problems += missing
    else:
        print("  OK: all md blocks (heading, metadata, paragraphs, list items) "
              "are present verbatim in the PDF text")

    print("\nresult:", "PASS" if problems == 0 else f"FAIL ({problems} problem(s))")
    return problems


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    if not MD_PATH.exists():
        raise SystemExit(f"missing source: {MD_PATH}")
    md = MD_PATH.read_text(encoding="utf-8")

    body = render_blocks(parse_blocks(md))
    BODY_TEX.write_text(body, encoding="utf-8", newline="\n")
    WRAP_TEX.write_text(PREAMBLE, encoding="utf-8", newline="\n")
    print(f"wrote {BODY_TEX.name} ({len(body)} bytes) and {WRAP_TEX.name}")

    if PDF_PATH.exists():
        PDF_PATH.unlink()
    run_engine()
    if not PDF_PATH.exists():
        raise SystemExit("pdflatex produced no coverletter.pdf")
    print(f"wrote {PDF_PATH.name} ({PDF_PATH.stat().st_size} bytes)")

    txt = extract_text()
    return 1 if verify(txt, md) else 0


if __name__ == "__main__":
    sys.exit(main())
