#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent verification of coverletter.pdf against cover_letter.md.

Deliberately does NOT import anything from _make_coverletter.py: it re-derives
the expected plain text from the Markdown with its own independent logic and
uses PyMuPDF (a different PDF text extractor than the pdftotext used by the
build script) so a bug in one extractor cannot hide a defect.

Run:  python _verify_coverletter.py
"""
import re
import sys
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF

HERE = Path(__file__).resolve().parent
MD = HERE / "cover_letter.md"
PDF = HERE / "coverletter.pdf"

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)


print("=" * 68)
print("INDEPENDENT VERIFICATION  (PyMuPDF extractor, independent md parser)")
print("=" * 68)

# ---------------------------------------------------------------- extract
doc = fitz.open(str(PDF))
pages = doc.page_count
text = "\n".join(page.get_text() for page in doc)
print(f"\npdf: {PDF.name}   pages: {pages}   chars: {len(text)}")

md = MD.read_text(encoding="utf-8")

# ------------------------------------------------- 1. markdown residue
print("\n-- 1. Markdown marker residue in the PDF text layer --")
residue = {
    "'**'  (bold marker)": r"\*\*",
    "'`'   (code marker)": r"`",
    "'#'   (heading mark)": r"#",
    "'---' (hr / em-dash)": r"-{3,}",
}
for label, pat in residue.items():
    hits = re.findall(pat, text)
    check(f"no {label}", not hits, f"hits={hits[:5]}")

# The only ASCII '*' allowed is the corresponding-author mark after the name.
stars = [m.start() for m in re.finditer(r"\*", text)]
check("exactly one ASCII '*' (corresponding-author mark)",
      len(stars) == 1,
      f"found {len(stars)} at {stars}")
if len(stars) == 1:
    ctx = text[max(0, stars[0] - 30):stars[0] + 30].replace("\n", " ")
    check("that '*' sits right after 'Liwei Tian'",
          "Liwei Tian*" in ctx, f"context={ctx!r}")

# '_' is legitimate only inside the three file names.
scan = text
for fn in ("drone_acoustic_joint.py", "baseline_snr.py", "wsu_eval.py"):
    scan = scan.replace(fn, "")
stray = sorted({m.group(0) for m in re.finditer(r"\S*_\S*", scan) if "_" in m.group(0)})
check("no '_' outside the three file names", not stray, f"stray={stray}")

# ------------------------------------------------- 2. required strings
print("\n-- 2. Required literal strings --")
for s in ("baseline_snr.py", "wsu_eval.py", "drone_acoustic_joint.py",
          "--mode sim", "python drone_acoustic_joint.py --mode sim"):
    check(f"contains {s!r}", s in text)

# ------------------------------------------------- 3. title / sign-off
print("\n-- 3. Title & sign-off --")
NEW = ("Joint Detection and Azimuth Estimation of UAVs Using MEMS Microphone Arrays "
       "and Multi-Task Deep Learning: From Simulation to Real-World Recordings")
OLD = ("End-to-End Joint Drone Detection and DOA Estimation with MEMS Microphone "
       "Arrays: Multi-Task Deep Learning Using Uncertainty-Dynamic Loss Weighting")
flat = re.sub(r"\s+", " ", text)
check("new title present verbatim", NEW in flat)
check("old title absent", OLD not in flat)
check("'Sincerely,' present", "Sincerely," in flat)
check("sign-off 'Liwei Tian' present", "Liwei Tian" in flat)
check("sign-off 'Corresponding author' present", "Corresponding author" in flat)
check("sign-off email 'tianliwei@gdust.edu.cn' present",
      "tianliwei@gdust.edu.cn" in flat)
check("sign-off 'on behalf of all authors' present",
      "on behalf of all authors" in flat)
check("MDPI originality declaration present",
      "under consideration for publication" in flat)

# ------------------------------------------------- 4. paragraph parity
print("\n-- 4. Every md paragraph appears verbatim in the PDF --")


def to_plain(fragment):
    """Independent Markdown -> plain-text reducer.

    Code spans, math spans and backslash escapes are parked in placeholders
    first, so the emphasis regexes cannot eat the '*' inside $\\gamma^*$.
    """
    park = []

    def hold(v):
        park.append(v)
        return f"\x00{len(park) - 1}\x00"

    fragment = re.sub(r"`([^`]*)`", lambda m: hold(m.group(1)), fragment)

    greek = {"lambda": "λ", "gamma": "γ", "alpha": "α", "beta": "β",
             "mu": "μ", "sigma": "σ", "pi": "π", "theta": "θ"}

    def m(mth):
        s = mth.group(1)
        for k, v in greek.items():
            s = s.replace("\\" + k, v)
        s = s.replace("{", "").replace("}", "").replace("\\", "")
        s = s.replace("_", "").replace("^", "")
        return hold(s)

    fragment = re.sub(r"\$([^$]*)\$", m, fragment)
    fragment = re.sub(r"\\(.)", lambda m: hold(m.group(1)), fragment)
    fragment = re.sub(r"\*\*(.+?)\*\*", r"\1", fragment)
    fragment = re.sub(r"\*(.+?)\*", r"\1", fragment)
    return re.sub(r"\x00(\d+)\x00", lambda m: park[int(m.group(1))], fragment)


def _base(s):
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("∗", "*")
    s = s.replace("—", "--").replace("–", "-")
    return s


def norm(s):
    """Canonical form, with a line-break hyphen dropped (auto-hyphenation)."""
    s = re.sub(r"-\s*\n", "", _base(s))
    return re.sub(r"\s+", "", s)


def norm_keep_hyphen(s):
    """Canonical form, with the line-break hyphen kept."""
    s = re.sub(r"-\s*\n", "-", _base(s))
    return re.sub(r"\s+", "", s)


pdf_norm = norm(text)
pdf_norm_keep = norm_keep_hyphen(text)

# split md into comparable units
units = []
for block in re.split(r"\n\s*\n", md):
    b = block.strip()
    if not b:
        continue
    if re.fullmatch(r"-{3,}", b):
        continue
    if b.startswith("# "):
        units.append(("heading", b[2:].strip()))
        continue
    # a block may mix leading prose lines with a numbered list (no blank line
    # between them in the source), so split it line by line.
    prose, items = [], []
    for line in b.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"\d+\.\s+(.*)", line)
        if m:
            items.append(m.group(1))
        elif items:
            items[-1] += " " + line          # continuation of the list item
        else:
            prose.append(line)
    if prose:
        units.append(("para", " ".join(prose)))
    for it in items:
        units.append(("list", it))

missing = []
for kind, unit in units:
    plain = to_plain(unit).strip()
    if len(norm(plain)) < 8:
        continue
    n_drop = norm(plain)          # PDF variant with the line-break hyphen removed
    n_keep = norm_keep_hyphen(plain)
    if n_drop not in pdf_norm and n_keep not in pdf_norm_keep:
        missing.append((kind, plain))
for kind, plain in missing:
    print(f"      MISSING [{kind}]: {plain[:100]!r}")
check(f"all {len(units)} md units present in PDF", not missing,
      f"missing={len(missing)}")

# ------------------------------------------------- 5. monospace check
print("\n-- 5. File names are set in a monospace font --")
monospace_fonts = set()
for page in doc:
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if any(k in span["text"] for k in ("baseline_snr.py", "wsu_eval.py",
                                                   "drone_acoustic_joint.py")):
                    monospace_fonts.add((span["font"], round(span["size"], 1)))
print(f"      fonts carrying the file names: {sorted(monospace_fonts)}")
check("file names use a monospace (typewriter) font",
      bool(monospace_fonts) and all(
          re.search(r"tt|cmtt|ntxtt|courier", f[0], re.I) for f in monospace_fonts),
      str(sorted(monospace_fonts)))

# ------------------------------------------------- 6. page geometry
print("\n-- 6. Page geometry --")
r = doc[0].rect
check("A4 page size (595 x 842 pt)",
      abs(r.width - 595.28) < 2 and abs(r.height - 841.89) < 2,
      f"{r.width:.1f} x {r.height:.1f} pt")

print("\n" + "=" * 68)
print("RESULT:", "PASS - all independent checks succeeded" if not fails
      else f"FAIL - {len(fails)} check(s) failed: {fails}")
print("=" * 68)
sys.exit(1 if fails else 0)
