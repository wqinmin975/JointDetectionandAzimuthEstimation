#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export all 5 figures of paper_mdpi.pdf as standalone high-resolution PNGs.

Figure 1-3 are vector TikZ/pgfplots drawings embedded in the PDF: they are
rendered from a tight clip at 4x zoom (~288 dpi).
Figure 4-5 are raster images embedded in the PDF: the original embedded
pixmaps (3930x1230 and 3300x1260) are extracted losslessly.

All clip boxes were verified against the PDF text-block coordinates; figure
captions are deliberately excluded so the exports are pure artwork.
"""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[3]
PDF = ROOT / "paper_mdpi.pdf"
OUT = ROOT / "output" / "rr-mdpi-20261126" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# page index (0-based), clip rect, output filename
VECTOR_FIGURES = [
    (4, (164, 258, 563, 391), "Figure_1_Testbed_Geometry.png"),
    (5, (164, 333, 563, 502), "Figure_2_Network_Architecture.png"),
    (12, (164, 310, 563, 439), "Figure_3_SNR_Sensitivity.png"),
]
# page index, expected embedded image size, output filename
RASTER_FIGURES = [
    (17, (3930, 1230), "Figure_4_IDMT_Validation.png"),
    (19, (3300, 1260), "Figure_5_WSU_Baselines.png"),
]

doc = fitz.open(PDF)
manifest = []

for pno, clip, name in VECTOR_FIGURES:
    page = doc[pno]
    pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=fitz.Rect(clip), alpha=False)
    path = OUT / name
    pix.save(path)
    manifest.append((name, pix.width, pix.height, f"page {pno+1} vector clip @4x"))
    print(f"rendered {name}: {pix.width}x{pix.height}")

for pno, expected, name in RASTER_FIGURES:
    page = doc[pno]
    images = page.get_image_info(xrefs=True)
    candidates = [im for im in images if (im["width"], im["height"]) == expected]
    if not candidates:
        raise RuntimeError(f"expected image {expected} not found on page {pno+1}")
    xref = candidates[0]["xref"]
    info = doc.extract_image(xref)
    path = OUT / name
    path.write_bytes(info["image"])
    manifest.append((name, info["width"], info["height"], f"page {pno+1} embedded original"))
    print(f"extracted {name}: {info['width']}x{info['height']} ({info['ext']})")

(OUT / "manifest.txt").write_text(
    "\n".join(f"{n}\t{w}x{h}\t{src}" for n, w, h, src in manifest) + "\n",
    encoding="utf-8",
)
print(f"\n{len(manifest)} figures exported to {OUT}")
