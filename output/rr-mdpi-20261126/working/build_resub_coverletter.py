#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build coverletter_resubmission.pdf via the existing _make_coverletter.py,
with input/output constants redirected so the original cover letter is intact.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("_mcl", ROOT / "_make_coverletter.py")
m = importlib.util.module_from_spec(spec)
sys.modules["_mcl"] = m
spec.loader.exec_module(m)

m.MD_PATH = ROOT / "output/rr-mdpi-20261126/stage1/cover_letter_resubmission.md"
m.BODY_TEX = ROOT / "_resub_coverletter_body.tex"
m.WRAP_TEX = ROOT / "_resub_coverletter.tex"
m.PDF_PATH = ROOT / "coverletter_resubmission.pdf"
m.JOB = "coverletter_resubmission"
m.BUILD_LOG = ROOT / "_resub_coverletter_build.log"
m.TXT_PATH = ROOT / "_resub_coverletter_text.txt"
m.PREAMBLE = m.PREAMBLE.replace("_coverletter_body.tex", m.BODY_TEX.name)

# The draft uses bare Unicode degree / minus / plus-minus; the stock escaper
# only covers dashes and quotes. Add LaTeX-safe mappings without touching the
# delivered Markdown source.
m._PLAIN_ESC = dict(m._PLAIN_ESC)
m._PLAIN_ESC["\u00b0"] = r"$^\circ$"   # degree sign
m._PLAIN_ESC["\u2212"] = r"$-$"        # mathematical minus
m._PLAIN_ESC["\u00b1"] = r"$\pm$"      # plus-minus

sys.exit(m.main())
