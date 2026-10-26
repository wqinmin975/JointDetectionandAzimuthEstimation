# -*- coding: utf-8 -*-
"""Convert paper_v8.tex (elsarticle) -> paper_mdpi.tex (MDPI Sensors 2023 class)."""
import re

SRC = "paper_v8.tex"
OUT = "paper_mdpi.tex"

with open(SRC, encoding="utf-8") as f:
    src = f.read()

# ---- extract title / abstract / keywords from the elsarticle frontmatter ----
title = re.search(r"\\title\{(.*?)\}", src, re.S).group(1).strip()
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S)
abstract = m.group(1).strip()
m = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", src, re.S)
keywords = m.group(1).strip().replace(" \\sep ", "; ")

# ---- body: between \end{frontmatter} and \section*{Acknowledgements} ----
i_front = src.index("\\end{frontmatter}")
i_ack = src.index("\\section*{Acknowledgements}")
body = src[i_front + len("\\end{frontmatter}"):i_ack]

# ---- transform body for MDPI style ----
body = body.replace("\\begin{figure*}[t]", "\\begin{figure}[H]")
body = body.replace("\\end{figure*}", "\\end{figure}")
body = body.replace("\\begin{table*}[t]", "\\begin{table}[H]")
body = body.replace("\\end{table*}", "\\end{table}")
body = body.replace("\\begin{table}[t]", "\\begin{table}[H]")
body = body.replace("\\begin{figure}[t]", "\\begin{figure}[H]")
body = body.replace("Fig.~\ref", "Figure~\ref")
# The author-biography block is rendered by MDPI in the back matter (after the
# Conflicts of Interest statement); strip the source copy from the body to avoid
# duplicating it.
i_bio = body.find("\\section*{Author biographies}")
if i_bio >= 0:
    body = body[:i_bio]
# scale the architecture tikzpicture once (only the arch block starts with font=\footnotesize,)
body = body.replace("    font=\\footnotesize,", "    scale=0.82, transform shape, font=\\footnotesize,", 1)

# ---- MDPI preamble + frontmatter ----
preamble = (
    "% Converted to MDPI Sensors format (MDPI LaTeX class, 25 March 2023).\n"
    "\\documentclass[sensors,article,submit,pdftex,moreauthors]{Definitions/mdpi}\n\n"
    "%=================================================================\n"
    "% MDPI internal commands - do not modify\n"
    "\\firstpage{1}\n"
    "\\pubvolume{1}\n"
    "\\issuenum{1}\n"
    "\\articlenumber{0}\n"
    "\\pubyear{2026}\n"
    "\\copyrightyear{2026}\n"
    "\\datereceived{ }\n"
    "\\daterevised{ }\n"
    "\\dateaccepted{ }\n"
    "\\datepublished{ }\n"
    "\\hreflink{https://doi.org/}\n\n"
    "%=================================================================\n"
    "% Packages required by this manuscript. The MDPI class already loads\n"
    "% amsmath, amssymb, graphicx, booktabs, tikz, hyperref, caption, etc.\n"
    "\\usepackage{pgfplots}\n"
    "\\pgfplotsset{compat=1.17}\n"
    "\\usetikzlibrary{arrows.meta,positioning,calc,fit}\n"
    "\\usepackage{subcaption}\n"
    "\\usepackage{siunitx}\n\n"
    "% Author ORCID identifiers\n"
    "\\newcommand{\\orcidauthorA}{0009-0006-0447-0482} % Qinmin Wang\n"
    "\\newcommand{\\orcidauthorD}{0009-0007-4278-771X} % Hongyuan Wang\n"
    "\\newcommand{\\orcidauthorE}{0009-0006-0345-654X} % Liwei Tian\n\n"
    "%=================================================================\n"
    "\\Title{" + title + "}\n\n"
    "\\TitleCitation{" + title + "}\n\n"
    "% Authors (order: 1 Qinmin Wang, 2 Zixin Wang, 3 Zhenzhen Hu, 4 Hongyuan Wang, 5 Liwei Tian*)\n"
    "\\Author{Qinmin Wang $^{1}$\\orcidA{}, Zixin Wang $^{2}$, Zhenzhen Hu $^{3}$, Hongyuan Wang $^{1}$\\orcidD{} and Liwei Tian $^{1,*}$\\orcidE{}}\n\n"
    "\\AuthorNames{Qinmin Wang, Zixin Wang, Zhenzhen Hu, Hongyuan Wang and Liwei Tian}\n"
    "\\AuthorCitation{Wang, Q.; Wang, Z.; Hu, Z.; Wang, H.; Tian, L.}\n\n"
    "\\address{%\n"
    "$^{1}$ \\quad School of Computer and Artificial Intelligence, Guangdong University of Science and Technology, Dongguan 523083, China; wangqinmin@gdust.edu.cn (Q.W.); wanghongyuan@gdust.edu.cn (H.W.); tianliwei@gdust.edu.cn (L.T.)\n\n"
    "$^{2}$ \\quad College of Electronic and Communication Engineering, Tianjin Normal University, Tianjin 300387, China; 396514839@qq.com (Z.W.)\n\n"
    "$^{3}$ \\quad Chengdu University of Information Technology, Chengdu 610225, China; hzzuestc@163.com (Z.H.)}\n\n"
    "\\corres{Correspondence: tianliwei@gdust.edu.cn}\n\n"
    "\\abstract{" + abstract + "}\n\n"
    "\\keyword{" + keywords + "}\n\n"
    "% -------------------------------------------------------------------\n"
    "% Clean typeset PDF: the MDPI class enables marginal line numbering in\n"
    "% the 'submit' status; neutralise the switch so no line numbers are drawn.\n"
    "\\renewcommand{\\linenumbers}{}\n"
    "% -------------------------------------------------------------------\n"
    "% The class-provided \\bio command lays the author biography out in a fixed\n"
    "% 13.86 cm pbox (Definitions/mdpi.cls), which is the width of the MDPI\n"
    "% published text block but is wider than the 'submit' text width once the\n"
    "% name column and the 1.1 cm gap are added -- it overflows the margin by\n"
    "% ~77 pt. Redefine it with an adaptive width so the block fits this class's\n"
    "% text width while keeping the same two-column name/text house layout.\n"
    "\\makeatletter\n"
    "\\renewcommand{\\bio}[2]{\\noindent\\makebox[2.2cm][l]{#1}\\noindent"
    "{\\fontsize{9}{11.2}\\selectfont \\pbox[b]{\\dimexpr\\textwidth-2.2cm\\relax}{#2}}"
    "\\par\\vspace{6pt}}\n"
    "\\makeatother\n"
    "\\begin{document}\n"


)

# ---- MDPI meta block (back matter) ----
# Order follows the MDPI Sensors back-matter sequence required by the journal's
# Instructions for Authors: Supplementary Materials -> Author Contributions ->
# Funding -> Institutional Review Board Statement -> Informed Consent Statement
# -> Data Availability Statement -> Acknowledgments -> Conflicts of Interest ->
# References -> Author Biographies.
#
# NOTE on escaping: every LaTeX backslash below is written as \\ in this
# non-raw Python string, so the generated file receives a single literal
# backslash. A bare "\b" would be parsed by Python as U+0008 (backspace) and
# silently inject a control character into the manuscript source.
meta = (
    "\n%=================================================================\n"
    "\\supplementary{The reproducibility package containing the simulation code, the exported "
    "result tables, and the rendered validation figures is provided as supplementary material; "
    "the accompanying LaTeX source of this manuscript and the MDPI class files are included in "
    "the same package.}\n\n"
    "\\authorcontributions{Conceptualization, Q.W. and L.T.; methodology, Q.W. and L.T.; "
    "software, Q.W.; validation, Q.W., Z.W., Z.H., H.W. and L.T.; formal analysis, Q.W.; "
    "investigation, Q.W.; resources, L.T.; data curation, Q.W.; writing---original draft "
    "preparation, Q.W.; writing---review and editing, Z.W., Z.H., H.W. and L.T.; visualization, "
    "Q.W.; supervision, L.T.; project administration, L.T.; funding acquisition, L.T. All authors "
    "have read and agreed to the published version of the manuscript.}\n\n"
    "\\funding{This research was funded by the National Natural Science Foundation of China under "
    "Grant No. 62101076, and by the 2025 Science and Technology Innovation Capability Enhancement "
    "Project of Guangdong University of Science and Technology under Grant No. GKY-2025BSQDK-46.}\n\n"
    "\\institutionalreview{Not applicable. This study is a numerical simulation combined with the "
    "evaluation of two publicly available, peer-reviewed acoustic corpora; it involved no human "
    "participants, no identifiable personal data, and no animal experiments.}\n\n"
    "\\informedconsent{Not applicable. This study involved no human participants; the two recorded "
    "drone-audio corpora used for validation are publicly available and contain no personally "
    "identifiable information.}\n\n"
    "\\dataavailability{The synthetic dataset and the accompanying simulation code are released by "
    "the authors. The recorded-drone corpora used for validation are publicly available: the "
    "Fraunhofer IDMT drone-audio dataset \\cite{idmt2025drone} and the Western Sydney University "
    "UAV Acoustic Localization Dataset \\cite{wsu2024dataset}.}\n\n"
    "\\acknowledgments{The authors acknowledge the support of the National Natural Science Foundation "
    "of China and Guangdong University of Science and Technology.}\n\n"
    "\\conflictsofinterest{The authors declare no conflicts of interest.}\n\n"

    "%=================================================================\n"
    "% References section title (the MDPI class leaves \\@reftitle empty by default)\n"
    "\\reftitle{References}\n\n"
    "\\externalbibliography{yes}\n"
    "\\bibliography{refs}\n\n"

    "%=================================================================\n"
    "% Author biographies. The MDPI class provides the \\bio command for this block\n"
    "% (Definitions/mdpi.cls: \\newcommand{\\bio}[2]); it sets the author name in a\n"
    "% narrow left column with the biography text in a 9pt pbox beside it, which is the\n"
    "% MDPI house format. The block is placed after the References.\n"
    "\\section*{Author Biographies}\n\n"
    "\\bio{Liwei Tian}{received his M.Sc. degree from Guangxi Normal University, Guilin, China, in 2009, "
    "and his Ph.D. degree from Macau University of Science and Technology, Macau, China, in 2019. He is "
    "currently a Professor with the School of Computer and Artificial Intelligence, Guangdong University "
    "of Science and Technology, Dongguan, China. He is a Senior Member of the China Computer Federation "
    "(CCF) and a member of its Dongguan Branch Executive Committee, a member of the Technical Committee "
    "on Intelligent Simulation, Optimization and Scheduling of the Chinese Society of Simulation, and a "
    "member of the Technical Committee on Discrete and Intelligent Computing of the Chinese Association "
    "of Artificial Intelligence. His research interests include array signal processing, acoustic source "
    "localization, and multi-task deep learning for UAV detection and localization.}\n\n"

    "\\end{document}\n"
)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(preamble + body + meta)

print("Wrote", OUT, "chars:", len(preamble + body + meta))
