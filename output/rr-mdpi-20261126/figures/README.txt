FIGURE PACKAGE — manuscript sensors-4688906 (MDPI Sensors, revised submission)
Source: paper_mdpi.pdf, compiled 2026-11-24 from paper_mdpi.tex (24 pages).

Contents (5 figures, matching the figure numbering of the manuscript):

  Figure_1_Testbed_Geometry.png      1596 x 532   Fig. 1  Test-bed geometry and azimuth convention
                                                          (a) six-element ULA; (b) eight-element IDMT cubic array
  Figure_2_Network_Architecture.png  1596 x 676   Fig. 2  Architecture of the six-microphone raw-waveform network
                                                          (solid = 6-mic simulation net; dashed = separate 8-mic net)
  Figure_3_SNR_Sensitivity.png       1596 x 516   Fig. 3  Pd and 180-deg-folded azimuth MAE versus SNR
                                                          NOTE: subplot (a) y-axis is truncated and starts at 0.98;
                                                          the true Pd range is [0,1]. See the figure caption.
  Figure_4_IDMT_Validation.png       3930 x 1230  Fig. 4  IDMT zero-shot / fine-tuned scatter and per-type MAE
                                                          (both scatter panels include the black dashed y=x diagonal)
  Figure_5_WSU_Baselines.png         3300 x 1260  Fig. 5  WSU ROC (chance diagonal) and classical-baseline MAE
                                                          (gray dashed line = synthetic baseline ~45 deg)

Provenance:
  - Figures 1-3 are vector TikZ/pgfplots drawings in the LaTeX source. They were rendered
    from a tight clip of the final PDF at 4x zoom (~288 dpi). Figure captions are excluded,
    so these are pure artwork for separate figure upload.
  - Figures 4-5 are raster images embedded in the manuscript (real_eval.png, wsu_roc.png).
    The original embedded pixmaps were extracted losslessly at native resolution.

Full captions, the 180-deg-folded chance level (45 deg), and all scope warnings are in
paper_mdpi.tex / paper_mdpi.pdf. All reported values are in-sample where so labelled;
WSU results concern classical baselines only (the learned network was not evaluated there).
