# NNGA Final Report Checkpoint

Last updated: 2026-11-25

## Deliverable

- Format: Microsoft Word DOCX
- Language: English
- Author line: MEHMET YUSUF OCAK - ERASMUS
- Intended reader: NNGA course professor at POLITEHNICA Bucharest
- Output: `deliverables/NNGA_Final_Report_Mehmet_Yusuf_Ocak_Erasmus.docx`

## Frozen report scope

1. Project objective and verified contribution
2. Multi Cancer dataset provenance and 8-parent/26-leaf taxonomy
3. Full 130,000-image audit
4. Exact and perceptual duplicate investigation
5. Practical known-family-grouped 70/15/15 split
6. Frozen-head ResNet18 baseline
7. From-scratch SmallCNN baseline
8. ResNet18 last-block fine-tuning ablation
9. Error, confidence and calibration audit
10. Grad-CAM explainability audit
11. Equal-budget genetic algorithm and random-search comparison
12. Shared-test paired evaluation and McNemar result
13. Limitations, reproducibility and final conclusions
14. Academic references and dataset-source appendix

## Verified headline results

- Dataset: 130,000 accepted images, 8 parent groups, 26 leaf labels
- Exact duplicates: 1,318 groups, 2,673 members, 1,355 redundant copies
- Split: 91,000 train / 19,500 validation / 19,500 test
- Split SHA-256: `999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`
- Frozen-head ResNet18 leaf accuracy/macro-F1: 0.913026 / 0.912455
- SmallCNN leaf accuracy/macro-F1: 0.955846 / 0.955916
- Last-block ResNet18 leaf accuracy/macro-F1: 0.996462 / 0.996462
- Best-model test errors: 69 / 19,500
- Grad-CAM: 37 frozen examples, 37/37 predictions reproduced
- GA proxy macro-F1: 0.952281; random proxy macro-F1: 0.949612
- Full-validation GA/random macro-F1: 0.996104 / 0.995640
- Shared-test GA/random leaf accuracy: 0.995795 / 0.995744
- Shared-test paired counts: 52 GA-only, 51 random-only, exact McNemar p = 1.0
- Sampled natural-reference recovery: 22/23 = 95.65%; this is not an exhaustive full-dataset recall claim

## Planned report figures

- Project workflow diagram
- Eight-group modality sample grid
- Dataset parent-group distribution
- DCT pHash process diagram
- Exact-duplicate example
- Candidate load by source group
- Family-grouped split diagram and split counts
- SmallCNN architecture diagram
- Model comparison chart
- Best-model training curve and confusion matrices
- Error-pair distribution
- Reliability diagram
- Grad-CAM examples
- Genetic algorithm workflow
- GA versus random-search comparison

## Progress

- [x] Evidence and document rules inspected
- [x] Report scope and headline metrics frozen
- [x] External primary references identified
- [x] Figure assets generated and audited (20 figures)
- [x] English manuscript authored
- [x] DOCX generated (29 pages)
- [x] DOCX rendered to PNG pages through Microsoft Word and Poppler
- [x] Every page visually inspected
- [x] Layout defects corrected and re-rendered
- [x] Word fields and table of contents updated and saved
- [x] Accessibility audit: 0 high-severity findings; all 20 figures have alternative descriptions
- [x] Heading and image-placement audits passed
- [x] Final DOCX committed and pushed

## Restart instruction

Read this file, inspect `research/final_report/figures`, `scripts/build_final_report_figures.py`,
and `scripts/build_final_report_docx.py`, then continue from the first unchecked item. Do not rerun
any model training. All reportable model results are already frozen.
