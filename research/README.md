# Research Evidence Index

This directory stores the traceable evidence behind the final report and README. Professor-facing
English summaries are available in:

- [Methods and Validation](../docs/METHODS_AND_VALIDATION.md)
- [Experiments and Results](../docs/EXPERIMENTS_AND_RESULTS.md)
- [Repository and Evidence Guide](../docs/REPOSITORY_GUIDE.md)
- [Reproducibility Guide](../docs/REPRODUCIBILITY.md)
- [Model Evaluation and Graphics Catalogue](../docs/MODEL_EVALUATION_GRAPHICS.md)
- [Live Inference Demonstration](../docs/LIVE_DEMO.md)

## Primary evidence

| Area | Location |
|---|---|
| Fixed split | [`splits/practical_leakage_reduced_split_v1.csv`](splits/practical_leakage_reduced_split_v1.csv) |
| Split verification | [`splits/practical_leakage_reduced_split_v1_report.json`](splits/practical_leakage_reduced_split_v1_report.json) |
| Frozen-head ResNet18 | [`model_results/resnet18_26leaf_baseline_v1/`](model_results/resnet18_26leaf_baseline_v1/) |
| SmallCNN | [`model_results/smallcnn_26leaf_baseline_v1/`](model_results/smallcnn_26leaf_baseline_v1/) |
| Best fine-tuned ResNet18 | [`model_results/resnet18_lastblock_finetune_v1/`](model_results/resnet18_lastblock_finetune_v1/) |
| Error/calibration audit | [`model_results/resnet18_lastblock_finetune_v1/error_analysis_v1/`](model_results/resnet18_lastblock_finetune_v1/error_analysis_v1/) |
| Grad-CAM audit | [`model_results/resnet18_lastblock_finetune_v1/gradcam_v1/`](model_results/resnet18_lastblock_finetune_v1/gradcam_v1/) |
| GA versus random shared test | [`model_results/ga_random_shared_test_v1/`](model_results/ga_random_shared_test_v1/) |
| GA/random full-validation curves | [`model_results/ga_random_full_confirmation_v1/`](model_results/ga_random_full_confirmation_v1/) |
| GA/random proxy-search table | [`model_results/ga_resnet18_lastblock_search_v1/`](model_results/ga_resnet18_lastblock_search_v1/) |
| Final-report figures | [`final_report/figures/`](final_report/figures/) |

## Historical research logs

Files ending in `_TR.md` are the original Turkish chronological research records created while the
project was being developed. They preserve plans frozen before execution, failed hypotheses,
prospective audit boundaries, runtime incidents, hashes, and later verified results. They are kept
for provenance and are not the recommended entrypoint for an English-speaking reader.

The final English documentation does not hide unsuccessful candidate-generator versions. Those
records explain why a transparent practical known-family-grouped split was selected instead of
claiming a perfect unknown-near-duplicate retrieval rule.

## Evidence levels

- **Full-scope verified:** covers the stated complete dataset/run and passed integrity checks.
- **Pilot:** limited-scale method or infrastructure evidence; not project performance.
- **Development:** used to design a rule or choose a hypothesis.
- **Protected holdout/prospective audit:** opened only after its rule or selection was frozen.
- **Post-hoc audit:** analyzes a completed model but does not train or select it.

Smoke-test and pilot metrics must never be presented as final classifier performance.
