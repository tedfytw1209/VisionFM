# Public-Benchmark Fundus RETFound-Protocol Design

## Goal

Make the VisionFM public-fundus runner use the same data-partition and model-selection protocol that RETFound actually executes, while preventing silent split-label mismatches and producing mathematically correct probability-based metrics.

## Scope

This change covers the public-fundus VisionFM fine-tuning runner and its test-only inference runner. It does not alter VisionFM's encoder, optimizer, augmentation, normalization, or the benchmark folders themselves.

## Data flow

The dataset root must contain `train`, `val`, and `test` ImageFolder splits. Training reads only `train`; model selection reads only `val`; held-out reporting reads only `test` after training has completed and the best validation checkpoint has been reloaded. No test score or test example may influence optimization, early stopping, or checkpoint choice.

Each split must expose exactly the same `class_to_idx` mapping as `train`. The runner must fail before training or inference if a split is missing, adds, renames, or reorders classes. It must also reject identical relative image paths appearing in more than one split, which catches a common directory-copy error. Subject-level or pixel-identical duplicate detection remains out of scope because the ImageFolder benchmark supplies neither subject identifiers nor a file-identity manifest.

## Evaluation

For every evaluated split, collect softmax probabilities, argmax predictions, and integer labels. Compute macro one-vs-rest AUROC and macro average precision from probabilities; compute accuracy, macro F1, precision, recall, Jaccard, Hamming loss, Cohen's kappa, and MCC from hard predictions.

To reproduce the supplied RETFound runner exactly, select the checkpoint by the RETFound evaluator's effective fallback score:

`(macro_f1 + macro_roc_auc + kappa) / 3`

This is intentionally not a new interpretation of `--eval_score auc`: RETFound's evaluator recognizes only `mcc` and `roc_auc`, so the provided script's value `auc` falls through to this composite. The runner will name this score explicitly so it is auditable.

Ties retain the first checkpoint, matching RETFound's strict `max_score < val_score` comparison. The training process reloads that checkpoint and evaluates `test` once; the shell script no longer launches a separate test process.

## Files and interfaces

- `finetune_visionfm_publicbench_fundus.py` will own split construction, split validation, probability-based metric calculation, RETFound-compatible score selection, and in-process final test reporting.
- `inference_visionfm_publicbench_fundus.py` will reuse the same split-validation and metric behavior for standalone checkpoint evaluation.
- `finetune_visionfm_publicbench_fundus.sh` will invoke only fine-tuning, because that command produces the final held-out test result itself.
- A small dependency-free protocol helper module and unit tests will make split validation and score calculation testable without CUDA, pretrained weights, or benchmark data.

## Error handling

The runner will stop with a readable error before model construction when a split is absent, has a different class mapping, or reuses a relative image path from another split. Metric calculation will report an actionable error when a split lacks the label support required for a macro metric rather than silently substituting hard labels or saving a misleading best checkpoint.

## Verification

Unit tests will establish the RETFound fallback selection score, probability-based AUROC/AP inputs, class-map mismatch rejection, duplicate-relative-path rejection, and the allowed disjoint-split case. The public scripts will also be syntax-checked.
