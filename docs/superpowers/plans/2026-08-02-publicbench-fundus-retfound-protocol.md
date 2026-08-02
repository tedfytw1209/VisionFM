# Public-Benchmark Fundus RETFound-Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align VisionFM public-fundus split validation, best-checkpoint selection, and held-out evaluation with the RETFound runner's executed protocol.

**Architecture:** Add a compact protocol module for class-map/path-split validation and the RETFound fallback selection score. The fine-tuner and standalone inference runner use it; fine-tuning reloads its selected validation checkpoint and evaluates the test split in-process.

**Tech Stack:** Python 3.8, PyTorch 1.11, torchvision ImageFolder, NumPy, scikit-learn, unittest.

## Global Constraints

- Train only from `train`; select only from `val`; test only once after reloading the selected checkpoint.
- Select with `(macro_f1 + macro_roc_auc + kappa) / 3`, matching RETFound's supplied `--eval_score auc` fallback behavior.
- Reject different class mappings and duplicate relative image paths across splits before model construction.
- Calculate AUROC and average precision from softmax probabilities, and hard-label metrics from argmax labels.
- Preserve VisionFM's existing model, optimizer, modality normalization, and augmentation.

---

### Task 1: Add and test protocol guards

**Files:**

- Create: `publicbench_fundus_protocol.py`
- Create: `tests/test_publicbench_fundus_protocol.py`

**Interfaces:**

- Produces: `validate_split_class_maps(train_map, split_maps) -> None`
- Produces: `validate_disjoint_relative_paths(split_paths) -> None`
- Produces: `retfound_fallback_score(metrics) -> float`

- [ ] **Step 1: Write the failing tests**

```python
from publicbench_fundus_protocol import (
    retfound_fallback_score,
    validate_disjoint_relative_paths,
    validate_split_class_maps,
)


def test_retfound_fallback_score_uses_f1_auc_and_kappa():
    score = retfound_fallback_score({'f1': 0.6, 'auc': 0.9, 'kappa': 0.3})
    assert score == 0.6


def test_class_map_mismatch_is_rejected():
    try:
        validate_split_class_maps({'healthy': 0, 'dr': 1}, {'val': {'dr': 0, 'healthy': 1}})
    except ValueError as error:
        assert 'class_to_idx' in str(error)
    else:
        raise AssertionError('expected mismatched class map to fail')


def test_duplicate_relative_path_across_splits_is_rejected():
    try:
        validate_disjoint_relative_paths({'train': ['dr/a.jpg'], 'val': ['dr/a.jpg'], 'test': ['dr/b.jpg']})
    except ValueError as error:
        assert 'dr/a.jpg' in str(error)
    else:
        raise AssertionError('expected duplicate split path to fail')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_publicbench_fundus_protocol -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'publicbench_fundus_protocol'`.

- [ ] **Step 3: Write minimal implementation**

```python
def validate_split_class_maps(train_map, split_maps):
    for split_name, class_to_idx in split_maps.items():
        if class_to_idx != train_map:
            raise ValueError(
                f"{split_name} class_to_idx does not match train: "
                f"expected {train_map}, got {class_to_idx}"
            )


def validate_disjoint_relative_paths(split_paths):
    seen = {}
    for split_name, paths in split_paths.items():
        for path in paths:
            if path in seen:
                raise ValueError(f"duplicate relative image path {path!r} in {seen[path]} and {split_name}")
            seen[path] = split_name


def retfound_fallback_score(metrics):
    return (metrics['f1'] + metrics['auc'] + metrics['kappa']) / 3.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_publicbench_fundus_protocol -v`

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add publicbench_fundus_protocol.py tests/test_publicbench_fundus_protocol.py
git commit -m "feat: validate public benchmark split protocol"
```

### Task 2: Make the fine-tuner select and test like RETFound

**Files:**

- Modify: `finetune_visionfm_publicbench_fundus.py`
- Modify: `finetune_visionfm_publicbench_fundus.sh`
- Test: `tests/test_publicbench_fundus_protocol.py`

**Interfaces:**

- Consumes: protocol module from Task 1.
- Produces: `checkpoint_best_finetune.pth` with `best_score`, `selection_metric`, and `class_to_idx`; `test_stats.json` and prediction arrays in the fine-tuning output directory.

- [ ] **Step 1: Extend the failing tests**

```python
def test_disjoint_paths_and_matching_maps_are_allowed():
    validate_split_class_maps({'healthy': 0, 'dr': 1}, {
        'val': {'healthy': 0, 'dr': 1},
        'test': {'healthy': 0, 'dr': 1},
    })
    validate_disjoint_relative_paths({
        'train': ['healthy/a.jpg'],
        'val': ['healthy/b.jpg'],
        'test': ['dr/c.jpg'],
    })


def test_retfound_score_requires_every_component():
    try:
        retfound_fallback_score({'f1': 0.6, 'auc': 0.9})
    except KeyError as error:
        assert error.args == ('kappa',)
    else:
        raise AssertionError('expected missing kappa to fail')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_publicbench_fundus_protocol -v`

Expected: FAIL because the allowed-case test has not yet been added to the suite.

- [ ] **Step 3: Integrate split validation and score selection**

```python
dataset_train = build_dataset('train', args)
dataset_val = build_dataset('val', args)
dataset_test = build_dataset('test', args)
validate_publicbench_splits(dataset_train, {'val': dataset_val, 'test': dataset_test})

val_stats.update(compute_metrics(preds, targets, output_labels, args.num_labels))
val_score = retfound_fallback_score(val_stats)
if val_score > best_score:
    torch.save({... , 'best_score': val_score,
                'selection_metric': 'retfound_fallback_f1_auc_kappa'}, checkpoint_path)
```

At the end of training, load `checkpoint_best_finetune.pth` into both model components, call the existing evaluation collector on `dataset_test`, compute the same probability-based metrics, and write scalar metrics and arrays to the fine-tuning output directory. Replace the shell script's separate inference call with a note that fine-tuning now evaluates the held-out test split itself.

- [ ] **Step 4: Run focused and syntax checks**

Run: `python -m unittest tests.test_publicbench_fundus_protocol -v; python -m py_compile finetune_visionfm_publicbench_fundus.py publicbench_fundus_protocol.py`

Expected: PASS with no syntax errors.

- [ ] **Step 5: Commit**

```bash
git add finetune_visionfm_publicbench_fundus.py finetune_visionfm_publicbench_fundus.sh tests/test_publicbench_fundus_protocol.py
git commit -m "fix: align VisionFM fundus evaluation with RETFound"
```

### Task 3: Keep standalone inference protocol-safe

**Files:**

- Modify: `inference_visionfm_publicbench_fundus.py`
- Test: `tests/test_publicbench_fundus_protocol.py`

**Interfaces:**

- Consumes: `class_to_idx` saved by Task 2 and protocol functions from Task 1.
- Produces: standalone test metrics calculated from probabilities only when the on-disk split mapping matches the checkpoint mapping.

- [ ] **Step 1: Add the failing test**

```python
def test_test_mapping_can_be_checked_against_checkpoint_mapping():
    try:
        validate_split_class_maps({'healthy': 0, 'dr': 1}, {'test': {'healthy': 0}})
    except ValueError as error:
        assert 'test class_to_idx' in str(error)
    else:
        raise AssertionError('expected checkpoint/test map mismatch to fail')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_publicbench_fundus_protocol -v`

Expected: FAIL because the new test is absent.

- [ ] **Step 3: Validate checkpoint mapping and reuse corrected metrics**

```python
checkpoint = torch.load(args.pretrained_weights, map_location='cpu')
expected_class_to_idx = checkpoint['class_to_idx']
dataset_test = build_dataset(args)
validate_split_class_maps(expected_class_to_idx, {'test': dataset_test.class_to_idx})
args.num_labels = len(expected_class_to_idx)
```

Replace local hard-label AUROC/AP calculation with the fine-tuner's shared probability-based metric function, retaining the existing standalone output files.

- [ ] **Step 4: Run focused and syntax checks**

Run: `python -m unittest tests.test_publicbench_fundus_protocol -v; python -m py_compile inference_visionfm_publicbench_fundus.py publicbench_fundus_protocol.py`

Expected: PASS with no syntax errors.

- [ ] **Step 5: Commit**

```bash
git add inference_visionfm_publicbench_fundus.py tests/test_publicbench_fundus_protocol.py
git commit -m "fix: validate standalone fundus inference split"
```

### Task 4: Final verification

**Files:**

- Verify: `publicbench_fundus_protocol.py`
- Verify: `tests/test_publicbench_fundus_protocol.py`
- Verify: `finetune_visionfm_publicbench_fundus.py`
- Verify: `inference_visionfm_publicbench_fundus.py`
- Verify: `finetune_visionfm_publicbench_fundus.sh`

- [ ] **Step 1: Run full protocol tests**

Run: `python -m unittest discover -s tests -v`

Expected: PASS with zero failures.

- [ ] **Step 2: Compile all modified Python entry points**

Run: `python -m py_compile publicbench_fundus_protocol.py finetune_visionfm_publicbench_fundus.py inference_visionfm_publicbench_fundus.py`

Expected: exit 0.

- [ ] **Step 3: Review the diff**

Run: `git diff --check; git status --short`

Expected: no whitespace errors and only the intended public-benchmark protocol files changed.

