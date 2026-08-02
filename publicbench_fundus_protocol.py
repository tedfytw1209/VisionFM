"""Split validation and checkpoint-selection helpers for public fundus benchmarks."""


def validate_split_class_maps(train_map, split_maps):
    """Require every non-training split to preserve train's ImageFolder labels."""
    for split_name, class_to_idx in split_maps.items():
        if class_to_idx != train_map:
            raise ValueError(
                f'{split_name} class_to_idx does not match train: '
                f'expected {train_map}, got {class_to_idx}'
            )


def validate_disjoint_relative_paths(split_paths):
    """Reject the same class-relative filename in more than one split."""
    seen = {}
    for split_name, paths in split_paths.items():
        for path in paths:
            if path in seen:
                raise ValueError(
                    f'duplicate relative image path {path!r} in '
                    f'{seen[path]} and {split_name}'
                )
            seen[path] = split_name


def retfound_fallback_score(metrics):
    """Match RETFound's fallback selection score for ``--eval_score auc``."""
    return (metrics['f1'] + metrics['auc'] + metrics['kappa']) / 3.0
