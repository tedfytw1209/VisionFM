"""Split validation and checkpoint-selection helpers for public fundus benchmarks."""

import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


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


def validate_imagefolder_splits(train_dataset, split_datasets):
    """Validate ImageFolder class labels and paths across benchmark splits."""
    validate_split_class_maps(
        train_dataset.class_to_idx,
        {name: dataset.class_to_idx for name, dataset in split_datasets.items()},
    )
    split_paths = {
        'train': _imagefolder_relative_paths(train_dataset),
        **{
            name: _imagefolder_relative_paths(dataset)
            for name, dataset in split_datasets.items()
        },
    }
    validate_disjoint_relative_paths(split_paths)


def _imagefolder_relative_paths(dataset):
    return [
        os.path.relpath(path, dataset.root).replace(os.sep, '/')
        for path, _ in dataset.samples
    ]


def compute_classification_metrics(probabilities, targets, num_classes):
    """Compute probability and hard-label metrics with their proper inputs."""
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets).reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[1] != num_classes:
        raise ValueError(
            'probabilities must have shape (n_samples, num_classes); '
            f'got {probabilities.shape} for {num_classes} classes'
        )
    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError(
            'probabilities and targets must contain the same number of samples; '
            f'got {probabilities.shape[0]} and {targets.shape[0]}'
        )
    if np.any(targets < 0) or np.any(targets >= num_classes):
        raise ValueError('targets contain a class index outside the configured class range')

    target_one_hot = np.eye(num_classes, dtype=int)[targets]
    predicted_labels = probabilities.argmax(axis=1)
    predicted_one_hot = np.eye(num_classes, dtype=int)[predicted_labels]
    try:
        auc = roc_auc_score(
            target_one_hot, probabilities, average='macro', multi_class='ovr'
        )
        aupr = average_precision_score(target_one_hot, probabilities, average='macro')
    except ValueError as error:
        raise ValueError(
            'cannot calculate macro probability metrics; every class must have '
            'both positive and negative examples in this split'
        ) from error

    return {
        'auc': auc,
        'aupr': aupr,
        'mcc': matthews_corrcoef(targets, predicted_labels),
        'accuracy': accuracy_score(targets, predicted_labels),
        'hamming': hamming_loss(target_one_hot, predicted_one_hot),
        'jaccard': jaccard_score(
            target_one_hot, predicted_one_hot, average='macro', zero_division=0
        ),
        'average_precision': aupr,
        'kappa': cohen_kappa_score(targets, predicted_labels),
        'f1': f1_score(target_one_hot, predicted_one_hot, zero_division=0, average='macro'),
        'roc_auc': auc,
        'precision': precision_score(
            target_one_hot, predicted_one_hot, zero_division=0, average='macro'
        ),
        'recall': recall_score(
            target_one_hot, predicted_one_hot, zero_division=0, average='macro'
        ),
    }


def retfound_fallback_score(metrics):
    """Match RETFound's fallback selection score for ``--eval_score auc``."""
    return (metrics['f1'] + metrics['auc'] + metrics['kappa']) / 3.0
