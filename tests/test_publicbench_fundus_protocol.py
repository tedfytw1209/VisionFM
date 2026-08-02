import unittest

import numpy as np

from publicbench_fundus_protocol import (
    compute_classification_metrics,
    retfound_fallback_score,
    validate_disjoint_relative_paths,
    validate_imagefolder_splits,
    validate_split_class_maps,
)


class PublicbenchFundusProtocolTests(unittest.TestCase):
    def test_retfound_fallback_score_uses_f1_auc_and_kappa(self):
        score = retfound_fallback_score({'f1': 0.6, 'auc': 0.9, 'kappa': 0.3})

        self.assertEqual(score, 0.6)

    def test_class_map_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'class_to_idx'):
            validate_split_class_maps(
                {'healthy': 0, 'dr': 1},
                {'val': {'dr': 0, 'healthy': 1}},
            )

    def test_duplicate_relative_path_across_splits_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'dr/a.jpg'):
            validate_disjoint_relative_paths({
                'train': ['dr/a.jpg'],
                'val': ['dr/a.jpg'],
                'test': ['dr/b.jpg'],
            })

    def test_imagefolder_splits_reject_duplicate_relative_filename(self):
        class Dataset:
            def __init__(self, root, samples):
                self.root = root
                self.class_to_idx = {'healthy': 0, 'dr': 1}
                self.samples = samples

        train = Dataset('benchmark/train', [('benchmark/train/dr/a.jpg', 1)])
        val = Dataset('benchmark/val', [('benchmark/val/healthy/b.jpg', 0)])
        test = Dataset('benchmark/test', [('benchmark/test/dr/a.jpg', 1)])

        with self.assertRaisesRegex(ValueError, 'dr/a.jpg'):
            validate_imagefolder_splits(train, {'val': val, 'test': test})

    def test_auc_and_average_precision_use_probabilities(self):
        probabilities = np.array([
            [0.9, 0.1],
            [0.7, 0.3],
            [0.4, 0.6],
            [0.1, 0.9],
        ])
        targets = np.array([0, 1, 0, 1])

        metrics = compute_classification_metrics(probabilities, targets, num_classes=2)

        self.assertAlmostEqual(metrics['auc'], 0.75)
        self.assertAlmostEqual(metrics['roc_auc'], metrics['auc'])
        self.assertAlmostEqual(metrics['aupr'], 5.0 / 6.0)
        self.assertAlmostEqual(metrics['average_precision'], metrics['aupr'])


if __name__ == '__main__':
    unittest.main()
