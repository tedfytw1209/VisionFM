import unittest


from publicbench_fundus_protocol import (
    retfound_fallback_score,
    validate_disjoint_relative_paths,
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


if __name__ == '__main__':
    unittest.main()
