from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.training.second_stage import (  # noqa: E402
    FEATURE_SOURCES,
    assemble_features,
    harvest_out_of_sample_representations,
    inner_partitions,
    project_embedding_features,
    run_second_stage_fold,
    statistical_feature_matrix,
)

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None

CODES = ("flash", "blur", "tunnel")


@dataclass(frozen=True)
class FakeRecord:
    sample_id: str
    fold: int
    target: tuple[float, ...]


@dataclass(frozen=True)
class FakeSecondStage:
    feature_source: str = "embedding"
    inner_folds: int = 4
    representation_mode: str = "full_signal"
    encoder_epochs: int = 1
    enabled: bool = True


@dataclass(frozen=True)
class FakeConfig:
    second_stage: FakeSecondStage
    target_codes: tuple[str, ...] = CODES
    seed: int = 7


class FakeIndex:
    def __init__(self, channels: int = 3, samples: int = 64) -> None:
        self._signal = np.arange(channels * samples, dtype=np.float64).reshape(channels, samples)

    def load_signal(self, record):
        return self._signal, np.arange(self._signal.shape[1], dtype=np.float64)


def _records(counts=(3, 3, 3, 3, 3)):
    records = []
    for fold, count in enumerate(counts):
        for position in range(count):
            records.append(
                FakeRecord(
                    sample_id=f"f{fold}-w{position}",
                    fold=fold,
                    target=(1.0, float(position % 2), 0.0),
                )
            )
    return tuple(records)


class InnerPartitionTests(unittest.TestCase):
    def test_partitions_cover_every_training_fold_and_exclude_the_outer_fold(self) -> None:
        training = (1, 2, 3, 4)
        for count in (2, 3, 4):
            partitions = inner_partitions(training, count)
            flattened = [fold for group in partitions for fold in group]
            self.assertEqual(sorted(flattened), list(training))
            self.assertEqual(len(partitions), count)
            self.assertNotIn(0, flattened)

    def test_rejects_a_single_inner_fold(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            inner_partitions((1, 2, 3, 4), 1)

    def test_rejects_more_inner_folds_than_training_folds(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            inner_partitions((1, 2), 3)


class HarvestTests(unittest.TestCase):
    def _run_deep(self, calls):
        def run_deep(train_records, valid_records, *, harvest_embeddings=False):
            calls.append((tuple(r.sample_id for r in train_records), tuple(r.sample_id for r in valid_records)))
            representations = np.stack(
                [np.full((3, 4), float(r.fold), dtype=np.float64) for r in valid_records]
            )
            return np.zeros((len(valid_records), 3)), {
                "weld_representations": representations,
                "sample_ids": tuple(r.sample_id for r in valid_records),
            }

        return run_deep

    def test_harvest_covers_every_training_weld_once(self) -> None:
        records = _records()
        training = tuple(r for r in records if r.fold != 0)
        calls: list = []

        harvested = harvest_out_of_sample_representations(
            training,
            fold=0,
            partitions=inner_partitions((1, 2, 3, 4), 4),
            run_deep=self._run_deep(calls),
        )

        self.assertEqual(harvested.shape, (len(training), 3, 4))
        self.assertEqual(len(calls), 4)
        # In record order: the first three training welds belong to fold 1.
        self.assertTrue(np.all(harvested[:3] == 1.0))

    def test_inner_validation_sets_never_overlap_and_never_include_the_outer_fold(self) -> None:
        records = _records()
        training = tuple(r for r in records if r.fold != 0)
        calls: list = []

        harvest_out_of_sample_representations(
            training,
            fold=0,
            partitions=inner_partitions((1, 2, 3, 4), 4),
            run_deep=self._run_deep(calls),
        )

        seen: list[str] = []
        for train_ids, valid_ids in calls:
            self.assertFalse(set(train_ids) & set(valid_ids))
            self.assertFalse(any(sample_id.startswith("f0-") for sample_id in train_ids))
            self.assertFalse(any(sample_id.startswith("f0-") for sample_id in valid_ids))
            seen.extend(valid_ids)
        self.assertEqual(len(seen), len(training))
        self.assertEqual(len(set(seen)), len(training))

    def test_rejects_a_partition_that_contains_the_outer_fold(self) -> None:
        records = _records()
        training = tuple(r for r in records if r.fold != 0)

        with self.assertRaisesRegex(ValueError, "outer held-out fold"):
            harvest_out_of_sample_representations(
                training,
                fold=0,
                partitions=((0, 1), (2, 3, 4)),
                run_deep=self._run_deep([]),
            )

    def test_rejects_a_partition_outside_the_training_folds(self) -> None:
        records = _records(counts=(3, 3, 3, 3, 3))
        training = tuple(r for r in records if r.fold != 0)

        with self.assertRaisesRegex(ValueError, "not a subset"):
            harvest_out_of_sample_representations(
                training,
                fold=1,
                partitions=((0, 2), (3, 4)),
                run_deep=self._run_deep([]),
            )

    def test_rejects_an_incomplete_harvest(self) -> None:
        records = _records()
        training = tuple(r for r in records if r.fold != 0)

        def run_deep(train_records, valid_records, *, harvest_embeddings=False):
            # Drop the last weld to simulate a missing representation.
            trimmed = valid_records[:-1]
            return np.zeros((len(trimmed), 3)), {
                "weld_representations": np.zeros((len(trimmed), 3, 4)),
                "sample_ids": tuple(r.sample_id for r in trimmed),
            }

        with self.assertRaisesRegex(ValueError, "match the inner holdout size|No out-of-sample"):
            harvest_out_of_sample_representations(
                training,
                fold=0,
                partitions=inner_partitions((1, 2, 3, 4), 4),
                run_deep=run_deep,
            )


class AssembleFeatureTests(unittest.TestCase):
    def test_stats_arm_has_eighteen_columns(self) -> None:
        features = assemble_features(
            np.zeros((5, 18)), None, "stats", representation_shared=True
        )
        self.assertEqual(features.shape, (5, 18))

    def test_shared_representation_keeps_one_slice(self) -> None:
        representations = np.random.default_rng(0).random((5, 3, 4))

        features = assemble_features(None, representations, "embedding", representation_shared=True)

        self.assertEqual(features.shape, (5, 4))
        np.testing.assert_allclose(features, representations[:, 0, :])

    def test_per_target_representation_is_flattened(self) -> None:
        representations = np.random.default_rng(0).random((5, 3, 4))

        features = assemble_features(None, representations, "embedding", representation_shared=False)

        self.assertEqual(features.shape, (5, 12))

    def test_both_arm_concatenates_statistics_and_representation(self) -> None:
        statistics = np.zeros((5, 18))
        representations = np.zeros((5, 3, 4))

        features = assemble_features(
            statistics, representations, "both", representation_shared=True
        )

        self.assertEqual(features.shape, (5, 22))

    def test_rejects_an_unknown_feature_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature_source"):
            assemble_features(np.zeros((2, 18)), None, "power_spectrum", representation_shared=True)

    def test_rejects_a_missing_block(self) -> None:
        with self.assertRaisesRegex(ValueError, "Statistics are required"):
            assemble_features(None, np.zeros((2, 3, 4)), "both", representation_shared=True)
        with self.assertRaisesRegex(ValueError, "Representations are required"):
            assemble_features(None, None, "embedding", representation_shared=True)


class StatisticalMatrixTests(unittest.TestCase):
    def test_shape_matches_records_and_channels(self) -> None:
        records = _records(counts=(2, 2, 2, 2, 2))
        matrix = statistical_feature_matrix(FakeIndex(), records)
        self.assertEqual(matrix.shape, (len(records), 18))


class SecondStageFoldTests(unittest.TestCase):
    def _harness(self, feature_source):
        records = _records(counts=(4, 4, 4, 4, 4))
        train_records = tuple(r for r in records if r.fold != 0)
        valid_records = tuple(r for r in records if r.fold == 0)
        config = FakeConfig(second_stage=FakeSecondStage(feature_source=feature_source))
        deep_calls: list = []
        tree_inputs: dict[str, np.ndarray] = {}

        def run_deep(train_subset, valid_subset, *, harvest_embeddings=False):
            deep_calls.append(
                (tuple(r.sample_id for r in train_subset), tuple(r.sample_id for r in valid_subset))
            )
            representations = np.stack(
                [np.full((3, 4), float(r.fold), dtype=np.float64) for r in valid_subset]
            )
            return np.full((len(valid_subset), 3), 0.25), {
                "weld_representations": representations,
                "sample_ids": tuple(r.sample_id for r in valid_subset),
            }

        def fit_tree(train_features, train_targets, valid_features):
            tree_inputs["train"] = np.asarray(train_features)
            tree_inputs["valid"] = np.asarray(valid_features)
            tree_inputs["targets"] = np.asarray(train_targets)
            return np.full((len(valid_features), 3), 0.75)

        probabilities, info = run_second_stage_fold(
            index=FakeIndex(),
            train_records=train_records,
            valid_records=valid_records,
            config=config,
            fold=0,
            run_deep=run_deep,
            fit_tree=fit_tree,
            representation_shared=True,
        )
        return probabilities, info, deep_calls, tree_inputs, train_records, valid_records

    def test_embedding_arm_trains_one_inner_encoder_per_partition_plus_one_outer(self) -> None:
        _, info, deep_calls, _, _, _ = self._harness("embedding")

        self.assertEqual(len(deep_calls), 5)  # four inner folds + the outer model
        self.assertEqual(info["second_stage"]["inner_partitions"], [[1], [2], [3], [4]])
        self.assertEqual(info["second_stage"]["feature_count"], 4)
        self.assertEqual(info["second_stage"]["representation_dim"], 4)
        self.assertIsNotNone(info["second_stage"]["deep_macro_pr_auc"])

    def test_outer_model_trains_on_all_training_welds(self) -> None:
        _, _, deep_calls, _, train_records, valid_records = self._harness("embedding")

        outer_train, outer_valid = deep_calls[-1]
        self.assertEqual(set(outer_train), {r.sample_id for r in train_records})
        self.assertEqual(set(outer_valid), {r.sample_id for r in valid_records})

    def test_tree_is_fitted_on_the_out_of_sample_matrix(self) -> None:
        _, _, _, tree_inputs, train_records, _ = self._harness("embedding")

        # The fake encoder stamps the held-out inner fold id onto each vector, so
        # the first column of the tree's training matrix must reproduce the fold
        # of every training weld in record order.  That is only true when the tree
        # received the harvested matrix rather than in-sample embeddings.
        expected = np.asarray([float(r.fold) for r in train_records])
        self.assertEqual(tree_inputs["train"].shape[0], len(train_records))
        np.testing.assert_allclose(tree_inputs["train"][:, 0], expected)

    def test_both_arm_concatenates_statistics(self) -> None:
        _, info, _, tree_inputs, _, _ = self._harness("both")

        self.assertEqual(info["second_stage"]["feature_count"], 22)
        self.assertEqual(tree_inputs["train"].shape[1], 22)

    def test_stats_arm_bypasses_the_encoder_entirely(self) -> None:
        probabilities, info, deep_calls, tree_inputs, _, valid_records = self._harness("stats")

        self.assertEqual(deep_calls, [])
        self.assertEqual(info["second_stage"]["inner_folds"], 0)
        self.assertEqual(tree_inputs["train"].shape[1], 18)
        self.assertEqual(probabilities.shape, (len(valid_records), 3))

    def test_harvested_representations_follow_validation_record_order(self) -> None:
        def run_deep(train_subset, valid_subset, *, harvest_embeddings=False):
            # Deliberately shuffle the harvested order.
            shuffled = tuple(reversed(valid_subset))
            return np.zeros((len(valid_subset), 3)), {
                "weld_representations": np.zeros((len(valid_subset), 3, 4)),
                "sample_ids": tuple(r.sample_id for r in shuffled),
            }

        def fit_tree(train_features, train_targets, valid_features):
            return np.zeros((len(valid_features), 3))

        records = _records(counts=(4, 4, 4, 4, 4))
        with self.assertRaisesRegex(ValueError, "validation record order"):
            run_second_stage_fold(
                index=FakeIndex(),
                train_records=tuple(r for r in records if r.fold != 0),
                valid_records=tuple(r for r in records if r.fold == 0),
                config=FakeConfig(second_stage=FakeSecondStage(feature_source="embedding")),
                fold=0,
                run_deep=run_deep,
                fit_tree=fit_tree,
                representation_shared=True,
            )

    def test_feature_sources_are_the_documented_three(self) -> None:
        self.assertEqual(FEATURE_SOURCES, ("stats", "embedding", "both"))

    def test_embedding_projection_none_preserves_dimensions(self) -> None:
        train = np.arange(20, dtype=np.float64).reshape(5, 4)
        valid = np.arange(8, dtype=np.float64).reshape(2, 4)
        projected_train, projected_valid, info = project_embedding_features(
            train,
            valid,
            projection="none",
            components=8,
            representation_shared=True,
        )
        np.testing.assert_array_equal(projected_train, train)
        np.testing.assert_array_equal(projected_valid, valid)
        self.assertEqual(info["output_dimension"], 4)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "requires scikit-learn")
    def test_pca_projection_fits_on_training_matrix_only(self) -> None:
        train = np.arange(40, dtype=np.float64).reshape(10, 4)
        valid = np.full((2, 4), 100.0)
        projected_train, projected_valid, info = project_embedding_features(
            train,
            valid,
            projection="pca",
            components=2,
            representation_shared=True,
        )
        self.assertEqual(projected_train.shape, (10, 2))
        self.assertEqual(projected_valid.shape, (2, 2))
        self.assertEqual(info["input_dimension"], 4)
        self.assertEqual(info["output_dimension"], 2)


if __name__ == "__main__":
    unittest.main()
