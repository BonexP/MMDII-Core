from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import savemat


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))
sys.path.insert(0, str(CORE_ROOT / "tests"))

from mmdii.data.dataset_publication import prepare_dataset
from mmdii.data.dataset_config import load_dataset_config
from mmdii.data.training_dataset import (
    FoldNormalizer,
    FullSignalSpec,
    DatasetIndex,
    WeldRecord,
    WeldWindowDataset,
    WindowSpec,
    collate_weld_batch,
    resample_signal,
    target_vector,
    window_signal,
)
from test_dataset_builder import signal_payload, write_annotation_release, write_config


class TrainingDatasetTests(unittest.TestCase):
    def test_dataset_index_keeps_weld_rows_and_excludes_pore_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            write_annotation_release(root / "annotation-release")
            for weld_id in range(7, 12):
                savemat(
                    source / f"2018-7-9-1.0-1.1-10000-200-{weld_id}~.mat",
                    signal_payload(),
                )
            config = load_dataset_config(
                write_config(root, contract_version="0.2.0")
            )
            release = prepare_dataset(config, release_id="training-index")

            index = DatasetIndex.from_release(
                release.release_directory,
                target_codes=("flash", "blur", "tunnel"),
            )

            self.assertEqual(len(index), 5)
            self.assertEqual(index.target_codes, ("flash", "blur", "tunnel"))
            self.assertEqual({record.fold for record in index.records}, {0, 1, 2, 3, 4})
            self.assertEqual(target_vector(("flash", "pore"), index.target_codes), (1.0, 0.0, 0.0))
            self.assertEqual(index.records[0].signal_path.name.endswith(".npz"), True)

    def test_window_signal_uses_seconds_and_pads_only_short_tail(self) -> None:
        time = np.arange(10, dtype=np.float64) / 2.0
        signal = np.vstack((time, time * 2.0))
        spec = WindowSpec(target_fs=2.0, window_seconds=2.0, stride_seconds=1.0)

        result = window_signal(signal, time, spec)

        self.assertEqual(result.windows.shape, (4, 2, 4))
        self.assertEqual(result.starts_seconds, (0.0, 1.0, 2.0, 3.0))
        self.assertTrue(result.window_mask.all())
        self.assertTrue(result.sample_mask.all())

        short_time = np.arange(2, dtype=np.float64) / 2.0
        short = window_signal(signal[:, :2], short_time, spec)
        self.assertEqual(short.windows.shape, (1, 2, 4))
        self.assertEqual(int(short.sample_mask[0].sum()), 2)
        self.assertTrue(short.window_mask[0])
        np.testing.assert_array_equal(short.windows[0, :, 2:], 0.0)

    def test_resample_signal_changes_sample_count_deterministically(self) -> None:
        source = np.arange(8, dtype=np.float64).reshape(2, 4)

        result = resample_signal(source, source_fs=4.0, target_fs=2.0)

        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(
            result,
            resample_signal(source, source_fs=4.0, target_fs=2.0),
        )

    def test_fold_normalizer_uses_only_supplied_training_records(self) -> None:
        normalizer = FoldNormalizer.from_arrays(
            [np.array([[0.0, 2.0], [10.0, 14.0]])]
        )
        transformed = normalizer.transform(np.array([[0.0, 2.0], [10.0, 14.0]]))

        np.testing.assert_allclose(transformed.mean(axis=1), 0.0)
        self.assertEqual(normalizer.means.shape, (2,))
        self.assertEqual(normalizer.stds.shape, (2,))

    def test_full_signal_spec_rescales_one_complete_weld(self) -> None:
        time = np.arange(8, dtype=np.float64) / 4.0
        signal = np.vstack((time, time * 2.0))

        result = FullSignalSpec(target_fs=4.0, output_samples=4).transform(signal, time)

        self.assertEqual(result.windows.shape, (1, 2, 4))
        self.assertEqual(result.starts_seconds, (0.0,))
        self.assertTrue(result.window_mask[0])
        self.assertTrue(result.sample_mask[0].all())

    def test_weld_dataset_and_collator_pad_only_window_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = []
            for index, sample_count in enumerate((10, 6)):
                sample_id = f"sample-{index}"
                time = np.arange(sample_count, dtype=np.float64) / 2.0
                signal_path = root / f"{sample_id}.npz"
                np.savez(
                    signal_path,
                    time=time,
                    af=time,
                    sf=time * 2.0,
                    axialf=time * 3.0,
                )
                records.append(
                    WeldRecord(
                        sample_id=sample_id,
                        weld_id=str(index),
                        image_group=f"image-{index}",
                        fold=index,
                        target=(float(index), 0.0, 0.0),
                        defect_codes=() if index == 0 else ("flash",),
                        is_normal=index == 0,
                        signal_path=signal_path,
                        metadata=(),
                    )
                )
            dataset_index = DatasetIndex(
                root,
                ("flash", "blur", "tunnel"),
                tuple(records),
            )
            normalizer = FoldNormalizer.fit(dataset_index, records[:1])
            dataset = WeldWindowDataset(
                dataset_index,
                folds={0, 1},
                spec=WindowSpec(2.0, 2.0, 1.0),
                normalizer=normalizer,
            )

            batch = collate_weld_batch([dataset[0], dataset[1]])

            self.assertEqual(batch["windows"].shape, (2, 4, 3, 4))
            self.assertEqual(batch["window_mask"].shape, (2, 4))
            self.assertEqual(batch["window_mask"][1].tolist(), [True, True, False, False])
            self.assertEqual(batch["sample_mask"].shape, (2, 4, 4))
            self.assertEqual(batch["targets"].shape, (2, 3))
            self.assertEqual(batch["sample_ids"], ("sample-0", "sample-1"))


if __name__ == "__main__":
    unittest.main()
