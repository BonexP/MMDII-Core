"""Tests for immutable dataset release publication."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from scipy.io import savemat


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))
sys.path.insert(0, str(CORE_ROOT / "tests"))

from mmdii.data.dataset_config import load_dataset_config
from mmdii.data.dataset_publication import (
    DatasetReleaseError,
    prepare_dataset,
    sha256_file,
    validate_dataset_release,
)
from test_dataset_builder import signal_payload, write_annotation_release, write_config


class DatasetPublicationTests(unittest.TestCase):
    def test_publishes_checksums_and_current_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            write_annotation_release(root / "annotation-release")
            savemat(
                source / "2018-7-9-1.0-1.1-10000-200-7~.mat",
                signal_payload(),
            )
            config = load_dataset_config(write_config(root))

            published = prepare_dataset(config, release_id="release-a")
            manifest = validate_dataset_release(published.release_directory)

            self.assertEqual(published.release_id, "release-a")
            self.assertEqual(manifest["release_id"], "release-a")
            self.assertEqual(manifest["counts"]["accepted"], 1)
            artifacts = manifest["artifacts"]
            self.assertNotIn("dataset_manifest.json", {item["path"] for item in artifacts})
            for artifact in artifacts:
                path = published.release_directory / artifact["path"]
                self.assertEqual(artifact["sha256"], sha256_file(path))
                self.assertEqual(artifact["size_bytes"], path.stat().st_size)
            pointer = json.loads(published.current_pointer.read_text(encoding="utf-8"))
            self.assertEqual(pointer["release_id"], "release-a")
            self.assertEqual(pointer["manifest"], "releases/release-a/dataset_manifest.json")

    def test_rejects_existing_release_and_tampered_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            write_annotation_release(root / "annotation-release")
            savemat(source / "2018-7-9-1.0-1.1-10000-200-7~.mat", signal_payload())
            config = load_dataset_config(write_config(root))
            published = prepare_dataset(config, release_id="release-a")

            with self.assertRaises(FileExistsError):
                prepare_dataset(config, release_id="release-a")
            with (published.release_directory / "samples.csv").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(DatasetReleaseError, "mismatch"):
                validate_dataset_release(published.release_directory)

    def test_failed_build_preserves_existing_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "source").mkdir()
            write_annotation_release(root / "annotation-release")
            config_path = write_config(root)
            config_text = config_path.read_text(encoding="utf-8").replace(
                'expected_annotation_release_id = "annotation-release"',
                'expected_annotation_release_id = "wrong-release"',
            )
            config_path.write_text(config_text, encoding="utf-8")
            config = load_dataset_config(config_path)
            config.destination.mkdir()
            pointer = config.destination / "current-dataset.json"
            pointer.write_text('{"release_id":"previous"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "release ID"):
                prepare_dataset(config, release_id="failed-release")

            self.assertEqual(pointer.read_text(encoding="utf-8"), '{"release_id":"previous"}\n')
            self.assertFalse((config.destination / "releases" / "failed-release").exists())
            self.assertFalse((config.destination / ".stage-failed-release").exists())


if __name__ == "__main__":
    unittest.main()
