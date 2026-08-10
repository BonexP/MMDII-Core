from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))
sys.path.insert(0, str(CORE_ROOT / "scripts"))

from mmdii.training.cross_validation import ExperimentConfig
from mmdii.training.cross_validation import load_experiment_config
from mmdii.training import readiness
import check_training_environment
import smoke_train


REAL_CONFIG = CORE_ROOT / "configs" / "moderntcn_mil_v0_1.toml"
REAL_RELEASE = CORE_ROOT.parent.parent / "data" / "interim" / "releases" / "19db7f9cf4c04682bd33acbdd61ad3f4"
TORCH_AVAILABLE = __import__("importlib.util").util.find_spec("torch") is not None


class TrainingReadinessTests(unittest.TestCase):
    def test_environment_report_covers_packages_dataset_device_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(
                ExperimentConfig.for_test(),
                release_directory=root / "release",
                output_directory=root / "outputs" / "run",
            )
            index = SimpleNamespace(
                records=tuple(
                    SimpleNamespace(fold=fold, image_group=f"image-{fold}")
                    for fold in range(5)
                ),
                target_codes=config.target_codes,
            )
            package_report = {
                "scipy": {"available": True, "version": "1.11"},
                "sklearn": {"available": True, "version": "1.4"},
                "torch": {"available": True, "version": "2.2"},
            }
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: False,
                    device_count=lambda: 0,
                    get_device_name=lambda position: "unused",
                )
            )

            with patch.object(
                readiness, "_inspect_packages", return_value=(package_report, fake_torch, [])
            ):
                with patch.object(
                    readiness.DatasetIndex, "from_release", return_value=index
                ):
                    report = readiness.inspect_environment(config)

        self.assertTrue(report["ok"])
        self.assertEqual(report["dataset"]["sample_count"], 5)
        self.assertEqual(report["dataset"]["folds"], [0, 1, 2, 3, 4])
        self.assertEqual(report["dataset"]["image_group_count"], 5)
        self.assertEqual(report["dataset"]["target_codes"], list(config.target_codes))
        self.assertEqual(report["packages"], package_report)
        self.assertFalse(report["accelerator"]["cuda_available"])
        self.assertTrue(report["output"]["writable"])
        self.assertEqual(report["errors"], [])

    def test_environment_report_collects_missing_packages_and_dataset_error(self) -> None:
        config = ExperimentConfig.for_test()
        package_report = {
            "scipy": {"available": True, "version": "1.11"},
            "sklearn": {"available": False, "version": None},
            "torch": {"available": False, "version": None},
        }

        with patch.object(
            readiness,
            "_inspect_packages",
            return_value=(package_report, None, ["Missing package: sklearn", "Missing package: torch"]),
        ):
            with patch.object(
                readiness.DatasetIndex,
                "from_release",
                side_effect=RuntimeError("release is invalid"),
            ):
                with patch.object(
                    readiness, "_check_output_directory", side_effect=OSError("read only")
                ):
                    report = readiness.inspect_environment(config)

        self.assertFalse(report["ok"])
        self.assertIn("Missing package: torch", report["errors"])
        self.assertIn("Dataset check failed: release is invalid", report["errors"])
        self.assertIn("Output check failed: read only", report["errors"])
        self.assertFalse(report["dataset"]["valid"])
        self.assertFalse(report["output"]["writable"])

    def test_package_inspection_reports_broken_import_without_crashing(self) -> None:
        def load_package(name: str) -> SimpleNamespace:
            if name == "torch":
                raise OSError("CUDA DLL could not be loaded")
            return SimpleNamespace(__version__="test-version")

        with patch.object(readiness, "import_module", side_effect=load_package):
            packages, torch, errors = readiness._inspect_packages()

        self.assertIsNone(torch)
        self.assertFalse(packages["torch"]["available"])
        self.assertIn("torch import failed: CUDA DLL could not be loaded", errors)

    def test_environment_report_collects_cuda_probe_failure(self) -> None:
        config = ExperimentConfig.for_test()
        package_report = {
            "scipy": {"available": True, "version": "1.11"},
            "sklearn": {"available": True, "version": "1.4"},
            "torch": {"available": True, "version": "2.2"},
        }
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: (_ for _ in ()).throw(RuntimeError("driver error")),
            )
        )
        index = SimpleNamespace(
            records=tuple(
                SimpleNamespace(fold=fold, image_group=f"image-{fold}")
                for fold in range(5)
            ),
            target_codes=config.target_codes,
        )

        with patch.object(
            readiness, "_inspect_packages", return_value=(package_report, fake_torch, [])
        ):
            with patch.object(readiness.DatasetIndex, "from_release", return_value=index):
                with patch.object(readiness, "_check_output_directory"):
                    report = readiness.inspect_environment(config)

        self.assertFalse(report["ok"])
        self.assertIn("Accelerator check failed: driver error", report["errors"])

    def test_smoke_rejects_invalid_fold_and_batch_before_loading_training_extra(self) -> None:
        config = ExperimentConfig.for_test()

        with patch.object(readiness, "_require_torch") as require_torch:
            with self.assertRaisesRegex(ValueError, "fold"):
                readiness.run_real_data_smoke(config, fold=5)
            with self.assertRaisesRegex(ValueError, "batch_size"):
                readiness.run_real_data_smoke(config, batch_size=0)

        require_torch.assert_not_called()

    def test_smoke_uses_real_release_and_only_non_held_out_folds(self) -> None:
        config = ExperimentConfig.for_test()
        records = tuple(
            SimpleNamespace(
                fold=fold,
                target=(float(fold % 2), float((fold + 1) % 2), float(fold % 2)),
            )
            for fold in range(5)
        )
        index = SimpleNamespace(records=records, target_codes=config.target_codes)
        class FakeDataset:
            def __init__(self, rows: tuple[SimpleNamespace, ...]) -> None:
                self.records = rows

            def __len__(self) -> int:
                return len(self.records)

        dataset = FakeDataset(records[1:])
        expected = {
            "ok": True,
            "fold": 0,
            "sample_ids": ["mat-real"],
            "loss": 0.5,
        }

        with patch.object(readiness, "_require_torch", return_value=object()):
            with patch.object(
                readiness.DatasetIndex, "from_release", return_value=index
            ) as load_index:
                with patch.object(
                    readiness.FoldNormalizer, "fit", return_value=object()
                ) as fit_normalizer:
                    with patch.object(
                        readiness, "WeldWindowDataset", return_value=dataset
                    ) as build_dataset:
                        with patch.object(
                            readiness,
                            "compute_positive_class_weights",
                            return_value=np.ones(3),
                        ):
                            with patch.object(
                                readiness, "_execute_smoke_batch", return_value=expected
                            ) as execute:
                                result = readiness.run_real_data_smoke(config, fold=0)

        self.assertEqual(result, expected)
        load_index.assert_called_once_with(config.release_directory.resolve(), config.target_codes)
        fit_records = fit_normalizer.call_args.args[1]
        self.assertEqual({record.fold for record in fit_records}, {1, 2, 3, 4})
        self.assertEqual(build_dataset.call_args.args[1], {1, 2, 3, 4})
        self.assertEqual(execute.call_args.kwargs["held_out_fold"], 0)

    def test_smoke_requires_the_training_extra(self) -> None:
        with patch.object(readiness, "import_module", side_effect=ImportError("torch")):
            with self.assertRaisesRegex(RuntimeError, "train extra"):
                readiness.run_real_data_smoke(ExperimentConfig.for_test())

    def test_environment_cli_prints_json_and_returns_failure_status(self) -> None:
        report = {
            "ok": False,
            "python": {"version": "3.11.0", "supported": True},
            "packages": {},
            "accelerator": {"cuda_available": False, "device_count": 0, "devices": []},
            "dataset": {"valid": False},
            "output": {"writable": True},
            "errors": ["Missing package: torch"],
        }
        output = io.StringIO()

        with patch.object(
            check_training_environment, "inspect_environment", return_value=report
        ):
            with redirect_stdout(output):
                exit_code = check_training_environment.main(
                    [
                        "--config",
                        str(CORE_ROOT / "configs" / "moderntcn_mil_v0_1.toml"),
                        "--json",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertEqual(json.loads(output.getvalue()), report)

    def test_smoke_cli_prints_machine_readable_summary(self) -> None:
        summary = {"ok": True, "held_out_fold": 0, "loss": 0.4}
        output = io.StringIO()

        with patch.object(smoke_train, "run_real_data_smoke", return_value=summary):
            with redirect_stdout(output):
                exit_code = smoke_train.main(
                    ["--config", str(REAL_CONFIG), "--fold", "0", "--batch-size", "1"]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue()), summary)

    def test_smoke_cli_reports_runtime_failure_without_traceback(self) -> None:
        error = io.StringIO()

        with patch.object(
            smoke_train,
            "run_real_data_smoke",
            side_effect=RuntimeError("install the train extra"),
        ):
            with redirect_stderr(error):
                exit_code = smoke_train.main(["--config", str(REAL_CONFIG)])

        self.assertEqual(exit_code, 1)
        self.assertEqual(error.getvalue().strip(), "error: install the train extra")

    @unittest.skipUnless(
        TORCH_AVAILABLE and REAL_RELEASE.exists(),
        "requires PyTorch and the published real Dataset v0.2 release",
    )
    def test_real_data_smoke_completes_one_update(self) -> None:
        config = load_experiment_config(REAL_CONFIG)
        summary = readiness.run_real_data_smoke(config, fold=0, batch_size=1, device_override="cpu")

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["held_out_fold"], 0)
        self.assertGreater(summary["loss"], 0.0)
        self.assertGreater(summary["gradient_norm"], 0.0)
        self.assertGreater(summary["parameter_max_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
