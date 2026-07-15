# MMDII-Core

MMDII-Core is the standalone Python code repository for MMDII. It owns reusable
MAT readers, dataset-preparation logic, multimodal models, training workflows,
and evaluation code.

It is used by the parent MMDII integration repository as a Git submodule. The
parent owns raw-data locations, labels, project documentation, and workflow
orchestration. Core code receives those locations through explicit arguments or
configuration paths and must not hard-code parent-repository paths.

## Layout

- `src/mmdii/data/`: source configuration and future preparation code.
- `src/mmdii/models/`: model definitions.
- `src/mmdii/training/`: training workflows.
- `src/mmdii/evaluation/`: evaluation workflows.
- `tests/`: standalone package tests.

Run the package checks from this directory:

```powershell
python -m unittest discover -s tests -v
```

## MAT Header Inspection

Install the Core package into the active Python environment:

```powershell
python -m pip install -e .
```

`mmdii.data.inspect_mat_directory` reads MAT variable names, MATLAB classes,
and shapes through `scipy.io.whosmat`. It does not load signal arrays and does
not require every file to contain the same variables. Unreadable files are
reported individually while the remaining files are inspected.
