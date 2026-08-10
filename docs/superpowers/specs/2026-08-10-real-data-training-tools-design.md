# Real-Data Training Tools Design

## Goal

Provide a small, repeatable training-host workflow that validates the real
Dataset v0.2 release and performs one real-data ModernTCN-MIL optimization step
before a full five-fold run is started.

## Scope

The tools use only a published Dataset v0.2 release. They do not create or use
synthetic training data, alter the release, publish model artifacts, or replace
the formal cross-validation runner.

The parent repository gains a single progress ledger. The Core submodule gains
one readiness module and two thin command-line scripts:

- `check_training_environment.py` reports Python/package/device status, validates
  the configured Dataset v0.2 release, and checks that the output location can
  be written.
- `smoke_train.py` loads real windows from one training fold, performs one
  ModernTCN-MIL forward pass, weighted BCE loss, backward pass, and optimizer
  update, then exits without writing formal experiment outputs.

The existing `train_baseline.py` remains the only formal training entry point.

## Interfaces

`mmdii.training.readiness.inspect_environment(config, release_override=None,
output_override=None)` returns a JSON-serializable report with `ok`, Python and
package versions, accelerator information, dataset counts/targets/folds,
output-path status, and an `errors` list. Missing optional training packages
are reported as failures rather than causing an import traceback.

`mmdii.training.readiness.run_real_data_smoke(config, release_override=None,
fold=0, batch_size=1, device_override=None)` returns a JSON-serializable
summary with the selected fold, real sample IDs, tensor shapes, resolved
device, finite loss/logit/gradient checks, and the post-update parameter delta.
It raises a clear runtime error when the train extra is unavailable and never
creates `oof_predictions.csv` or other formal run artifacts.

Both scripts accept `--config` and optional `--release-dir`. The environment
script also accepts `--output-dir` and `--json`; the smoke script accepts
`--fold`, `--batch-size`, and `--device`.

## Real-data smoke flow

```text
config + Dataset v0.2 release
    -> validate manifest and folds
    -> fit normalizer on training records only
    -> load one real batch from the selected training fold
    -> ModernTCN window encoder
    -> configured MIL head
    -> weighted BCE
    -> backward + one AdamW update
    -> finite-value and parameter-change report
```

The smoke path uses the same three formal targets, channel order, window
parameters, normalization rule, model configuration, and loss weighting as
the formal runner. It is intentionally one fold and one step: it verifies
environment and tensor plumbing, not model quality.

## Documentation

`docs/research-progress.md` records the data audit, annotation and Dataset v0.2
milestones, weak-supervision boundary, model route, implementation status,
known data limitations, branch status, and the exact next commands. It links
to detailed contracts and design documents instead of copying their full
contents.

## Testing

- Unit tests verify environment report shape, missing-package reporting, output
  path checks, CLI JSON behavior, and smoke argument validation.
- The smoke implementation has a train-extra-dependent test path; it is
  skipped when PyTorch is unavailable, matching the existing model tests.
- On a training host, the acceptance sequence is environment check, real-data
  smoke, then the existing five-fold `train_baseline.py` command.

## Non-goals

- No synthetic-data fallback.
- No automatic multi-experiment orchestrator.
- No change to Dataset v0.2 labels, folds, target codes, or release files.
- No claim that a successful smoke step means the model is scientifically
  validated.
