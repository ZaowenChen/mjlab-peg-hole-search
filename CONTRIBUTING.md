# Contributing

Contributions should preserve the repository's main compatibility contracts:

- do not expose hole ground truth to the actor;
- do not change control, reward or success semantics as part of a structural
  refactor;
- write every run to a new output directory;
- retain effective configuration and provenance with results;
- add a focused test for behavior or serialization changes.

For a source-only checkout, run the portable checks used by GitHub Actions:

```bash
python -m compileall -q src scripts tests
PYTHONPATH=src python -m unittest discover -s tests -p 'test_configuration.py' -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_frozen_source.py' -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_repository_layout.py' -v
```

With the compatible project runtime installed, run all CPU tests. Historical
workspace validation additionally requires the original local experiment data
and frozen source archives; these are not included in a normal clone:

```bash
bash scripts/run_local.sh python -m unittest discover -s tests -v
bash scripts/run_local.sh python scripts/validate_workspace.py
```

Use `--compare-working-tree` only when checking that development source still
equals the original formal baseline. New development normally differs from that
baseline; historical validation checks matching snapshots instead.

GPU or physics changes should state the runtime, device, seeds, exact cases and
known repeatability limits. Large checkpoints, caches, trajectories and logs
must not be committed.
