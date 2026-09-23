# Contributing

Contributions should preserve the repository's main compatibility contracts:

- do not expose hole ground truth to the actor;
- do not change control, reward or success semantics as part of a structural
  refactor;
- write every run to a new output directory;
- retain effective configuration and provenance with results;
- add a focused test for behavior or serialization changes.

Before opening a pull request, run:

```bash
bash scripts/run_local.sh python -m unittest discover -s tests -v
bash scripts/run_local.sh python scripts/validate_workspace.py
```

GPU or physics changes should state the runtime, device, seeds, exact cases and
known repeatability limits. Large checkpoints, caches, trajectories and logs
must not be committed.
