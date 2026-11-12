# Release manifest

- Upstream repository: `https://github.com/MCG-NJU/MOTIP.git`
- Upstream base commit: `ffc0e905ac196a603027eca8d18fb0dff48c8bcc`
- BCPR release date: 2026-11-12
- Default BCPR trainable parameters: 45,378
- BFT training split checked locally: 45 sequences, 8,482 frames
- CPU regression result: 6 passed
- Existing BFT epoch-21 checkpoint strict-load compatibility: passed

The compatible internal checkpoint used for the strict-load audit is not distributed.
Its SHA-256 is recorded only for provenance:

```text
4048976f71231fc809f4c4c0a092f8676bc96d1dc7218d6adb05eb66806cf12f
```

The source release excludes datasets, detector pretraining files, experiment outputs,
trained checkpoints, Python caches, and Git credentials.
