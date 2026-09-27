# Project3 status — finalized

Updated 2026-09-27 02:02 UTC.

The requested implementation and validation targets are complete for the documented
single-machine Docker and controlled-failure scope.

- Three-worker CIFAR-10 benchmark: **2.83× speedup, 94.3% efficiency** at global batch 16,384.
- Heartbeat detection: **maximum 3.41s** across nine timed trials.
- Fault campaign: **1,000/1,000 completed after recovery**, including 200 injected faults.
- Tests: **24/24 local and 24/24 Docker**.
- Container crash recovery, rejoining and coordinator checkpoint resume: passed.

Read [FINAL_REPORT.md](FINAL_REPORT.md) for test conditions, limitations and evidence,
[README.md](README.md) to run it.
Earlier status is retained in `results/status-before-final-validation.md`.
