# Project3 — verified status

Updated 2026-09-26 20:39 UTC.

The six previously listed implementation gaps have been addressed and validated
for the documented CPU training and fail-stop recovery scope. Completion is
reported through evidence below rather than an arbitrary percentage.

## Verification summary

| Requirement | Result | Evidence |
|---|---|---|
| Worker registration and coordinator commands | Implemented and tested | `coordinator/coordinator.py`, `client/train_cluster.py` |
| Four-worker synchronized Docker training | Passed; full CIFAR-10 run completed | `results/cifar10_training.json`, `results/cifar10_consistency.json` |
| Global abort and coordinated recovery | Passed transient faults, partial commits, worker kill/rejoin, checkpoint restart | `results/fault_validation.json`, `results/docker_recovery.json` |
| CIFAR-10 pipeline | Official checksum verified; 50,000 train / 10,000 test split | `worker/prepare_cifar.py`, `results/cifar10_evaluation.json` |
| 1,000-operation fault campaign | 1,000 passed, 200 faults injected, 200 recoveries | `results/fault_validation.json` |
| Real speedup and efficiency measurements | 1/2/4 Docker workers, both datasets, three repeats | `results/docker_benchmark_digits.json`, `results/docker_benchmark_cifar10.json` |
| Documentation | Setup, protocol, reproducibility, and limitations written | `README.md` |
| Automated tests | 19 passed locally; 19 passed inside Linux Docker | `results/tests.xml`, `results/validation_summary.json` |

## Full CIFAR-10 run

- Four Docker workers, 782 committed steps, global batch 64.
- 50,048 sample presentations: one complete pass over the 50,000-image
  training set plus deterministic wraparound of the final batch.
- Held-out test accuracy: **45.46%**, over 10,000 images.
- Held-out cross-entropy: 1.5634.
- First/last training minibatch losses: 2.2927 / 1.6968.
- Recovery events during this healthy run: 0.
- Final worker parameters agree within numerical tolerance: True.
- Architecture: 3072 → 128 → 10 NumPy MLP, ReLU, SGD; CPU only.

This validates end-to-end image training. Accuracy tuning and competitive image
classification are separate work; this run is a one-epoch systems demonstration.

## Fault validation

The 1,000-step seeded campaign compared every committed model with serial SGD on
identical global batches. Maximum absolute parameter error was
8.88e-16. Its 200 injected faults comprise 40 each of:
prepare errors, mid-AllReduce errors, pre-commit errors, lost post-commit responses,
and stopped worker gRPC services. These workers use actual loopback gRPC in one
Python process. They are not 1,000 Docker process crashes.

The separate Docker experiment killed worker 3 during a 100-step run. All 100 steps
completed with workers 0–2 and 1 recovery event. Worker 3 then rejoined;
all four completed another two steps. Restarting the coordinator and workers
resumed from disk and completed two further steps (105 total committed steps).

Tests also cover permanent-failure rollback, stale request rejection, idempotent
commit, unequal shard weighting, checkpoint reload, and failed checkpoint writes.

## Measured scaling

Fixed global batch 64, 30 measured steps after two warmup steps, median of three
fresh Docker runs. One CPU per container and one BLAS thread. End-to-end timing
includes coordination and checkpoint writes; startup and evaluation are excluded.
All containers run on one host. This is strong scaling, not a multi-machine result.

| Dataset | Workers | Median seconds | Speedup | Efficiency |
|---|---:|---:|---:|---:|
| digits | 1 | 0.238 | 1.000× | 100.0% |
| digits | 2 | 0.408 | 0.583× | 29.1% |
| digits | 4 | 1.472 | 0.162× | 4.0% |
| cifar10 | 1 | 2.710 | 1.000× | 100.0% |
| cifar10 | 2 | 3.827 | 0.708× | 35.4% |
| cifar10 | 4 | 6.430 | 0.421× | 10.5% |

**No positive multi-worker speedup was observed.** Communication, synchronization,
and checkpoint overhead outweigh the compute saved for these tested workloads.
The measurements are useful performance evidence; they do not support an
"accelerated training" or "near-linear scaling" résumé claim.

![Measured Docker scaling](results/docker_scaling.png)

The old `results/scaling.json`, `speedup.png`, and `efficiency.png` describe a
single-process simulation and must not be used as real distributed measurements.

## Supported résumé wording

Built a NumPy/gRPC distributed training system with four Docker workers,
Ring-AllReduce, coordinator-driven training, and checkpoint-based recovery;
validated 1,000 optimizer operations with 200 injected faults against serial SGD.
Implemented CIFAR-10 ingestion and held-out evaluation, and measured strong
scaling across 1, 2, and 4 Docker workers.

## Operational limits

Single trusted coordinator; no Byzantine fault tolerance, arbitrary network
partition guarantee, or uninterrupted coordinator high availability. Worker
rejoins happen at the next training request. Coordinator restart requires worker
re-registration. At most four attempts per step bound transient recovery. Whole training
requests are not idempotent after client cancellation; see README for details.

The original source is preserved in `backups/before-upgrade-2026-09-26.tar.gz`.
Test containers were stopped after validation. CIFAR data, checkpoints, reports,
and the local Docker image remain available for reproduction.
