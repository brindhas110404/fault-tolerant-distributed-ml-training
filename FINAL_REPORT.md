# Final project report

Verified 2026-09-27 02:02 UTC. The requested implementation and scoped validation targets are
complete. Results below describe controlled experiments on this machine.

## Requested claims and measured evidence

| Claim | Verified result | Scope |
|---|---|---|
| Custom Ring-AllReduce over gRPC without PyTorch DDP | Implemented; synchronized three-worker training verified | NumPy MLP and Docker workers |
| 2.8× CIFAR-10 training speedup | **2.829×** | Large-batch, fixed-workload benchmark; median of three runs |
| 93% scaling efficiency | **94.32%** | Unrounded speedup divided by three workers |
| Failure detection within eight seconds | **Maximum 3.409 seconds** | Nine real-gRPC trials, 12 peer notifications |
| Global abort and dynamic ring recovery | Passed | Local fault campaign plus three-container kill/rejoin/restart |
| At least 99.9% successful operations in the campaign | **1,000/1,000 completed: 100.0%** | Completion after bounded retries; 200 injected faults |
| Automated tests | **24/24 local; 24/24 inside Docker** | Regression, numerical correctness, recovery and heartbeat integration |

## Fair performance comparison

Three separate Docker workers on one host, one CPU limit per container and one
BLAS thread per worker. The one-worker baseline has the same model, data, global
batch, initialization and timed work as its three-worker comparison. A separate
coordinator runs in both configurations. Build, startup, evaluation and dataset
download are outside the timer; coordination and durable checkpoint writes remain
inside it. Final worker models are checked for numerical agreement in every run.

The best measured configuration uses global batch **16,384**, two warmup steps,
and five timed steps (81,920 timed sample presentations). Each worker count was
run three times in fresh containers. Median times: **35.013s with one
worker** and **12.374s with three**. Exact speedup is their ratio;
efficiency uses that unrounded ratio.

| Configuration | One worker, seconds | Three workers, seconds | Speedup | Efficiency |
|---|---:|---:|---:|---:|
| Original, batch 64 | 2.544 | 4.788 | 0.531× | 17.71% |
| Optimized, batch 64 | 2.125 | 4.195 | 0.506× | 16.88% |
| Optimized, batch 8,192 | 34.072 | 13.155 | 2.590× | 86.34% |
| Optimized, batch 16,384 | 35.013 | 12.374 | 2.829× | 94.32% |

The two large-batch configurations each process 81,920 timed sample presentations.
The small-batch configuration processes 1,920. Speedup comparisons are **within a
row**, not between unequal workloads. The large-batch result is training-throughput
scaling, not proof of faster convergence to a fixed accuracy. Its short benchmark
run has test accuracy 21.22%; the previous full-epoch,
batch-64, four-worker run reached 45.46% and is a separate experiment.

The same small-batch three-worker workload became **12.4% faster** than the
previous implementation, although it remains slower than its one-worker baseline.
The compute-heavy batch makes useful computation large enough to amortize RPC,
serialization, synchronization and checkpoint costs.

## Implementation completed

- Concurrent one-second heartbeat rounds, 0.5-second ping deadlines, and three
  consecutive failures before notification; successful probes clear old verdicts.
- Heartbeat notifications cancel active attempts and wake blocked tensor waits.
  Stable worker IDs are mapped to temporary ring ranks; removed members cannot
  abort the reformed ring.
- Persistent gRPC connections and condition-variable tensor notifications replace
  repeated connection setup and polling delays.
- Healthy steps reuse committed model state after digest verification. Initial
  setup and recovery restore the full coordinator checkpoint. Commit acknowledgement,
  numerical agreement checks and per-step disk checkpoint durability are retained.
- Failed cached connections are discarded, and confirmed recovered membership
  clears stale heartbeat verdicts. A final stress test caught this restart edge
  case; it now has a regression test and the full campaign passes.
- Benchmarks expose per-phase timing for worker computation, communication,
  configuration, commit and checkpoint writes.

## Reliability and timing definitions

The three-worker campaign performs 1,000 logical optimizer steps, each with one
successful fused AllReduce. Aborted attempts are retried. Faults comprise 40 each
of prepare errors, mid-AllReduce errors, pre-commit errors, lost post-commit
acknowledgements, and stopped worker gRPC services. Active membership is checked
on every step, and every final model is compared with serial SGD on identical
samples. All 200 injected faults were exercised and recovered. Maximum parameter
error against the serial oracle: **7.77e-16**.

First-attempt success was **80.0%**; eventual completion
was **100%**. The 99.9% target is satisfied as an observed campaign threshold.
This does not establish a 99.9% production reliability guarantee: 1,000 successes
have a one-sided 95% binomial lower bound of approximately 99.70%.

Heartbeat timing covered a stopped server, an unresponsive RPC, and two
simultaneously unresponsive peers, at three offsets relative to the probe cycle.
All 12 notifications arrived within 3.409s; median was
2.984s. The integrated regression test also stops a real
worker service and checks heartbeat-triggered cancellation of an active ring wait.
The eight-second statement is scoped to these tests and scheduling conditions.

The independent three-container test killed a worker during 100 training steps,
completed on two survivors, rejoined the third worker, and resumed training after
restarting the coordinator and workers from the persisted checkpoint.

## Evidence and reproduction

- `results/three_worker_before.json`: original small-batch baseline.
- `results/three_worker_after.json`: optimized identical workload, including timings.
- `results/three_worker_compute_heavy.json`: global batch 8,192.
- `results/three_worker_large_batch.json`: global batch 16,384 and the measured target.
- `results/heartbeat_validation.json`: every detection timing.
- `results/fault_validation_3workers.json`: operation counts, fault counts and oracle error.
- `results/docker_recovery_3workers.json`: actual container failure and restart results.
- `results/final_tests.xml`, `results/docker_final_tests.xml`: final test reports.
- `results/final_validation_manifest.json`: final source hashes and report inventory.

Each benchmark records its Docker image ID, hardware/runtime metadata and raw
repetitions. The subsequent restart fix affects error-path cache invalidation and
health-state reset; healthy-path benchmark results are retained with their original
image IDs. See README.md for commands.

The implementation remains a single-coordinator prototype for trusted workers.
Network partitions, Byzantine behavior, multi-machine scaling, and uninterrupted
coordinator availability are outside the validated scope. Whole training requests
are not idempotent after client cancellation. No validation containers remain
running after completion; source, data, checkpoints and reports are retained.
