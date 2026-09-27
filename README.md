# Distributed ML training with NumPy, gRPC, and Docker

Final verified results: **2.83× speedup, 94.3% efficiency**, maximum **3.41s heartbeat detection**, and **1,000/1,000 fault-campaign completions after recovery**. See [FINAL_REPORT.md](FINAL_REPORT.md) for workload and test conditions.

A synchronous data-parallel training system with a NumPy MLP, fused Ring-AllReduce,
coordinator-controlled steps, checkpoint rollback, and worker recovery. Supports the
scikit-learn digits dataset and the official CIFAR-10 train/test split.

For a short runnable demonstration, see [DEMO.md](DEMO.md).

## Quick start

Use Python 3.11 or newer and Docker Desktop. From this directory:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. communication/protocol.proto
docker compose up --build -d
.venv/bin/python -m client.train_cluster --steps 100 --batch-size 64 --out results/training.json
docker compose down
```

The existing local environment is `project3/bin/python`. Because the project folder
was moved, invoke that executable directly instead of relying on old activation
scripts or the old `/Users/brindha/Desktop/DS` path.

`--batch-size` is the **global batch size**, divided across active workers. It must
be at least the active worker count and must remain fixed for a checkpoint lineage.
Uneven shards are weighted by sample count. Worker IDs stay stable after failure;
temporary ring ranks are assigned by the coordinator.

Training resumes from `results/docker/checkpoint-digits.npz`. To start a new run,
move that checkpoint to a backup name before bringing the cluster up. Use only one
coordinator per checkpoint. The coordinator port is 50050; internal worker ports
are 50060–50063.

## CIFAR-10

```sh
project3/bin/python -m worker.prepare_cifar --data-dir data
DATASET=cifar10 docker compose up --build -d
project3/bin/python -m client.train_cluster --steps 100 --batch-size 64 --out results/cifar10_training.json
docker compose exec worker0 python -m experiments.evaluate_cluster --dataset cifar10
docker compose down
```

For one full pass over 50,000 training images, use at least 782 steps at global
batch size 64 (the last batch wraps deterministically). Evaluation reports actual
test accuracy; short benchmark runs are not convergence studies.

The loader downloads the official binary archive, checks MD5
`c32a1d4ab5d03f1284b67883e8d87530`, and reads the six expected binary members without
executing pickle or extracting arbitrary paths. The official split is 50,000 train
and 10,000 test images. Data is memory-mapped as uint8; each minibatch is flattened
to 3,072 features and normalized to `pixel / 255 - 0.5`.

This is a CPU MLP with 128 hidden units, ReLU, ten output classes, and SGD with a
learning rate of 0.05. It is a distributed-systems demonstration, not a competitive
CIFAR image-classification model. Evaluation uses the held-out test split and is
outside the benchmark timer. No augmentation or GPU training is implemented.

Dataset source: https://www.cs.toronto.edu/~kriz/cifar.html
Reference: Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.

## Training and recovery protocol

1. Workers register their configured ID and endpoint with the coordinator.
2. The coordinator restores a shared checkpoint (or verifies the previously committed local model digest) and assigns a unique attempt token
   plus an ordered active membership to every worker.
3. Workers compute their part of the same deterministic global batch. Gradients
   are fused into one vector and averaged through real gRPC Ring-AllReduce.
4. Workers stage candidate parameters. The coordinator checks sample counts and
   numerical agreement before issuing idempotent commit commands.
5. Only after all commits acknowledge does the coordinator atomically replace its
   checkpoint and advance the global step counter.
6. Any RPC/worker failure triggers a global abort. Peers stop waiting in AllReduce;
   reachable survivors restore the previous checkpoint, receive a new attempt
   token and membership, and retry the **same** global batch.

An uncertain commit (a worker applies the update but its acknowledgement is lost)
therefore cannot silently advance only part of the cluster. There are at most four
attempts per step. Persistent failures stop training with the last checkpoint
retained. A restarted worker joins at the next training request after registering
and receives the current checkpoint. On coordinator restart, restart workers to
register again; the default Docker volume preserves coordinator checkpoints.

Early AllReduce messages wait through safe NOT_FOUND retries. Duplicate tensor
messages are idempotent. Attempt tokens fence stale train/commit messages, and old
AllReduce state is cleared at the next configured attempt.

## Validation

```sh
PYTHONDONTWRITEBYTECODE=1 project3/bin/python -m pytest -q -p no:cacheprovider --junitxml=results/tests.xml
OPENBLAS_NUM_THREADS=1 project3/bin/python -m experiments.fault_validation --operations 1000
project3/bin/python -m experiments.docker_recovery
```

The fault campaign uses four in-process workers communicating through real
loopback gRPC. It performs 1,000 global optimizer steps: 800 healthy steps and 200
faulted steps, split across prepare failures, mid-AllReduce failures, failures
before commit, lost responses after commit, and stopped worker gRPC services.
Every step is compared with serial SGD using identical input samples. This is
**1,000 operations with 200 injected faults**, not 1,000 separate process crashes.
The Docker recovery check separately kills a worker container during training,
checks survivor completion, rejoins the worker, and restarts the coordinator.

## Real scaling measurements

```sh
docker build -t project3-training:local .
OPENBLAS_NUM_THREADS=1 project3/bin/python -m experiments.docker_benchmark --dataset digits --steps 30 --repeats 3
OPENBLAS_NUM_THREADS=1 project3/bin/python -m experiments.docker_benchmark --dataset cifar10 --steps 30 --repeats 3
```

Each run uses separate Docker containers for one coordinator and 1, 2, or 4 workers
on the same host. Each container has a one-CPU limit; BLAS is limited to one thread.
Model, seed, global batch size, two warmup steps, and measured steps stay fixed.
Three fresh clusters per configuration provide median elapsed times. Timings
include training control RPCs and checkpoint writes but exclude startup, image
build, evaluation, and dataset download. Speedup is `T1 / Tp`; efficiency is
`speedup / p`. Container startup and teardown may take longer than these small
training workloads. Run benchmarks without other heavy work on the machine.

`results/docker_benchmark_*.json` contains raw runs, held-out accuracy, runtime
metadata, image ID, median times, speedup, and efficiency. Slowdown is a valid
measurement: small CPU workloads can cost more in communication and coordination
than they save in computation. Do not claim positive speedup unless the data shows it.

Historical `main.py`, `experiments/benchmark.py`, and `results/scaling.json` are a
single-process simulation. They are retained for reference and **are not evidence
of distributed performance**. Use the Docker benchmark for current results.

## Scope and limits

- One trusted coordinator and trusted workers on a private network; no TLS/auth.
- Worker fail-stop/transient RPC failures, bounded retries, and checkpoint restart
  are supported. Byzantine behavior, split-brain coordinators, arbitrary network
  partitions, and uninterrupted coordinator high availability are not claimed.
- Models may differ by roundoff after Ring-AllReduce; comparison uses numerical
  tolerance. Healthy steps reuse their validated local committed models; recovery restores a canonical coordinator checkpoint.
- Checkpoints contain parameters, dataset, step, and fixed global batch size. SGD
  has no momentum state. Model topology/learning-rate changes require a new run.
- No elastic joins in the middle of a StartTraining request. Registration readiness
  is not continuous health; live membership is checked at training/recovery time.
- A canceled client request may leave its server-side training job running. Do not
  retry an uncertain entire StartTraining call as if it were idempotent; inspect
  the checkpoint/run state first.

The exact Docker dependency snapshot is in `requirements.lock.txt`. The original
source backup is in `backups/before-upgrade-2026-09-26.tar.gz`.

See [STATUS.md](STATUS.md) for measured results from this implementation.


## Final heartbeat and three-worker validation

Heartbeats now probe peers concurrently every one second, with a 0.5-second
per-peer deadline and a threshold of three consecutive failures. A failure
notification for an active member cancels the local training attempt and wakes
AllReduce waits; the coordinator then performs the existing global abort and
rollback. Temporary ring ranks are mapped to stable worker IDs. Removed members
cannot cancel a new ring, and successful probes clear stale failure status.

The eight-second goal is tested under explicit fail-stop and unresponsive-RPC
scenarios. It is not a real-time guarantee under arbitrary host starvation or
network partition. `results/heartbeat_validation.json` records every timed trial.

The optimized transport reuses gRPC connections and uses condition notifications
instead of fixed-interval tensor polling. Healthy steps avoid re-sending model
checkpoints when worker digests match their last acknowledged commits. Full
checkpoints remain mandatory for initial setup and recovery; disk durability and
commit checks were retained. Benchmark reports include phase-level timing.

```sh
project3/bin/python -m experiments.heartbeat_validation
project3/bin/python -m experiments.fault_validation --workers 3 --operations 1000 --out results/fault_validation_3workers.json
project3/bin/python -m experiments.docker_benchmark --dataset cifar10 --workers 1 3 --steps 30 --repeats 3 --out results/three_worker_after.json
project3/bin/python -m experiments.docker_benchmark --dataset cifar10 --workers 1 3 --batch-size 8192 --steps 10 --repeats 3 --out results/three_worker_compute_heavy.json
```

Reliability is defined as numerically correct completion of a logical optimizer
step (including its fused AllReduce) after bounded recovery. The campaign reports
first-attempt success separately from completion after retries. An observed
1,000/1,000 completion result meets a 99.9% empirical campaign threshold, but does
not establish 99.9% production reliability. With 1,000 successes and no failures,
the one-sided 95% binomial lower bound is approximately 99.70%.

Use `FINAL_REPORT.md` for the final measured claims. Earlier results
remain available for provenance; workload sizes and build versions must not be
mixed when comparing speedups.
