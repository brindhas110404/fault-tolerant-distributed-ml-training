# Run the distributed-training demo

Requires Docker Desktop and Python 3.11 or newer. Run commands from the repository root.

## Set up the local client

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. communication/protocol.proto
```

## Train with four workers

```sh
docker compose up --build -d
docker compose ps
.venv/bin/python -m client.train_cluster --steps 100 --batch-size 64
docker compose logs coordinator
docker compose down
```

Expected: 100 completed steps and four active workers. This quick demo uses the digits dataset. Existing checkpoints resume training; keep the global batch size at 64.

## Demonstrate failure recovery with three workers

Stop the normal cluster first so port 50050 is available. The image must have been built by the command above.

```sh
.venv/bin/python -m experiments.docker_recovery --workers 3 --out results/recovery-demo.json
```

The script kills one worker during training, checks survivor completion, rejoins the worker, and restarts the coordinator and workers to check checkpoint recovery. Success prints `"passed": true`. The script stops its containers afterward.

## Suggested 90-second demo recording

1. Show the coordinator and workers running.
2. Show a completed synchronized training request.
3. Run the recovery demonstration and show its successful result.
4. Explain that saved performance results use a separate, large-batch CIFAR-10 benchmark.

Read [FINAL_REPORT.md](FINAL_REPORT.md) for measured results and experimental limits. This project exposes a command-line interface, not a hosted web application. A static project page or recorded demonstration does not run the training cluster.
