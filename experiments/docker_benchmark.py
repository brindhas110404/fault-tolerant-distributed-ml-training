"""Strong-scaling measurements across separate Docker worker containers.

Fixed global batches, model, initial weights, warmup and steps for every size.
Includes coordinator/checkpoint RPC overhead; excludes image build and startup.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time
import grpc
import numpy as np
from client.train_cluster import train
from communication import protocol_pb2 as pb, protocol_pb2_grpc as rpc
from coordinator.coordinator import OPTIONS
from worker.dataset import load_dataset
from worker.model import MLP, softmax_cross_entropy
from worker.training import decode_state


def command(args):
    return subprocess.check_output(args, text=True).strip()


def evaluate(state, dataset, data_dir):
    _, X, _, y = load_dataset(dataset, data_dir)
    model = MLP(X.shape[1], 128, 10, seed=7)
    for p, value in zip(model.parameters(), decode_state(state)):
        p[:] = value
    total_loss, correct = 0., 0
    for start in range(0, len(X), 128):
        x = X[start:start+128].astype(np.float64)
        if dataset == 'cifar10':
            x = x / 255.0 - 0.5
        labels = y[start:start+128]
        logits = model.forward(x)
        loss, _ = softmax_cross_entropy(logits, labels)
        total_loss += loss * len(labels)
        correct += int(np.sum(logits.argmax(axis=1) == labels))
    return dict(test_samples=len(X), test_loss=total_loss / len(X), test_accuracy=correct / len(X))


def run(dataset='digits', steps=20, repeats=3, batch_size=64, output=None, workers=(1, 2, 4), image='project3-training:local'):
    root = Path(__file__).resolve().parents[1]
    rows = []
    if not workers or workers[0] != 1:
        raise ValueError("benchmark must start with one-worker baseline")
    for count in workers:
        times, runs = [], []
        for repetition in range(repeats):
            project = f'p3bench-{dataset}-{count}-{repetition}'
            endpoints = ','.join(f'worker{i}:50060' for i in range(count))
            services = {'coordinator': dict(image=image, cpus=1,
                command=['python', '-m', 'coordinator.coordinator', '--workers', str(count),
                         '--worker-endpoints', endpoints, '--dataset', dataset],
                ports=['127.0.0.1::50050'])}
            for rank in range(count):
                services[f'worker{rank}'] = dict(image=image, cpus=1,
                    environment={'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
                    volumes=[f'{root / "data"}:/app/data:ro'],
                    ports=['127.0.0.1::50060'],
                    command=['python', '-m', 'worker.worker', '--worker-id', str(rank),
                             '--world-size', str(count), '--port', '50060', '--endpoints', endpoints,
                             '--coordinator', 'coordinator:50050', '--dataset', dataset])
            with tempfile.TemporaryDirectory(prefix='p3bench-') as directory:
                config = Path(directory) / 'compose.json'
                config.write_text(json.dumps({'services': services}))
                compose = ['docker', 'compose', '-p', project, '-f', str(config)]
                try:
                    subprocess.run(compose + ['up', '-d'], check=True, stdout=subprocess.DEVNULL)
                    endpoint = command(compose + ['port', 'coordinator', '50050'])
                    train(endpoint, 2, batch_size)
                    started = time.perf_counter()
                    result = train(endpoint, steps, batch_size)
                    elapsed = time.perf_counter() - started
                    times.append(elapsed)
                    states = []
                    for rank in range(count):
                        worker_endpoint = command(compose + ['port', f'worker{rank}', '50060'])
                        with grpc.insecure_channel(worker_endpoint, options=OPTIONS) as channel:
                            state = rpc.WorkerServiceStub(channel).GetState(pb.PingRequest(), timeout=10)
                            states.append(state.state)
                    first = decode_state(states[0])
                    for state in states[1:]:
                        for actual, expected in zip(decode_state(state), first):
                            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
                    metrics = evaluate(states[0], dataset, root / 'data')
                    profiles = []
                    for line in command(compose + ['logs', '--no-color', 'coordinator']).splitlines():
                        payload = line.split('|', 1)[-1].strip()
                        if payload.startswith('{"training_profile":'):
                            profiles.append(json.loads(payload)['training_profile'])
                    runs.append(dict(seconds=elapsed, result=result, profile=profiles[-1] if profiles else {}, **metrics))
                    print(f'{dataset}: {count} workers, repeat {repetition+1}: {elapsed:.3f}s, accuracy {metrics["test_accuracy"]:.3f}', flush=True)
                finally:
                    subprocess.run(compose + ['down', '--remove-orphans'], check=True, stdout=subprocess.DEVNULL)
        rows.append(dict(workers=count, median_seconds=statistics.median(times), runs=runs))
    baseline = rows[0]['median_seconds']
    for row in rows:
        row['speedup'] = baseline / row['median_seconds']
        row['efficiency'] = row['speedup'] / row['workers']
        row['samples_per_second'] = steps * batch_size / row['median_seconds']
    report = dict(dataset=dataset, steps=steps, warmup_steps=2, repeats=repeats,
                  global_batch_size=batch_size, cpu_limit_per_container=1,
                  host=platform.platform(), docker=command(['docker', 'info', '--format', '{{json .}}']),
                  image=command(['docker', 'image', 'inspect', image, '--format', '{{.Id}}']),
                  method='separate Docker containers on one host; median wall time; fixed global workload', rows=rows)
    # Store only hardware/runtime fields, not unrelated Docker daemon configuration.
    info = json.loads(report['docker'])
    report['docker'] = {key: info.get(key) for key in ('ServerVersion', 'NCPU', 'MemTotal', 'Architecture', 'OperatingSystem')}
    target = Path(output or root / 'results' / f'docker_benchmark_{dataset}.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n')
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['digits', 'cifar10'], default='digits')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out')
    parser.add_argument('--workers', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--image', default='project3-training:local')
    args = parser.parse_args()
    run(args.dataset, args.steps, args.repeats, args.batch_size, args.out, args.workers, args.image)
