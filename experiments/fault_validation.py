"""Seeded, numerical-oracle fault campaign over actual gRPC requests.

Operations are global optimizer steps. Faults are injected during prepare,
AllReduce, commit, commit-response delivery, and a live server outage.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from experiments.local_cluster import LocalCluster
from worker.dataset import load_data
from worker.model import MLP, softmax_cross_entropy
from worker.optimizer import SGD
from worker.training import decode_state


def run(operations=1000, seed=2026, output='results/fault_validation.json', workers=4):
    rng = np.random.default_rng(seed)
    reference = MLP(64, 128, 10, seed=7)
    optimizer = SGD(reference.parameters(), learning_rate=0.05)
    X, _, y, _ = load_data()
    counts = Counter()
    recoveries = 0
    first_attempt_successes = 0
    maximum_error = 0.0
    started = time.perf_counter()
    with LocalCluster(workers) as cluster:
        for step in range(operations):
            # Every fifth step injects a fault, cycling through all five phases.
            mode = ['prepare', 'allreduce', 'commit_before', 'commit_after', 'server_outage'][(step // 5) % 5] if step % 5 == 0 else 'healthy'
            rank = int(rng.integers(0, workers))
            worker = cluster.workers[rank]
            restore = None
            if mode in ('prepare', 'commit_before', 'commit_after', 'server_outage'):
                name = 'train_transaction' if mode in ('prepare', 'server_outage') else 'commit_training'
                original = getattr(worker, name)
                fired = [False]
                def injected(*args, _original=original, _mode=mode, **kwargs):
                    if not fired[0]:
                        fired[0] = True
                        if _mode == 'commit_after':
                            _original(*args, **kwargs)
                        if _mode == 'server_outage':
                            cluster.servers[rank].stop(0)
                        raise RuntimeError('injected ' + _mode)
                    return _original(*args, **kwargs)
                setattr(worker, name, injected)
                restore = lambda: setattr(worker, name, original)
            elif mode == 'allreduce':
                original = worker.receive_tensor
                fired = [False]
                def injected(*args, **kwargs):
                    if not fired[0]:
                        fired[0] = True
                        raise RuntimeError('injected mid-AllReduce failure')
                    return original(*args, **kwargs)
                worker.receive_tensor = injected
                restore = lambda: setattr(worker, 'receive_tensor', original)
            batch_size = 13  # Uneven shards test weighted gradient averaging.
            try:
                cluster.coordinator.start_training(1, batch_size)
                counts[mode] += 1
                expected_members = workers - int(mode == 'server_outage')
                assert len(cluster.coordinator.last_run['active_workers']) == expected_members, 'unexpected membership loss'
                recovered = cluster.coordinator.last_run['recoveries']
                recoveries += recovered
                first_attempt_successes += int(recovered == 0)
                if mode != 'healthy':
                    assert fired[0] and recovered >= 1, f'fault not exercised: step={step}, mode={mode}, rank={rank}, active={cluster.coordinator.last_run["active_workers"]}'
                indices = (np.arange(batch_size) + step * batch_size) % len(X)
                _, grad = softmax_cross_entropy(reference.forward(X[indices]), y[indices])
                reference.backward(grad)
                optimizer.step(reference.gradients())
                expected = reference.parameters()
                actual = decode_state(cluster.coordinator.checkpoint)
                for a, b in zip(actual, expected):
                    maximum_error = max(maximum_error, float(np.max(np.abs(a - b))))
                    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)
                for r in cluster.coordinator.last_run['active_workers']:
                    for a, b in zip(cluster.workers[r].model.parameters(), expected):
                        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)
                    assert len(cluster.workers[r]._allreduce_ops) <= 1
                assert cluster.coordinator.total_steps == step + 1
            finally:
                if restore:
                    restore()
                if mode == 'server_outage':
                    cluster.restart(rank)
            if (step + 1) % 100 == 0:
                print(f'{step + 1}/{operations} operations verified; {recoveries} recoveries', flush=True)
    report = dict(operations=operations, seed=seed, passed=True, counts=dict(counts),
                  recoveries=recoveries, max_absolute_error=maximum_error,
                  elapsed_seconds=time.perf_counter() - started,
                  transport=f'real loopback gRPC, {workers} workers in one Python process',
                  worker_count=workers, logical_operation='one committed optimizer step with one successful fused AllReduce',
                  completion_rate=1.0, injected_faults=operations - counts['healthy'],
                  first_attempt_success_rate=first_attempt_successes / operations,
                  success_definition='numerically correct completion after bounded recovery; retries included',
                  empirical_target=.999, empirical_target_met=True,
                  one_sided_95_percent_lower_bound=.05 ** (1 / operations),
                  oracle='serial SGD with identical global batches',
                  scope='transient RPC faults and worker service outages; not network partitions or coordinator crashes')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--operations', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', default='results/fault_validation.json')
    args = parser.parse_args()
    print(json.dumps(run(args.operations, args.seed, args.out, args.workers), indent=2))
