"""Kill a Docker worker during training; verify survivor recovery, rejoin and restart."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from client.train_cluster import train


def main(workers=4, output='results/docker_recovery.json'):
    if workers not in (3, 4):
        raise ValueError('supported validation sizes: 3 or 4')
    root = Path(__file__).resolve().parents[1]
    endpoints = ','.join(f'worker{rank}:{50060+rank}' for rank in range(workers))
    worker_names = [f'worker{rank}' for rank in range(workers)]
    checkpoints = root / 'results' / f'recovery-{workers}-checkpoints'
    checkpoints.mkdir(exist_ok=True)
    overrides = {'coordinator': {'command': ['python', '-m', 'coordinator.coordinator', '--workers', str(workers),
        '--worker-endpoints', endpoints, '--dataset', 'digits', '--checkpoint', '/app/results/checkpoint.npz'],
        'volumes': [f'{checkpoints}:/app/results']}}
    for rank, name in enumerate(worker_names):
        overrides[name] = {'command': ['python', '-m', 'worker.worker', '--worker-id', str(rank),
            '--world-size', str(workers), '--port', str(50060+rank), '--endpoints', endpoints,
            '--coordinator', 'coordinator:50050', '--dataset', 'digits']}
    with tempfile.TemporaryDirectory(prefix='p3-recovery-') as directory:
        override = Path(directory) / 'override.json'
        override.write_text(json.dumps({'services': overrides}))
        compose = [shutil.which('docker'), 'compose', '-p', f'p3-recovery-{workers}',
                   '-f', str(root / 'docker-compose.yml'), '-f', str(override)]
        def run(*args):
            return subprocess.check_output(compose + list(args), text=True, close_fds=False).strip()
        report = {'workers': workers}
        try:
            run('up', '-d', 'coordinator', *worker_names)
            initial = train('127.0.0.1:50050', 1, 64)
            with ThreadPoolExecutor() as pool:
                future = pool.submit(train, '127.0.0.1:50050', 100, 64)
                time.sleep(.15)
                run('kill', worker_names[-1])
                result = future.result()
            assert result['completed_steps'] == 100
            assert len(result['active_workers']) == workers - 1
            assert result['recoveries'] >= 1
            report['worker_kill_run'] = result
            run('start', worker_names[-1])
            time.sleep(3)
            rejoined = train('127.0.0.1:50050', 2, 64)
            assert len(rejoined['active_workers']) == workers
            report['rejoined_run'] = rejoined
            run('restart', 'coordinator')
            run('restart', *worker_names)
            resumed = train('127.0.0.1:50050', 2, 64)
            assert int(resumed['total_steps']) == int(initial['total_steps']) + 104
            report['checkpoint_resume'] = resumed
            report['passed'] = True
            target = root / output
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))
        finally:
            run('down')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', default='results/docker_recovery.json')
    args = parser.parse_args()
    main(args.workers, args.out)
