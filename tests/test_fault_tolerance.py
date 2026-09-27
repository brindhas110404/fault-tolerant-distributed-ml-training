import socket
import threading
import time

import numpy as np

from communication.grpc_server import serve
from worker.exceptions import WorkerFailedError
from worker.worker import DistributedWorker


def _available_ports(count):
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def test_ring_reforms_after_worker_failure():
    ports = _available_ports(3)
    endpoints = {
        rank: f"127.0.0.1:{port}"
        for rank, port in enumerate(ports)
    }
    workers = {
        rank: DistributedWorker(rank, len(ports), endpoints)
        for rank in endpoints
    }
    servers = []
    results = {}
    errors = {}
    recovery_barrier = threading.Barrier(2, timeout=5)

    try:
        for rank, port in enumerate(ports):
            servers.append(serve(workers[rank], "127.0.0.1", port))

        for worker in workers.values():
            worker.heartbeat.interval = 0.1
            worker.heartbeat.max_failures = 2
        servers[2].stop(0).wait()

        deadline = time.monotonic() + 5
        while 2 not in workers[0].get_dead_workers() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert 2 in workers[0].get_dead_workers()
        assert 2 in workers[1].get_dead_workers()

        tensors = {
            0: np.array([1.0, 2.0, 3.0]),
            1: np.array([2.0, 3.0, 4.0]),
        }

        def run_worker(rank):
            try:
                try:
                    result = workers[rank].ring.allreduce(tensors[rank])
                except WorkerFailedError as exc:
                    workers[rank].handle_failure(exc.worker_id)
                    recovery_barrier.wait()
                    result = workers[rank].ring.allreduce(tensors[rank])
                results[rank] = result
            except Exception as exc:
                errors[rank] = exc

        threads = [
            threading.Thread(target=run_worker, args=(rank,), daemon=True)
            for rank in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not any(thread.is_alive() for thread in threads), "Recovery timed out"
        assert not errors
        for result in results.values():
            assert np.allclose(result, [1.5, 2.5, 3.5])
    finally:
        for server in servers:
            server.stop(0).wait()
        for worker in workers.values():
            worker.heartbeat.stop()
