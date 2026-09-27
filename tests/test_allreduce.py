import socket
import threading

import numpy as np

from communication.grpc_server import serve
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


def test_ring_allreduce_averages_all_workers():
    ports = _available_ports(3)
    endpoints = {
        rank: f"127.0.0.1:{port}"
        for rank, port in enumerate(ports)
    }
    tensors = {
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([2.0, 3.0, 4.0]),
        2: np.array([3.0, 4.0, 5.0]),
    }
    workers = {
        rank: DistributedWorker(rank, len(ports), endpoints)
        for rank in endpoints
    }
    servers = []
    results = {}
    errors = {}
    barrier = threading.Barrier(len(workers), timeout=5)

    try:
        for rank, port in enumerate(ports):
            servers.append(serve(workers[rank], "127.0.0.1", port))

        def run_worker(rank):
            try:
                barrier.wait()
                results[rank] = workers[rank].ring.allreduce(tensors[rank])
            except Exception as exc:
                errors[rank] = exc

        threads = [
            threading.Thread(target=run_worker, args=(rank,), daemon=True)
            for rank in workers
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not any(thread.is_alive() for thread in threads), "AllReduce timed out"
        assert not errors
        for result in results.values():
            assert np.allclose(result, [2.0, 3.0, 4.0])
    finally:
        for server in servers:
            server.stop(0).wait()
        for worker in workers.values():
            worker.heartbeat.stop()
