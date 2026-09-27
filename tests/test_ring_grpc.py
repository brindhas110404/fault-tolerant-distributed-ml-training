import multiprocessing
import socket
import time

import numpy as np

from communication.grpc_server import serve
from communication.ring_allreduce import RingAllReduce
from worker.worker import DistributedWorker


WORLD_SIZE = 4


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


def run_worker(rank, ports, result_queue):
    endpoints = {
        i: f"127.0.0.1:{port}"
        for i, port in enumerate(ports)
    }

    worker = DistributedWorker(
        worker_id=rank,
        world_size=WORLD_SIZE,
        endpoints=endpoints,
    )

    server = serve(
        worker,
        "127.0.0.1",
        ports[rank],
    )

    # Give the gRPC server a moment to start.
    time.sleep(1)

    tensor = np.array(
        [float(rank + 1)] * 4,
        dtype=np.float64,
    )

    result = worker.ring.allreduce(tensor)

    result_queue.put((rank, result))

    server.stop(0)


def test_real_grpc_ring_allreduce():
    ports = _available_ports(WORLD_SIZE)
    result_queue = multiprocessing.Queue()

    processes = []

    for rank in range(WORLD_SIZE):
        process = multiprocessing.Process(
            target=run_worker,
            args=(rank, ports, result_queue),
        )

        process.start()
        processes.append(process)

    results = {}

    for _ in range(WORLD_SIZE):
        rank, result = result_queue.get(timeout=30)
        results[rank] = result

    for process in processes:
        process.join(timeout=10)

    assert not any(process.is_alive() for process in processes)
    assert all(process.exitcode == 0 for process in processes)

    expected = np.array(
        [2.5, 2.5, 2.5, 2.5],
        dtype=np.float64,
    )

    for rank in range(WORLD_SIZE):
        assert np.allclose(
            results[rank],
            expected,
        )
