import socket

import grpc
import numpy as np

from communication import protocol_pb2
from communication import protocol_pb2_grpc
from communication.grpc_server import serve
from coordinator.coordinator import Coordinator
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


def test_coordinator_registers_workers_and_runs_synchronized_training():
    coordinator_port, *worker_ports = _available_ports(3)
    endpoints = {
        rank: f"127.0.0.1:{port}"
        for rank, port in enumerate(worker_ports)
    }
    workers = {
        rank: DistributedWorker(rank, 2, endpoints)
        for rank in endpoints
    }
    worker_servers = []
    coordinator = Coordinator(2, list(endpoints.values()))
    coordinator_server = None
    channel = None

    try:
        for rank, port in enumerate(worker_ports):
            worker_servers.append(serve(workers[rank], "127.0.0.1", port))
        coordinator_server = coordinator.start("127.0.0.1", coordinator_port)

        channel = grpc.insecure_channel(f"127.0.0.1:{coordinator_port}")
        stub = protocol_pb2_grpc.CoordinatorServiceStub(channel)
        for rank in endpoints:
            response = stub.RegisterWorker(
                protocol_pb2.RegisterWorkerRequest(
                    worker_id=rank,
                    endpoint=endpoints[rank],
                ),
                timeout=5,
            )
            assert response.accepted

        status = stub.GetClusterStatus(
            protocol_pb2.ClusterStatusRequest(),
            timeout=5,
        )
        assert status.ready
        assert list(status.registered_workers) == [0, 1]

        result = stub.StartTraining(
            protocol_pb2.TrainingRequest(steps=1, batch_size=8),
            timeout=30,
        )
        assert result.accepted
        assert result.completed_steps == 1
        assert len(result.mean_losses) == 1
        assert result.mean_losses[0] > 0

        for first, second in zip(
            workers[0].model.parameters(),
            workers[1].model.parameters(),
        ):
            assert np.allclose(first, second)
    finally:
        if channel is not None:
            channel.close()
        if coordinator_server is not None:
            coordinator_server.stop(0).wait()
        for server in worker_servers:
            server.stop(0).wait()
        for worker in workers.values():
            worker.heartbeat.stop()
