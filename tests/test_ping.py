import socket

import grpc

from communication import protocol_pb2
from communication import protocol_pb2_grpc
from communication.grpc_server import serve
from worker.worker import DistributedWorker


def test_worker_responds_to_ping():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    endpoints = {0: f"127.0.0.1:{port}"}
    worker = DistributedWorker(0, 1, endpoints)
    server = serve(worker, "127.0.0.1", port)
    channel = grpc.insecure_channel(endpoints[0])

    try:
        response = protocol_pb2_grpc.WorkerServiceStub(channel).Ping(
            protocol_pb2.PingRequest(worker_id=0),
            timeout=5,
        )
        assert response.ok
    finally:
        channel.close()
        server.stop(0).wait()
        worker.heartbeat.stop()
