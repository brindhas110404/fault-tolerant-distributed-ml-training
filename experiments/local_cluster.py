"""Real loopback gRPC cluster for correctness and fault validation (not benchmarks)."""
import socket
from communication.grpc_server import serve
from coordinator.coordinator import Coordinator
from worker.worker import DistributedWorker


class LocalCluster:
    def __init__(self, count=4, checkpoint_path=None):
        sockets = [socket.socket() for _ in range(count)]
        for sock in sockets:
            sock.bind(('127.0.0.1', 0))
        self.ports = [sock.getsockname()[1] for sock in sockets]
        for sock in sockets:
            sock.close()
        self.endpoints = {r: f'127.0.0.1:{p}' for r, p in enumerate(self.ports)}
        self.workers = {r: DistributedWorker(r, count, self.endpoints) for r in self.endpoints}
        self.servers = {r: serve(w, '127.0.0.1', self.ports[r]) for r, w in self.workers.items()}
        self.coordinator = Coordinator(count, list(self.endpoints.values()), checkpoint_path=checkpoint_path,
                                       rpc_timeout=5)
        for r in self.workers:
            self.coordinator.register_worker(r, self.endpoints[r])

    def restart(self, rank):
        self.servers[rank] = serve(self.workers[rank], '127.0.0.1', self.ports[rank])
        self.coordinator.register_worker(rank, self.endpoints[rank])

    def close(self):
        for server in self.servers.values():
            server.stop(0).wait()
        for worker in self.workers.values():
            worker.close()
        self.coordinator.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
