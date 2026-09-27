"""Concurrent, fixed-cadence failure detection with bounded per-peer RPCs.

Scheduling/network assumptions matter: the eight-second goal is validated by
experiments, not a guarantee under arbitrary process starvation or partitions.
"""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import grpc
from communication import protocol_pb2 as pb, protocol_pb2_grpc as rpc


class HeartbeatMonitor:
    def __init__(self, worker_id, world_size, endpoints, interval=1.0,
                 max_failures=3, ping_timeout=0.5, on_failure=None):
        if interval <= 0 or max_failures <= 0 or ping_timeout <= 0:
            raise ValueError('heartbeat timings must be positive')
        self.worker_id, self.world_size = worker_id, world_size
        self.endpoints = dict(endpoints)
        self.interval, self.max_failures, self.ping_timeout = interval, max_failures, ping_timeout
        self.on_failure = on_failure
        self._failure_counts = {r: 0 for r in endpoints if r != worker_id}
        self._dead_workers = set()
        self.detected_at = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=self.ping_timeout + self.interval + 1)

    def is_alive(self, rank):
        with self._lock:
            return rank not in self._dead_workers

    def get_dead_workers(self):
        with self._lock:
            return set(self._dead_workers)

    def confirm_membership(self, ranks):
        """Coordinator health checks confirmed a new/recovered membership."""
        with self._lock:
            for rank in ranks:
                if rank in self._dead_workers:
                    self._dead_workers.discard(rank)
                    self._failure_counts[rank] = 0

    def _ping(self, rank):
        try:
            with grpc.insecure_channel(self.endpoints[rank]) as channel:
                return rpc.WorkerServiceStub(channel).Ping(
                    pb.PingRequest(worker_id=self.worker_id), timeout=self.ping_timeout).ok
        except grpc.RpcError:
            return False

    def _run(self):
        peers = list(self._failure_counts)
        if not peers:
            return
        with ThreadPoolExecutor(max_workers=len(peers)) as pool:
            while not self._stop_event.is_set():
                started = time.monotonic()
                # A slow peer cannot delay checks of the others.
                results = list(pool.map(self._ping, peers))
                for rank, success in zip(peers, results):
                    notify = False
                    with self._lock:
                        if success:
                            self._failure_counts[rank] = 0
                            self._dead_workers.discard(rank)
                        else:
                            self._failure_counts[rank] += 1
                            if self._failure_counts[rank] >= self.max_failures and rank not in self._dead_workers:
                                self._dead_workers.add(rank)
                                self.detected_at[rank] = time.monotonic()
                                notify = True
                    if notify and self.on_failure:
                        self.on_failure(rank)
                self._stop_event.wait(max(0, self.interval - (time.monotonic() - started)))
