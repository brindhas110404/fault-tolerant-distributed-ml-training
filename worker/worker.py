import argparse
import time
import threading

import grpc
import numpy as np

from worker.heartbeat import HeartbeatMonitor
from worker.exceptions import WorkerFailedError

from worker.model import (
    MLP,
    softmax_cross_entropy
)

from worker.optimizer import SGD
from worker.dataset import load_data, shard_dataset

from communication.grpc_server import serve
from communication.ring_allreduce import RingAllReduce
from communication import protocol_pb2
from communication import protocol_pb2_grpc


from worker.training import TrainingTransactions


class DistributedWorker(TrainingTransactions):

    def __init__(
        self,
        worker_id,
        world_size,
        endpoints,
        input_dim=64,
        hidden_dim=128,
        num_classes=10,
        dataset="digits",
        data_dir="data"
    ):

        self.dataset = dataset
        self.data_dir = data_dir
        if dataset == "cifar10":
            input_dim = 3072
        self._attempt = ""
        self._aborted = threading.Event()
        self._pending = None
        self._committed_token = None
        self.worker_id = worker_id
        self.world_size = world_size
        self.endpoints = endpoints

        # -----------------------------------------------------
        # Heartbeat monitor
        # -----------------------------------------------------

        self.heartbeat = HeartbeatMonitor(
            worker_id=worker_id,
            world_size=world_size,
            endpoints=endpoints,
            interval=1.0,
            max_failures=3,
            ping_timeout=0.5,
            on_failure=self._heartbeat_failure
        )

        self._peer_channels = {}

        # -----------------------------------------------------
        # Model
        # -----------------------------------------------------

        self.model = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            seed=7
        )

        # -----------------------------------------------------
        # Optimizer
        # -----------------------------------------------------

        self.optimizer = SGD(
            self.model.parameters(),
            learning_rate=0.05
        )

        # -----------------------------------------------------
        # AllReduce state
        # -----------------------------------------------------

        self._allreduce_ops = {}
        self._allreduce_counter = 0
        self._received_tensors = {}
        self._allreduce_finished = {}
        self._lock = threading.Lock()
        self._tensor_condition = threading.Condition(self._lock)
        self._training_lock = threading.Lock()
        self._training_data = None
        self._next_batch_start = 0

        # -----------------------------------------------------
        # Ring
        # -----------------------------------------------------

        self.ring = RingAllReduce(
            rank=worker_id,
            world_size=world_size,
            endpoints=endpoints,
            timeout=60,
            worker=self
        )

        self.heartbeat.start()

    def _heartbeat_failure(self, stable_id):
        with self._lock:
            if self._attempt and stable_id in getattr(self, '_managed_peer_ids', {}).values():
                self._aborted.set()
                self._tensor_condition.notify_all()

    def close(self):
        self.heartbeat.stop()
        for channel in self._peer_channels.values():
            channel.close()
        self._peer_channels.clear()

    def wait_received_tensor(self, op_id, source, phase, round_id, chunk, timeout):
        key = (op_id, source, phase, round_id, chunk)
        with self._tensor_condition:
            self._tensor_condition.wait_for(
                lambda: key in self._received_tensors or self._aborted.is_set(), timeout=timeout)
            value = self._received_tensors.get(key)
            return None if value is None else value.copy()

    # =========================================================
    # START ALLREDUCE
    # =========================================================

    def start_allreduce(
        self,
        chunks
    ):

        with self._lock:

            op_id = (
                f"{self._attempt}/allreduce-{self._allreduce_counter}"
            )

            self._allreduce_counter += 1

            self._allreduce_ops[op_id] = [
                np.asarray(
                    chunk,
                    dtype=np.float64
                ).copy()
                for chunk in chunks
            ]

            self._allreduce_finished[op_id] = False

            return op_id

    # =========================================================
    # GET CHUNK
    # =========================================================

    def get_chunk(
        self,
        op_id,
        chunk
    ):

        with self._lock:

            if op_id not in self._allreduce_ops:

                raise LookupError(
                    f"Unknown AllReduce operation: {op_id}"
                )

            return self._allreduce_ops[
                op_id
            ][chunk].copy()

    # =========================================================
    # SET CHUNK
    # =========================================================

    def set_chunk(
        self,
        op_id,
        chunk,
        array
    ):

        with self._lock:

            if op_id not in self._allreduce_ops:

                raise LookupError(
                    f"Unknown AllReduce operation: {op_id}"
                )

            self._allreduce_ops[
                op_id
            ][chunk] = np.asarray(
                array,
                dtype=np.float64
            ).copy()

    # =========================================================
    # RECEIVE TENSOR
    # =========================================================

    def receive_tensor(
        self,
        source,
        round_id,
        phase,
        chunk,
        array,
        op_id
    ):

        with self._lock:

            if op_id not in self._allreduce_ops:

                raise LookupError(
                    f"Unknown AllReduce operation: {op_id}"
                )

            if self._attempt and (self._aborted.is_set() or not op_id.startswith(self._attempt + "/")):
                raise RuntimeError("stale or aborted operation")
            key = (op_id, source, phase, round_id, chunk)
            if key in self._received_tensors:
                return self._received_tensors[key].copy()

            array = np.asarray(
                array,
                dtype=np.float64
            ).copy()

            if phase == 0:

                current = self._allreduce_ops[
                    op_id
                ][chunk]

                reduced = current + array

                self._allreduce_ops[
                    op_id
                ][chunk] = reduced.copy()

                key = (
                    op_id,
                    source,
                    phase,
                    round_id,
                    chunk
                )

                self._received_tensors[key] = (
                    reduced.copy()
                )

                self._tensor_condition.notify_all()
                return reduced.copy()

            if phase == 1:

                self._allreduce_ops[
                    op_id
                ][chunk] = array.copy()

                key = (
                    op_id,
                    source,
                    phase,
                    round_id,
                    chunk
                )

                self._received_tensors[key] = (
                    array.copy()
                )

                self._tensor_condition.notify_all()
                return array.copy()

            raise RuntimeError(
                f"Invalid AllReduce phase: {phase}"
            )

    # =========================================================
    # GET RECEIVED TENSOR
    # =========================================================

    def get_received_tensor(
        self,
        op_id,
        source,
        phase,
        round_id,
        chunk
    ):

        key = (
            op_id,
            source,
            phase,
            round_id,
            chunk
        )

        with self._lock:

            value = self._received_tensors.get(key)

            if value is None:
                return None

            return value.copy()

    # =========================================================
    # FINISH ALLREDUCE
    # =========================================================

    def finish_allreduce(
        self,
        op_id
    ):

        with self._lock:

            if op_id not in self._allreduce_ops:
                return

            self._allreduce_finished[op_id] = True

    # =========================================================
    # HANDLE FAILURE
    #
    # Called when a WorkerFailedError is caught during
    # AllReduce. Removes the dead worker and re-forms the
    # ring with the surviving workers.
    # =========================================================

    def handle_failure(
        self,
        failed_rank
    ):

        print(
            f"[Recovery] Worker {self.worker_id}: "
            f"handling failure of worker {failed_rank}"
        )

        with self._lock:

            # -------------------------------------------------
            # Remove dead worker from endpoints.
            # -------------------------------------------------

            new_endpoints_full = {
                rank: addr
                for rank, addr in self.endpoints.items()
                if rank != failed_rank
            }

            # -------------------------------------------------
            # Re-map ranks sequentially starting from 0.
            #
            # Example: workers 0, 1, 2 — worker 1 dies.
            # Survivors: 0, 2.
            # New ranks: 0 -> 0, 2 -> 1.
            # -------------------------------------------------

            surviving_ranks = sorted(
                new_endpoints_full.keys()
            )

            rank_map = {
                old: new
                for new, old in enumerate(surviving_ranks)
            }

            new_endpoints = {
                rank_map[old]: addr
                for old, addr in new_endpoints_full.items()
            }

            new_rank = rank_map[self.worker_id]
            new_world_size = len(surviving_ranks)

            # Do not clear the old operation state here.
            #
            # A failure is normally discovered at different times by
            # different workers.  A surviving peer can therefore still
            # be delivering a message from the failed operation while
            # this worker is reforming its ring.  Clearing this state
            # made that valid in-flight RPC fail with "Unknown AllReduce
            # operation", preventing the peer from reaching recovery.
            #
            # The retry gets a new operation id, so retaining the old
            # state cannot be confused with the reformed-ring operation.
            # It is deliberately kept until a coordinated cleanup point
            # is available.

        self.reform_ring(
            new_rank=new_rank,
            new_world_size=new_world_size,
            new_endpoints=new_endpoints
        )

        print(
            f"[Recovery] Worker {self.worker_id}: "
            f"ring reformed. New rank={new_rank}, "
            f"world_size={new_world_size}"
        )

    # =========================================================
    # REFORM RING
    #
    # Rebuilds the RingAllReduce with surviving workers.
    # =========================================================

    def reform_ring(
        self,
        new_rank,
        new_world_size,
        new_endpoints
    ):

        self.world_size = new_world_size
        self.endpoints = new_endpoints

        self.ring = RingAllReduce(
            rank=new_rank,
            world_size=new_world_size,
            endpoints=new_endpoints,
            timeout=60,
            worker=self
        )

    # =========================================================
    # TRAIN
    # =========================================================

    def train_batch(
        self,
        X,
        y
    ):

        compute_start = time.perf_counter()

        logits = self.model.forward(X)

        loss, gradient = softmax_cross_entropy(
            logits,
            y
        )

        self.model.backward(gradient)

        compute_time = (
            time.perf_counter()
            - compute_start
        )

        communication_start = time.perf_counter()

        local_gradients = self.model.gradients()

        averaged_gradients = []

        for gradient in local_gradients:

            try:

                averaged_gradient = self.ring.allreduce(
                    gradient
                )

            except WorkerFailedError as exc:

                print(
                    f"[Train] Worker {self.worker_id}: "
                    f"worker {exc.worker_id} failed "
                    f"during AllReduce — triggering recovery"
                )

                self.handle_failure(exc.worker_id)

                # -----------------------------------------
                # Retry AllReduce with reformed ring.
                # -----------------------------------------

                averaged_gradient = self.ring.allreduce(
                    gradient
                )

            averaged_gradients.append(averaged_gradient)

        communication_time = (
            time.perf_counter()
            - communication_start
        )

        self.optimizer.step(averaged_gradients)

        return {
            "loss": loss,
            "compute_time": compute_time,
            "communication_time": communication_time
        }

    def train_next_batch(self, batch_size):
        """Train one local, deterministic data-parallel batch.

        Each worker loads the same source dataset but trains on its own shard.
        The coordinator invokes this method concurrently on every worker, which
        lets the existing Ring AllReduce synchronize each parameter gradient.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        with self._training_lock:
            if self._training_data is None:
                X_train, _, y_train, _ = load_data()
                self._training_data = shard_dataset(
                    X_train,
                    y_train,
                    self.worker_id,
                    self.world_size,
                )

            X_shard, y_shard = self._training_data
            indices = (
                np.arange(batch_size) + self._next_batch_start
            ) % len(X_shard)
            self._next_batch_start = (
                self._next_batch_start + batch_size
            ) % len(X_shard)

            return self.train_batch(X_shard[indices], y_shard[indices])

    # =========================================================
    # IS WORKER ALIVE
    # =========================================================

    def is_worker_alive(self, rank):
        return self.heartbeat.is_alive(
            self._managed_peer_ids[rank] if self._attempt else rank)

    def get_dead_workers(self):
        return self.heartbeat.get_dead_workers()


def parse_endpoints(value, world_size):
    """Parse a rank-ordered comma-separated endpoint list."""
    endpoints = [endpoint.strip() for endpoint in value.split(",")]

    if len(endpoints) != world_size or any(not endpoint for endpoint in endpoints):
        raise ValueError(
            "--endpoints must provide exactly one host:port value "
            "for each worker"
        )

    return dict(enumerate(endpoints))


def register_with_coordinator(worker_id, endpoint, coordinator_endpoint):
    """Register a worker after its service is listening."""
    channel = grpc.insecure_channel(coordinator_endpoint)
    stub = protocol_pb2_grpc.CoordinatorServiceStub(channel)

    try:
        for attempt in range(20):
            try:
                response = stub.RegisterWorker(
                    protocol_pb2.RegisterWorkerRequest(
                        worker_id=worker_id,
                        endpoint=endpoint,
                    ),
                    timeout=2,
                )
                if response.accepted:
                    print(f"Worker {worker_id} registered with coordinator")
                    return
                raise RuntimeError(response.message)
            except grpc.RpcError:
                if attempt == 19:
                    raise
                time.sleep(0.5)
    finally:
        channel.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run a distributed-training gRPC worker."
    )
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--endpoints",
        required=True,
        help="Comma-separated endpoints in rank order, for example worker0:50060,worker1:50061",
    )
    parser.add_argument(
        "--coordinator",
        help="Optional coordinator host:port used for worker registration.",
    )
    parser.add_argument("--dataset", choices=["digits", "cifar10"], default="digits")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    if not 0 <= args.worker_id < args.world_size:
        parser.error("--worker-id must be between 0 and --world-size - 1")

    try:
        endpoints = parse_endpoints(args.endpoints, args.world_size)
    except ValueError as exc:
        parser.error(str(exc))

    worker = DistributedWorker(
        worker_id=args.worker_id,
        world_size=args.world_size,
        endpoints=endpoints,
        dataset=args.dataset,
        data_dir=args.data_dir,
    )
    server = serve(worker, args.host, args.port)
    print(f"Worker {args.worker_id} listening on {args.host}:{args.port}")

    if args.coordinator:
        register_with_coordinator(
            args.worker_id,
            endpoints[args.worker_id],
            args.coordinator,
        )

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        worker.close()
        server.stop(0).wait()


if __name__ == "__main__":
    main()
