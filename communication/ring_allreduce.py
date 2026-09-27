import time

import grpc
import numpy as np

from communication import protocol_pb2
from communication import protocol_pb2_grpc

from communication.serialization import (
    pack_array,
    unpack_array
)

from worker.exceptions import WorkerFailedError


class RingAllReduce:

    def __init__(
        self,
        rank,
        world_size,
        endpoints,
        timeout=60,
        worker=None
    ):

        self.rank = rank
        self.world_size = world_size
        self.endpoints = endpoints
        self.timeout = timeout
        self.worker = worker

    # =========================================================
    # gRPC stub
    # =========================================================

    def _get_stub(
        self,
        rank
    ):

        endpoint = self.endpoints[rank]
        if endpoint not in self.worker._peer_channels:
            self.worker._peer_channels[endpoint] = grpc.insecure_channel(endpoint)
        return protocol_pb2_grpc.WorkerServiceStub(self.worker._peer_channels[endpoint])

    # =========================================================
    # SEND
    # =========================================================

    def _send(
        self,
        destination,
        chunk,
        array,
        phase,
        round_id,
        op_id
    ):

        # -----------------------------------------------------
        # Check heartbeat before even trying to send.
        # -----------------------------------------------------

        if (
            self.worker is not None
            and not self.worker.is_worker_alive(destination)
        ):
            raise WorkerFailedError(
                destination,
                reason="detected dead before send"
            )

        array = np.asarray(
            array,
            dtype=np.float64
        )

        request = protocol_pb2.TensorMessage(
            source=self.rank,
            round=round_id,
            phase=phase,
            chunk=chunk,
            payload=pack_array(array),
            shape=list(array.shape),
            op_id=op_id,
            request_only=False
        )

        deadline = time.monotonic() + self.timeout
        while True:
            if self.worker._aborted.is_set():
                raise RuntimeError("AllReduce aborted")
            try:
                response = self._get_stub(destination).SendTensor(
                    request, timeout=max(0.01, deadline - time.monotonic()))
                break
            except grpc.RpcError as exc:
                # Early arrivals are safe to retry: NOT_FOUND has made no mutation.
                if exc.code() == grpc.StatusCode.NOT_FOUND and time.monotonic() < deadline:
                    time.sleep(0.002)
                    continue
                if exc.code() == grpc.StatusCode.UNAVAILABLE:
                    channel = self.worker._peer_channels.pop(self.endpoints[destination], None)
                    if channel is not None:
                        channel.close()
                    raise WorkerFailedError(destination, reason=str(exc)) from exc
                raise

        if not response.payload:

            raise RuntimeError(
                f"Worker {self.rank} received an empty "
                f"response from worker {destination}"
            )

        return unpack_array(
            response.payload
        )

    # =========================================================
    # REQUEST EXACT RECEIVED CHUNK
    # =========================================================

    def _request_chunk(
        self,
        source,
        chunk,
        phase,
        round_id,
        op_id,
        retry_delay=0.01
    ):

        deadline = time.monotonic() + self.timeout

        while time.monotonic() < deadline:

            if self.worker._aborted.is_set():
                raise RuntimeError("AllReduce aborted")

            # -------------------------------------------------
            # Check heartbeat — if source is dead, stop waiting.
            # -------------------------------------------------

            if (
                self.worker is not None
                and not self.worker.is_worker_alive(source)
            ):
                raise WorkerFailedError(
                    source,
                    reason="detected dead while waiting for chunk"
                )

            result = self.worker.wait_received_tensor(
                op_id=op_id,
                source=source,
                phase=phase,
                round_id=round_id,
                chunk=chunk,
                timeout=min(0.1, max(0, deadline - time.monotonic()))
            )

            if result is not None:
                return result


        raise RuntimeError(
            f"Timed out waiting for "
            f"worker {source}, "
            f"phase={phase}, "
            f"round={round_id}, "
            f"chunk={chunk}, "
            f"op_id={op_id}"
        )

    # =========================================================
    # ALL REDUCE
    # =========================================================

    def allreduce(
        self,
        tensor
    ):

        tensor = np.asarray(
            tensor,
            dtype=np.float64
        ).copy()

        if self.world_size == 1:
            return tensor

        chunks = np.array_split(
            tensor,
            self.world_size,
            axis=0
        )

        chunks = [
            np.asarray(
                chunk,
                dtype=np.float64
            ).copy()
            for chunk in chunks
        ]

        op_id = self.worker.start_allreduce(
            chunks
        )

        next_rank = (
            self.rank + 1
        ) % self.world_size

        previous_rank = (
            self.rank - 1
        ) % self.world_size

        # =====================================================
        # REDUCE-SCATTER
        # =====================================================

        for step in range(
            self.world_size - 1
        ):

            send_chunk = (
                self.rank - step
            ) % self.world_size

            receive_chunk = (
                self.rank - step - 1
            ) % self.world_size

            send_array = self.worker.get_chunk(
                op_id,
                send_chunk
            )

            self._send(
                destination=next_rank,
                chunk=send_chunk,
                array=send_array,
                phase=0,
                round_id=step,
                op_id=op_id
            )

            reduced = self._request_chunk(
                source=previous_rank,
                chunk=receive_chunk,
                phase=0,
                round_id=step,
                op_id=op_id
            )

            self.worker.set_chunk(
                op_id,
                receive_chunk,
                reduced
            )

        # =====================================================
        # ALL-GATHER
        # =====================================================

        for step in range(
            self.world_size - 1
        ):

            send_chunk = (
                self.rank + 1 - step
            ) % self.world_size

            receive_chunk = (
                self.rank - step
            ) % self.world_size

            send_array = self.worker.get_chunk(
                op_id,
                send_chunk
            )

            self._send(
                destination=next_rank,
                chunk=send_chunk,
                array=send_array,
                phase=1,
                round_id=step,
                op_id=op_id
            )

            gathered = self._request_chunk(
                source=previous_rank,
                chunk=receive_chunk,
                phase=1,
                round_id=step,
                op_id=op_id
            )

            self.worker.set_chunk(
                op_id,
                receive_chunk,
                gathered
            )

        # =====================================================
        # RECONSTRUCT
        # =====================================================

        result_chunks = []

        for i in range(self.world_size):

            result_chunks.append(
                self.worker.get_chunk(
                    op_id,
                    i
                )
            )

        result = np.concatenate(
            result_chunks,
            axis=0
        )

        result = result / self.world_size

        self.worker.finish_allreduce(
            op_id
        )

        return result