from concurrent import futures

import grpc

from communication import protocol_pb2
from communication import protocol_pb2_grpc

from communication.serialization import (
    pack_array,
    unpack_array
)


class WorkerService(
    protocol_pb2_grpc.WorkerServiceServicer
):

    def __init__(self, worker):

        self.worker = worker

    # =========================================================
    # SendTensor
    # =========================================================

    def SendTensor(
        self,
        request,
        context
    ):

        try:

            if request.request_only:

                result = self.worker.get_received_tensor(
                    op_id=request.op_id,
                    source=request.source,
                    phase=request.phase,
                    round_id=request.round,
                    chunk=request.chunk
                )

                if result is None:

                    context.set_code(
                        grpc.StatusCode.NOT_FOUND
                    )

                    context.set_details(
                        "Requested tensor has not arrived yet"
                    )

                    return protocol_pb2.TensorResponse()

                return protocol_pb2.TensorResponse(
                    payload=pack_array(result),
                    shape=list(result.shape)
                )

            array = unpack_array(
                request.payload
            )

            result = self.worker.receive_tensor(
                source=request.source,
                round_id=request.round,
                phase=request.phase,
                chunk=request.chunk,
                array=array,
                op_id=request.op_id
            )

            return protocol_pb2.TensorResponse(
                payload=pack_array(result),
                shape=list(result.shape)
            )

        except LookupError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except Exception as exc:

            context.set_code(
                grpc.StatusCode.INTERNAL
            )

            context.set_details(
                str(exc)
            )

            return protocol_pb2.TensorResponse()

    # =========================================================
    # Ping
    # =========================================================

    def Ping(
        self,
        request,
        context
    ):

        return protocol_pb2.PingResponse(ok=True)

    # =========================================================
    # TrainStep
    # =========================================================

    def TrainStep(
        self,
        request,
        context
    ):

        try:
            result = (self.worker.train_transaction(request) if request.token
                      else self.worker.train_next_batch(request.batch_size))

            return protocol_pb2.TrainStepResponse(
                loss=result["loss"],
                compute_time=result["compute_time"],
                communication_time=result["communication_time"],
                state=result.get("state", b""), digest=result.get("digest", ""),
                samples=result.get("samples", 0),
            )

        except Exception as exc:

            context.set_code(
                grpc.StatusCode.INTERNAL
            )

            context.set_details(str(exc))

            return protocol_pb2.TrainStepResponse()

    def Configure(self, request, context):
        try:
            self.worker.configure_training(request)
            return protocol_pb2.Ack(ok=True)
        except Exception as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

    def AbortStep(self, request, context):
        self.worker.abort_training(request.token)
        return protocol_pb2.Ack(ok=True)

    def CommitStep(self, request, context):
        try:
            self.worker.commit_training(request.token)
            return protocol_pb2.Ack(ok=True)
        except Exception as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

    def GetState(self, request, context):
        state, digest = self.worker.get_state()
        return protocol_pb2.StateResponse(state=state, digest=digest, dataset=self.worker.dataset)


# =============================================================
# Start gRPC server
# =============================================================

def serve(
    worker,
    host,
    port
):

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=20
        ),
        options=[("grpc.max_receive_message_length", 32 * 1024 * 1024),
                 ("grpc.max_send_message_length", 32 * 1024 * 1024)]
    )

    protocol_pb2_grpc.add_WorkerServiceServicer_to_server(
        WorkerService(worker),
        server
    )

    bound_port = server.add_insecure_port(
        f"{host}:{port}"
    )

    if bound_port == 0:
        raise RuntimeError(
            f"Could not bind gRPC server to {host}:{port}"
        )

    server.start()

    return server
