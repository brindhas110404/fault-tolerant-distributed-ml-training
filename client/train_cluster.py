import argparse
import json
from pathlib import Path
import time
import grpc
from google.protobuf.json_format import MessageToDict
from communication import protocol_pb2 as pb, protocol_pb2_grpc as rpc


def train(endpoint, steps, batch_size, wait_seconds=60):
    with grpc.insecure_channel(endpoint) as channel:
        stub = rpc.CoordinatorServiceStub(channel)
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                status = stub.GetClusterStatus(pb.ClusterStatusRequest(), timeout=2)
                if status.ready:
                    break
            except grpc.RpcError:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError('workers did not register before readiness deadline')
            time.sleep(0.25)
        result = stub.StartTraining(pb.TrainingRequest(steps=steps, batch_size=batch_size),
                                   timeout=max(120, steps * 60))
        return MessageToDict(result, preserving_proto_field_name=True,
                             always_print_fields_with_no_presence=True)


def main():
    parser = argparse.ArgumentParser(description='Run synchronized training. Batch size is GLOBAL.')
    parser.add_argument('--coordinator', default='127.0.0.1:50050')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out')
    args = parser.parse_args()
    result = train(args.coordinator, args.steps, args.batch_size)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
