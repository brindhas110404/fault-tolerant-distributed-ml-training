"""Evaluate a live Docker worker against the held-out dataset and save a report."""
import argparse
import json
from pathlib import Path
import grpc
from communication import protocol_pb2 as pb, protocol_pb2_grpc as rpc
from coordinator.coordinator import OPTIONS
from experiments.docker_benchmark import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', default='worker0:50060')
    parser.add_argument('--dataset', choices=['digits', 'cifar10'], default='cifar10')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--out')
    args = parser.parse_args()
    with grpc.insecure_channel(args.worker, options=OPTIONS) as channel:
        state = rpc.WorkerServiceStub(channel).GetState(pb.PingRequest(), timeout=30)
    if state.dataset != args.dataset:
        raise ValueError('evaluation dataset differs from worker dataset')
    result = dict(dataset=args.dataset, model_digest=state.digest,
                  **evaluate(state.state, args.dataset, args.data_dir))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
