"""Measure failure notification latency with real gRPC and production defaults."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import statistics
import threading
import time
import grpc
from communication import protocol_pb2 as pb, protocol_pb2_grpc as rpc
from worker.heartbeat import HeartbeatMonitor


class Probe(rpc.WorkerServiceServicer):
    def __init__(self):
        self.stall = threading.Event()
        self.seen = threading.Event()

    def Ping(self, request, context):
        self.seen.set()
        while self.stall.is_set() and context.is_active():
            time.sleep(.01)
        return pb.PingResponse(ok=not self.stall.is_set())


def trial(mode, phase=0.0):
    probes, servers, endpoints = [], [], {0: '127.0.0.1:1'}
    for rank in (1, 2):
        probe = Probe()
        server = grpc.server(ThreadPoolExecutor(max_workers=4))
        rpc.add_WorkerServiceServicer_to_server(probe, server)
        port = server.add_insecure_port('127.0.0.1:0')
        server.start()
        probes.append(probe); servers.append(server)
        endpoints[rank] = f'127.0.0.1:{port}'
    notifications = {}
    monitor = HeartbeatMonitor(0, 3, endpoints, on_failure=lambda r: notifications.setdefault(r, time.monotonic()))
    try:
        monitor.start()
        for probe in probes:
            assert probe.seen.wait(3), 'healthy ping not observed'
        time.sleep(.1 + phase)
        assert not monitor.get_dead_workers(), 'false failure during healthy startup'
        started = time.monotonic()
        targets = [1, 2] if mode == 'simultaneous_timeouts' else [1]
        if mode == 'server_stop':
            servers[0].stop(0).wait()
        else:
            for rank in targets:
                probes[rank - 1].stall.set()
        deadline = started + 8
        while any(r not in notifications for r in targets) and time.monotonic() < deadline:
            time.sleep(.01)
        detected = {str(rank): notifications.get(rank, float('inf')) - started for rank in targets}
        assert max(detected.values()) < 8, f'detection target missed: {detected}'
        if 2 not in targets:
            assert monitor.is_alive(2), 'healthy peer incorrectly declared dead'
        return dict(mode=mode, phase_seconds=phase, detection_seconds=detected)
    finally:
        monitor.stop()
        for server in servers:
            server.stop(0).wait()


def run(repeats=3, output='results/heartbeat_validation.json'):
    trials = [trial(mode, (i % 3) * .35) for mode in
              ('server_stop', 'rpc_timeout', 'simultaneous_timeouts') for i in range(repeats)]
    values = [value for item in trials for value in item['detection_seconds'].values()]
    report = dict(passed=True, target_seconds=8, interval_seconds=1, ping_timeout_seconds=.5,
                  consecutive_failures=3, trials=trials, maximum_seconds=max(values),
                  median_seconds=statistics.median(values), notifications=len(values),
                  scope='local real gRPC; fail-stop and RPC timeout scenarios; includes simultaneous peer failures')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--out', default='results/heartbeat_validation.json')
    args = parser.parse_args()
    run(args.repeats, args.out)
