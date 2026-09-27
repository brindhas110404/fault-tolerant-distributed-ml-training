"""Synchronous training with coordinator-owned checkpoints and global rollback.

Workers stage updates. A checkpoint advances only after every commit acknowledges.
An uncertain commit is rolled back to the previous checkpoint on all survivors.
"""
import argparse
from concurrent import futures
import json
from pathlib import Path
import threading
import time
import uuid

import grpc
import numpy as np
from communication import protocol_pb2 as pb, protocol_pb2_grpc as rpc
from worker.training import decode_state, state_digest

OPTIONS = [('grpc.max_receive_message_length', 32 * 1024 * 1024),
           ('grpc.max_send_message_length', 32 * 1024 * 1024)]


class CoordinatorService(rpc.CoordinatorServiceServicer):
    def __init__(self, coordinator):
        self.coordinator = coordinator

    def RegisterWorker(self, request, context):
        try:
            self.coordinator.register_worker(request.worker_id, request.endpoint)
            return pb.RegisterWorkerResponse(accepted=True, message='worker registered')
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

    def StartTraining(self, request, context):
        try:
            # Hold through response construction: concurrent callers cannot mix metadata.
            with self.coordinator._request_lock:
                losses = self.coordinator.start_training(request.steps, request.batch_size)
                return pb.TrainingResponse(accepted=True, message='training completed',
                    completed_steps=len(losses), mean_losses=losses, **self.coordinator.last_run)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except (RuntimeError, grpc.RpcError, OSError) as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

    def GetClusterStatus(self, request, context):
        registered = self.coordinator.registered_workers()
        return pb.ClusterStatusResponse(expected_workers=self.coordinator.workers,
            registered_workers=registered, ready=len(registered) == self.coordinator.workers)


class Coordinator:
    def __init__(self, workers, endpoints, dataset='digits', checkpoint_path=None, rpc_timeout=15):
        if workers <= 0 or len(endpoints) != workers:
            raise ValueError('one worker endpoint is required for each worker')
        self.workers = workers
        self.endpoints = dict(enumerate(endpoints))
        self.dataset = dataset
        self._registered = {}
        self._lock = threading.Lock()
        self._training_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self.checkpoint = None
        self.total_steps = 0
        self.batch_size = None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.rpc_timeout = rpc_timeout
        self.last_run = {}
        self.events = []
        self._channels = {}
        self._known_digests = {}
        self.profile = {}
        if self.checkpoint_path and self.checkpoint_path.exists():
            with np.load(self.checkpoint_path, allow_pickle=False) as saved:
                if str(saved['dataset']) != dataset:
                    raise ValueError('checkpoint dataset mismatch')
                self.checkpoint = saved['state'].tobytes()
                self.total_steps = int(saved['step'])
                self.batch_size = int(saved['batch_size'])
                decode_state(self.checkpoint)

    def register_worker(self, worker_id, endpoint):
        if self.endpoints.get(worker_id) != endpoint:
            raise ValueError('unknown worker ID or unexpected endpoint')
        with self._lock:
            self._registered[worker_id] = endpoint

    def registered_workers(self):
        with self._lock:
            return sorted(self._registered)

    def _rpc(self, rank, method, request, timeout=None):
        with self._lock:
            if rank not in self._channels:
                self._channels[rank] = grpc.insecure_channel(self.endpoints[rank], options=OPTIONS)
            channel = self._channels[rank]
        try:
            return getattr(rpc.WorkerServiceStub(channel), method)(request, timeout=timeout or self.rpc_timeout)
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.UNAVAILABLE:
                # A restarted endpoint must not remain hidden by reconnect backoff.
                with self._lock:
                    if self._channels.get(rank) is channel:
                        self._channels.pop(rank)
                channel.close()
            raise

    def close(self):
        for channel in self._channels.values():
            channel.close()
        self._channels.clear()

    def _parallel(self, members, method, make_request):
        phase_started = time.perf_counter()
        results, errors = {}, {}
        with futures.ThreadPoolExecutor(max_workers=len(members)) as pool:
            pending = {pool.submit(self._rpc, rank, method, make_request(rank)): rank for rank in members}
            abort_sent = False
            for future in futures.as_completed(pending):
                rank = pending[future]
                try:
                    results[rank] = future.result()
                except Exception as exc:
                    errors[rank] = exc
                    if method == 'TrainStep' and not abort_sent:
                        # Signal before waiting for peers blocked inside AllReduce.
                        self._abort(members, self._token)
                        abort_sent = True
        self.profile[method + '_seconds'] = self.profile.get(method + '_seconds', 0) + time.perf_counter() - phase_started
        if method == 'TrainStep' and results:
            for field in ('compute_time', 'communication_time'):
                key = 'mean_worker_' + field + '_seconds'
                self.profile[key] = self.profile.get(key, 0) + sum(getattr(r, field) for r in results.values()) / len(results)
        return results, errors

    def _abort(self, members, token):
        with futures.ThreadPoolExecutor(max_workers=len(members)) as pool:
            jobs = [pool.submit(self._rpc, r, 'AbortStep', pb.StepControl(token=token), 2) for r in members]
            for job in jobs:
                try:
                    job.result()
                except grpc.RpcError:
                    pass

    def _live(self, members):
        results, _ = self._parallel(members, 'Ping', lambda _: pb.PingRequest())
        return sorted(rank for rank, result in results.items() if result.ok)

    def _save(self, state, step):
        save_started = time.perf_counter()
        if self.checkpoint_path:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.checkpoint_path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, state=np.frombuffer(state, dtype=np.uint8), step=step, dataset=self.dataset, batch_size=self.batch_size)
                stream.flush()
                import os
                os.fsync(stream.fileno())
            temporary.replace(self.checkpoint_path)
        self.checkpoint, self.total_steps = state, step
        self.profile["checkpoint_seconds"] = self.profile.get("checkpoint_seconds", 0) + time.perf_counter() - save_started

    def start_training(self, steps, batch_size):
        if steps <= 0 or batch_size <= 0:
            raise ValueError('steps and batch_size must be positive')
        with self._training_lock:
            self.profile = {}
            if self.batch_size is not None and batch_size != self.batch_size:
                raise ValueError("global batch size must match the checkpoint; use a new checkpoint to change it")
            self._known_digests.clear()
            members = self.registered_workers()
            if self.checkpoint is None and len(members) != self.workers:
                raise RuntimeError('not all workers have registered')
            members = self._live(members)
            if not members:
                raise RuntimeError('no live registered workers')
            if batch_size < len(members):
                raise ValueError('global batch size must be at least active workers')
            if self.checkpoint is None:
                initial = self._rpc(members[0], 'GetState', pb.PingRequest())
                if initial.dataset != self.dataset:
                    raise ValueError('dataset mismatch')
                self.batch_size = batch_size
                self._save(initial.state, 0)
            losses, recoveries = [], 0
            started = time.perf_counter()
            for _ in range(steps):
                for attempt in range(4):
                    self._token = uuid.uuid4().hex
                    token = self._token
                    endpoints = [self.endpoints[r] for r in members]
                    def config(rank):
                        return pb.ConfigureRequest(token=token, rank=members.index(rank),
                            endpoints=endpoints, state=b"" if rank in self._known_digests else self.checkpoint,
                            dataset=self.dataset, expected_digest=self._known_digests.get(rank, ""))
                    _, errors = self._parallel(members, 'Configure', config)
                    results = {}
                    if not errors:
                        results, errors = self._parallel(members, 'TrainStep', lambda _: pb.TrainStepRequest(
                            token=token, step=self.total_steps, batch_size=batch_size))
                    if not errors:
                        reference = decode_state(results[members[0]].state)
                        for rank, result in results.items():
                            state = decode_state(result.state)
                            if result.digest != state_digest(state) or not all(
                                np.allclose(a, b, rtol=1e-10, atol=1e-12)
                                for a, b in zip(reference, state)
                            ) or sum(x.samples for x in results.values()) != batch_size:
                                errors[rank] = RuntimeError('workers disagree on staged model or sample count')
                    if not errors:
                        _, errors = self._parallel(members, 'CommitStep', lambda _: pb.StepControl(token=token))
                    if not errors:
                        candidate = results[members[0]].state
                        try:
                            self._save(candidate, self.total_steps + 1)
                        except OSError as exc:
                            self._abort(members, token)
                            _, rollback_errors = self._parallel(members, 'Configure', lambda r: pb.ConfigureRequest(
                                token=uuid.uuid4().hex, rank=members.index(r), endpoints=endpoints,
                                state=self.checkpoint, dataset=self.dataset))
                            raise RuntimeError('checkpoint write failed; training stopped; rollback errors: '
                                               + str(rollback_errors)) from exc
                        self._known_digests = {rank: result.digest for rank, result in results.items()}
                        losses.append(sum(r.loss * r.samples for r in results.values()) / batch_size)
                        break
                    self._known_digests.clear()
                    self._abort(members, token)
                    recoveries += 1
                    self.events.append({'step': self.total_steps, 'attempt': attempt,
                                        'errors': {str(r): str(e) for r, e in errors.items()}})
                    self.events = self.events[-1000:]
                    members = self._live(members)
                    if not members:
                        raise RuntimeError('all workers unavailable; last checkpoint retained')
                    # Restore even after the final failed attempt, including partial commits.
                    rollback = uuid.uuid4().hex
                    _, restore_errors = self._parallel(members, 'Configure', lambda r: pb.ConfigureRequest(
                        token=rollback, rank=members.index(r), endpoints=[self.endpoints[x] for x in members],
                        state=self.checkpoint, dataset=self.dataset))
                    if restore_errors or attempt == 3:
                        raise RuntimeError('recovery exhausted; checkpoint retained; retry after worker repair')
                else:
                    raise RuntimeError('training retry budget exhausted')
            self.last_run = dict(recoveries=recoveries, active_workers=members,
                elapsed_seconds=time.perf_counter() - started, total_steps=self.total_steps,
                model_digest=state_digest(decode_state(self.checkpoint)))
            print(json.dumps({"training_profile": self.profile}), flush=True)
            return losses

    def start(self, host='[::]', port=50050):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=16), options=OPTIONS)
        rpc.add_CoordinatorServiceServicer_to_server(CoordinatorService(self), server)
        if server.add_insecure_port(f'{host}:{port}') == 0:
            raise RuntimeError('could not bind coordinator')
        server.start()
        print(f'Coordinator running for {self.workers} workers', flush=True)
        return server


def parse_endpoints(value, workers):
    endpoints = [x.strip() for x in value.split(',')]
    if len(endpoints) != workers or any(not x for x in endpoints):
        raise ValueError('--worker-endpoints must provide one host:port value per worker')
    return endpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--host', default='[::]')
    parser.add_argument('--port', type=int, default=50050)
    parser.add_argument('--worker-endpoints', required=True)
    parser.add_argument('--dataset', choices=['digits', 'cifar10'], default='digits')
    parser.add_argument('--checkpoint', default='results/checkpoint.npz')
    args = parser.parse_args()
    coordinator = Coordinator(args.workers, parse_endpoints(args.worker_endpoints, args.workers),
                              args.dataset, args.checkpoint)
    server = coordinator.start(args.host, args.port)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop(0).wait()
        coordinator.close()

if __name__ == '__main__':
    main()
