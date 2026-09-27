"""Coordinator-owned training transactions; stable worker IDs, attempt-scoped rings."""
import hashlib
import io
import time
import numpy as np
from communication.ring_allreduce import RingAllReduce
from worker.dataset import load_dataset
from worker.model import softmax_cross_entropy


def encode_state(parameters):
    buffer = io.BytesIO()
    np.savez(buffer, **{f'p{i}': p for i, p in enumerate(parameters)})
    return buffer.getvalue()


def decode_state(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return [archive[f'p{i}'].copy() for i in range(len(archive.files))]


def state_digest(parameters):
    digest = hashlib.sha256()
    for p in parameters:
        digest.update(np.ascontiguousarray(p).tobytes())
    return digest.hexdigest()


class TrainingTransactions:
    def get_state(self):
        with self._training_lock:
            params = self.model.parameters()
            return encode_state(params), state_digest(params)

    def configure_training(self, request):
        if request.dataset != self.dataset:
            raise ValueError('all workers must use the coordinator dataset')
        if not request.token or not 0 <= request.rank < len(request.endpoints):
            raise ValueError('invalid attempt or membership')
        with self._training_lock:
            if request.state:
                params = decode_state(request.state)
            else:
                if not request.expected_digest or state_digest(self.model.parameters()) != request.expected_digest:
                    raise ValueError('local checkpoint digest mismatch')
                params = self.model.parameters()
            if len(params) != len(self.model.parameters()) or any(
                p.shape != q.shape or not np.isfinite(p).all()
                for p, q in zip(params, self.model.parameters())
            ):
                raise ValueError('incompatible model checkpoint')
            for target, value in zip(self.model.parameters(), params):
                target[:] = value
            with self._lock:
                self._allreduce_ops.clear()
                self._allreduce_finished.clear()
                self._received_tensors.clear()
                self._allreduce_counter = 0
                self._attempt = request.token
                self._aborted.clear()
            self._pending = None
            self._committed_token = None
            stable_by_endpoint = {endpoint: rank for rank, endpoint in self.heartbeat.endpoints.items()}
            self._managed_peer_ids = {rank: stable_by_endpoint[endpoint]
                                      for rank, endpoint in enumerate(request.endpoints)}
            if request.state:
                self.heartbeat.confirm_membership(self._managed_peer_ids.values())
            self._managed_rank = request.rank
            self._managed_size = len(request.endpoints)
            self.ring = RingAllReduce(request.rank, len(request.endpoints),
                                      dict(enumerate(request.endpoints)), timeout=5, worker=self)

    def abort_training(self, token):
        with self._lock:
            if token == self._attempt:
                self._aborted.set()
                self._tensor_condition.notify_all()
        # Do not wait for TrainStep: its ring polls the cancellation flag.

    def commit_training(self, token):
        with self._training_lock:
            if token != self._attempt or self._aborted.is_set():
                raise RuntimeError('stale or aborted commit')
            if self._committed_token == token:
                return  # Retransmitted commit is idempotent.
            if self._pending is None:
                raise RuntimeError('step has not prepared')
            for target, value in zip(self.model.parameters(), self._pending[0]):
                target[:] = value
            self._committed_token = token

    def train_transaction(self, request):
        with self._training_lock:
            if request.token != self._attempt or self._aborted.is_set():
                raise RuntimeError('stale or aborted training request')
            if request.batch_size < self._managed_size:
                raise ValueError('global batch size must be at least the active worker count')
            if self._pending is not None:
                if self._pending[2] != (request.step, request.batch_size):
                    raise ValueError('attempt reused with different training inputs')
                return self._pending[1]
            start = time.perf_counter()
            if self._training_data is None:
                self._training_data = load_dataset(self.dataset, self.data_dir)
            X, _, y, _ = self._training_data
            global_indices = (np.arange(request.batch_size) + request.step * request.batch_size) % len(X)
            indices = np.array_split(global_indices, self._managed_size)[self._managed_rank]
            batch = X[indices].astype(np.float64)
            if self.dataset == 'cifar10':
                batch = batch / 255.0 - 0.5
            loss, grad = softmax_cross_entropy(self.model.forward(batch), y[indices])
            self.model.backward(grad)
            gradients = self.model.gradients()
            # Weight uneven shards so the result equals the global-batch mean.
            flat = np.concatenate([g.ravel() for g in gradients])
            flat *= len(indices) * self._managed_size / request.batch_size
            compute_time = time.perf_counter() - start
            start = time.perf_counter()
            averaged = self.ring.allreduce(flat)
            if self._aborted.is_set():
                raise RuntimeError('training aborted')
            proposed = []
            offset = 0
            for parameter in self.model.parameters():
                gradient = averaged[offset:offset + parameter.size].reshape(parameter.shape)
                proposed.append(parameter - self.optimizer.learning_rate * gradient)
                offset += parameter.size
            if not all(np.isfinite(p).all() for p in proposed):
                raise RuntimeError('non-finite model update')
            result = dict(loss=loss, compute_time=compute_time,
                          communication_time=time.perf_counter() - start,
                          state=encode_state(proposed), digest=state_digest(proposed), samples=len(indices))
            self._pending = (proposed, result, (request.step, request.batch_size))
            return result
