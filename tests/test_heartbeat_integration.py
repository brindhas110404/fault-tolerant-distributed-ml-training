import threading
import time
import pytest
from communication import protocol_pb2 as pb
from experiments.heartbeat_validation import trial
from experiments.local_cluster import LocalCluster


def test_concurrent_heartbeat_detection_with_real_timeouts():
    result = trial('simultaneous_timeouts')
    assert max(result['detection_seconds'].values()) < 8


def test_heartbeat_aborts_active_ring_wait_and_ignores_removed_peer():
    with LocalCluster(3) as cluster:
        c = cluster.coordinator
        c.start_training(1, 12)
        worker = cluster.workers[0]
        errors = []
        def wait_for_absent_tensor():
            try:
                worker.ring._request_chunk(1, 0, 0, 0, worker._attempt + '/absent')
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=wait_for_absent_tensor)
        thread.start()
        detected_start = time.monotonic()
        cluster.servers[1].stop(0).wait()
        thread.join(timeout=8)
        assert time.monotonic() - detected_start < 8
        assert 1 in worker.heartbeat.get_dead_workers()
        assert not thread.is_alive()
        assert errors and 'aborted' in str(errors[0]).lower()
        with pytest.raises(RuntimeError, match='aborted'):
            worker.commit_training(worker._attempt)
        state, _ = worker.get_state()
        worker.configure_training(pb.ConfigureRequest(token='new-membership', rank=0,
            endpoints=[cluster.endpoints[0], cluster.endpoints[2]], state=state, dataset='digits'))
        worker._heartbeat_failure(1)
        assert not worker._aborted.is_set()


def test_fast_checkpoint_path_rejects_corruption():
    with LocalCluster(1) as cluster:
        cluster.coordinator.start_training(1, 8)
        worker = cluster.workers[0]
        _, digest = worker.get_state()
        worker.model.parameters()[0][0, 0] += 1
        with pytest.raises(ValueError, match='digest mismatch'):
            worker.configure_training(pb.ConfigureRequest(token='bad-state', rank=0,
                endpoints=[cluster.endpoints[0]], dataset='digits', expected_digest=digest))


def test_fast_path_multi_step_partial_commit_matches_serial():
    import numpy as np
    from worker.dataset import load_data
    from worker.model import MLP, softmax_cross_entropy
    from worker.optimizer import SGD
    with LocalCluster(3) as cluster:
        worker = cluster.workers[1]
        commit = worker.commit_training
        calls = [0]
        def fail_once_after_commit(token):
            calls[0] += 1
            commit(token)
            if calls[0] == 3:
                raise RuntimeError('lost third-step commit acknowledgement')
        worker.commit_training = fail_once_after_commit
        cluster.coordinator.start_training(8, 13)
        assert cluster.coordinator.last_run['recoveries'] == 1
        X, _, y, _ = load_data()
        reference = MLP(64, 128, 10, seed=7)
        optimizer = SGD(reference.parameters(), learning_rate=.05)
        for step in range(8):
            indices = (np.arange(13) + step * 13) % len(X)
            _, grad = softmax_cross_entropy(reference.forward(X[indices]), y[indices])
            reference.backward(grad)
            optimizer.step(reference.gradients())
        for member in cluster.workers.values():
            for actual, expected in zip(member.model.parameters(), reference.parameters()):
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_rejoin_resets_cached_connections_and_stale_health():
    with LocalCluster(3) as cluster:
        c = cluster.coordinator
        c.start_training(1, 12)
        cluster.servers[2].stop(0).wait()
        c.start_training(1, 12)
        assert c.last_run['active_workers'] == [0, 1]
        # Model a heartbeat verdict retained from the outage, without a sleep.
        for rank in (0, 1):
            heartbeat = cluster.workers[rank].heartbeat
            with heartbeat._lock:
                heartbeat._dead_workers.add(2)
                heartbeat._failure_counts[2] = heartbeat.max_failures
        cluster.restart(2)
        c.start_training(1, 12)
        assert c.last_run['active_workers'] == [0, 1, 2]
        assert all(cluster.workers[r].heartbeat.is_alive(2) for r in (0, 1))
