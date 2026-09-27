import numpy as np
import pytest
from communication import protocol_pb2 as pb
from coordinator.coordinator import Coordinator
from experiments.local_cluster import LocalCluster
from experiments.fault_validation import run
from worker.training import decode_state


def test_four_workers_match_serial_sgd_through_faults(tmp_path):
    report = run(25, output=str(tmp_path / 'faults.json'))
    assert report['recoveries'] >= 5
    assert report['max_absolute_error'] < 1e-10


def test_checkpoint_restart_and_terminal_failure_roll_back(tmp_path):
    path = tmp_path / 'checkpoint.npz'
    with LocalCluster(checkpoint_path=path) as cluster:
        c = cluster.coordinator
        c.start_training(2, 12)
        state = c.checkpoint
        original = cluster.workers[0].commit_training
        def fail(token):
            original(token)  # Uncertain commit: this worker applied it before error.
            raise RuntimeError('permanent commit failure')
        cluster.workers[0].commit_training = fail
        with pytest.raises(RuntimeError, match='recovery exhausted'):
            c.start_training(1, 12)
        assert c.total_steps == 2
        for worker in cluster.workers.values():
            for actual, expected in zip(worker.model.parameters(), decode_state(state)):
                np.testing.assert_array_equal(actual, expected)
        cluster.workers[0].commit_training = original
        restored = Coordinator(4, list(cluster.endpoints.values()), checkpoint_path=path)
        for rank, endpoint in cluster.endpoints.items():
            restored.register_worker(rank, endpoint)
        assert restored.total_steps == 2
        restored.start_training(1, 12)
        assert restored.total_steps == 3


def test_abort_fences_stale_requests_and_duplicate_commit():
    with LocalCluster(1) as cluster:
        c = cluster.coordinator
        c.start_training(1, 8)
        worker = cluster.workers[0]
        token = worker._attempt
        before = worker.get_state()[1]
        worker.commit_training(token)
        assert worker.get_state()[1] == before
        worker.abort_training(token)
        with pytest.raises(RuntimeError, match='aborted'):
            worker.commit_training(token)
        with pytest.raises(RuntimeError, match='stale'):
            worker.train_transaction(pb.TrainStepRequest(token='old', step=0, batch_size=8))


def test_checkpoint_write_failure_restores_workers(tmp_path):
    with LocalCluster(2, checkpoint_path=tmp_path / 'saved.npz') as cluster:
        c = cluster.coordinator
        c.start_training(1, 8)
        checkpoint = c.checkpoint
        def disk_failure(*args):
            raise OSError('injected disk failure')
        c._save = disk_failure
        with pytest.raises(RuntimeError, match='checkpoint write failed'):
            c.start_training(1, 8)
        assert c.total_steps == 1
        for worker in cluster.workers.values():
            for actual, expected in zip(worker.model.parameters(), decode_state(checkpoint)):
                np.testing.assert_array_equal(actual, expected)
        with pytest.raises(ValueError, match='batch size'):
            c.start_training(1, 9)
