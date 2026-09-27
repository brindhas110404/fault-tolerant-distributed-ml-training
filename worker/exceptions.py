class WorkerFailedError(Exception):

    def __init__(self, worker_id, reason=""):

        self.worker_id = worker_id
        self.reason = reason

        super().__init__(
            f"Worker {worker_id} failed: {reason}"
        )