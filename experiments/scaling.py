import time
import json


def benchmark(trainer_factory, worker_counts):

    results = []

    for workers in worker_counts:

        print(
            f"\nRunning with {workers} workers"
        )

        trainer = trainer_factory(
            workers
        )

        start = time.perf_counter()

        trainer.train(
            epochs=3
        )

        elapsed = (
            time.perf_counter() - start
        )

        results.append({
            "workers": workers,
            "time": elapsed
        })

        print(
            f"Workers={workers} "
            f"Time={elapsed:.2f}s"
        )

    baseline = results[0]["time"]

    for result in results:

        result["speedup"] = (
            baseline / result["time"]
        )

        result["efficiency"] = (
            result["speedup"]
            / result["workers"]
        )

    with open(
        "results/scaling.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )


if __name__ == "__main__":

    benchmark(
        trainer_factory=lambda n: ...,
        worker_counts=[1, 2, 4]
    )   