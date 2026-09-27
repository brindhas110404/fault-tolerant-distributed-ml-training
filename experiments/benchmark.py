import argparse
import json
import os
import subprocess
import sys


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=[1, 2, 4]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    args = parser.parse_args()

    os.makedirs(
        "results",
        exist_ok=True
    )

    results = []

    for workers in args.workers:

        output_file = (
            f"results/run_{workers}.json"
        )

        subprocess.run(
            [
                sys.executable,
                "main.py",
                "--workers",
                str(workers),
                "--epochs",
                str(args.epochs),
                "--out",
                output_file
            ],
            check=True
        )

        with open(
            output_file
        ) as file:

            result = json.load(
                file
            )

        total_time = (
            result["history"][-1]["time"]
        )

        results.append({
            "workers": workers,
            "time": total_time
        })

    baseline = results[0]["time"]

    for result in results:

        result["speedup"] = (
            baseline
            / result["time"]
        )

        result["efficiency"] = (
            result["speedup"]
            / result["workers"]
        )

    with open(
        "results/scaling.json",
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )

    print()

    for result in results:

        print(
            f"Workers: "
            f"{result['workers']} | "
            f"Time: "
            f"{result['time']:.2f}s | "
            f"Speedup: "
            f"{result['speedup']:.2f}x | "
            f"Efficiency: "
            f"{result['efficiency']:.2%}"
        )


if __name__ == "__main__":

    main()