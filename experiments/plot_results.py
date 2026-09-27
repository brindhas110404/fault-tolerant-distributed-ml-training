import json

import matplotlib.pyplot as plt


with open(
    "results/scaling.json"
) as file:

    results = json.load(
        file
    )


workers = [
    result["workers"]
    for result in results
]

speedups = [
    result["speedup"]
    for result in results
]

efficiencies = [
    result["efficiency"]
    for result in results
]


# -----------------------------
# Speedup
# -----------------------------

plt.figure()

plt.plot(
    workers,
    speedups,
    marker="o"
)

plt.xlabel(
    "Number of Workers"
)

plt.ylabel(
    "Speedup"
)

plt.title(
    "Distributed Training Speedup"
)

plt.grid()

plt.savefig(
    "results/speedup.png",
    dpi=200,
    bbox_inches="tight"
)


# -----------------------------
# Efficiency
# -----------------------------

plt.figure()

plt.plot(
    workers,
    efficiencies,
    marker="o"
)

plt.xlabel(
    "Number of Workers"
)

plt.ylabel(
    "Parallel Efficiency"
)

plt.title(
    "Distributed Training Efficiency"
)

plt.grid()

plt.savefig(
    "results/efficiency.png",
    dpi=200,
    bbox_inches="tight"
)

print(
    "Plots saved to results/"
)