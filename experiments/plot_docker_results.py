"""Plot measured Docker scaling; does not use historical simulated results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1] / 'results'
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name, label in [('digits', 'Digits'), ('cifar10', 'CIFAR-10')]:
        report = json.loads((root / f'docker_benchmark_{name}.json').read_text())
        rows = report['rows']
        x = [row['workers'] for row in rows]
        axes[0].plot(x, [r['speedup'] for r in rows], 'o-', label=label)
        axes[1].plot(x, [r['efficiency'] * 100 for r in rows], 'o-', label=label)
    axes[0].axhline(1, color='gray', linestyle='--', linewidth=1)
    axes[0].set_ylabel('Speedup relative to one worker')
    axes[1].set_ylabel('Parallel efficiency (%)')
    for ax in axes:
        ax.set_xlabel('Docker workers')
        ax.set_xticks([1, 2, 4])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Measured strong scaling • fixed global batch • median of 3 runs')
    fig.savefig(root / 'docker_scaling.png', dpi=160)
    plt.close(fig)

if __name__ == '__main__':
    main()
