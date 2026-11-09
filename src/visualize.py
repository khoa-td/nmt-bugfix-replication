import sys
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

SIZE = sys.argv[1] if len(sys.argv) > 1 else "small"
OUT = ROOT / "results" / SIZE
OUT.mkdir(exist_ok=True, parents=True)

json_path = OUT / f"score_{SIZE}.json"

with open(json_path, "r", encoding="utf-8") as f:
    results = json.load(f)

beams = []
accuracies = []
syntax_acc = []
avg_time_bug = []
avg_time_patch = []

for k, metric in results.items():
    beams.append(int(k))
    accuracies.append(metric["acc"] * 100)
    syntax_acc.append(metric["syntax_acc"] * 100)
    avg_time_bug.append(metric.get("avg_time_bug", 0))
    avg_time_patch.append(metric.get("avg_time_patch", 0))

AUTHOR_DATA = {
    "small": {
        "beams": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "acc": [9.22, 27.33, 36.31, 40.37, 43.49, 45.14, 46.46, 47.40, 48.56, 49.68, 50.16],
        "syntax": [99.0, 87.0, 86.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 84.0, 82.0]
    },
    "medium": {
        "beams": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "acc": [3.22, 13.12, 17.82, 20.25, 22.16, 23.80, 25.36, 26.27, 27.15, 27.96, 28.55],
        "syntax": [98.0, 89.0, 87.0, 86.0, 86.0, 85.0, 85.0, 85.0, 85.0, 84.0, 85.0]
    }
}

size_key = SIZE.lower()
author_beams = AUTHOR_DATA[size_key]["beams"]
author_acc = AUTHOR_DATA[size_key]["acc"]
author_syn = AUTHOR_DATA[size_key]["syntax"]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(beams, accuracies, marker="o", color="orange", linewidth=2, label="My Evaluation")
for b, acc in zip(beams, accuracies):
    ax1.text(b, acc + 1.5, f"{acc:.1f}%", ha="center", va="bottom", fontsize=9, color="black")

if size_key in AUTHOR_DATA:
    ax1.plot(author_beams, author_acc, marker="s", color="green", linewidth=2, linestyle="--", label="Author's Paper")
    for b, acc in zip(author_beams, author_acc):
        ax1.text(b, acc - 2.5, f"{acc:.1f}%", ha="center", va="top", fontsize=9, color="green")

ax1.set_title(f"Accuracy Benchmarking ({SIZE.upper()})", fontsize=13, fontweight="bold", pad=10)
ax1.set_xlabel("Beam width", fontsize=11)
ax1.set_ylabel("Accuracy (%)", fontsize=11)
all_accs = accuracies + (author_acc if size_key in AUTHOR_DATA else [])
ax1.set_ylim(0, max(all_accs) + 10 if all_accs else 100)
ax1.set_xticks(beams)
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.legend(loc="lower right")

ax2.plot(beams, syntax_acc, marker="o", color="blue", linewidth=2, label="My Evaluation")
for b, syn in zip(beams, syntax_acc):
    ax2.text(b, syn + 1, f"{syn:.1f}%", ha="center", va="bottom", fontsize=9, color="black")

if size_key in AUTHOR_DATA:
    ax2.plot(author_beams, author_syn, marker="s", color="red", linewidth=2, linestyle="--", label="Author's Paper")
    for b, syn in zip(author_beams, author_syn):
        ax2.text(b, syn - 1.5, f"{syn:.1f}%", ha="center", va="top", fontsize=9, color="red")

ax2.set_title(f"Syntactic correctness Benchmarking ({SIZE.upper()})", fontsize=13, fontweight="bold", pad=10)
ax2.set_xlabel("Beam width", fontsize=11)
ax2.set_ylabel("Syntax Valid (%)", fontsize=11)
ax2.set_ylim(0, 105)
ax2.set_xticks(beams)
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.legend(loc="lower right")

plt.tight_layout()
plt.savefig(OUT / f"charts_{SIZE}.png", dpi=300)
plt.close()



if any(t > 0 for t in avg_time_bug):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(beams, avg_time_bug, marker="o", color="lightgreen", linewidth=2, label="My Eval - avg. per bug")
    ax.plot(beams, avg_time_patch, marker="o", color="darkorange", linewidth=2, label="My Eval - avg. per patch")
    

    for b, t in zip(beams, avg_time_bug):
        ax.text(b, t + (max(avg_time_bug)*0.03), f"{t:.3f}s", ha="center", va="bottom", fontsize=8, color="darkgreen")

    ax.set_title(f"Inference Time ({SIZE.upper()})", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Beam width", fontsize=11)
    ax.set_ylabel("Seconds", fontsize=11)
    
    max_time = max(max(avg_time_bug), 0.25)
    ax.set_ylim(0, max_time * 1.15)
    ax.set_xticks(beams)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(OUT / f"time_chart_{SIZE}.png", dpi=300)
    plt.close()