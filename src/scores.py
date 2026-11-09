import json
import re, sys
from pathlib import Path
import javalang
from configs import BEST_CONFIG

SIZE = sys.argv[1]
ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "checkpoints" / f"{SIZE}_cfg10"
TEST = ROOT / "processed" / SIZE / "test"
OUT = ROOT / "results" / SIZE                    
OUT.mkdir(exist_ok=True, parents=True)

BEAMS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


JOIN = [(r"(?<=[\w\]]) class\b", " . class"), (r" < < ", "  <<  "), (r" : : ", " :: "),
        (r" - > ", " -> "), (r" > > > ", " >>> "), (r" > > ", " >> ")]


def is_valid(code):
    # Verify whether the generated Java method snippet has correct syntax using the javalang library

    if not code.strip():
        return False
    for pat, rep in JOIN:
        code = re.sub(pat, rep, code)
    try:
        javalang.parse.parse("public class DummyClass {" + code + "}")
        return True
    except Exception:
        return False


with open(TEST / "fixed.txt", "r", encoding="utf-8") as f:
    fixed = [line.strip() for line in f]

with open(RUN / "times.json", "r") as f:
    times_dict = json.load(f)

results = {}

for k in BEAMS:
    path = RUN / f"predictions_k{k}.txt"
    if not path.exists():
        continue

    with open(path, "r", encoding="utf-8") as f:
        rows = [line.rstrip("\n").split(" <SEP> ") for line in f]


    perfect = sum(any(c.strip() == f for c in cands) for cands, f in zip(rows, fixed))
    total = sum(len(cands) for cands in rows)
    good = sum(is_valid(c) for cands in rows for c in cands)

    total_time = times_dict.get(str(k), 0.0)

    results[k] = dict(hit=perfect, total=len(rows), acc=perfect / len(rows), syntax_hit=good, syntax_total=total, syntax_acc=good / total, avg_time_bug = total_time / len(rows), avg_time_patch = total_time / (len(rows) * k))


name = f"score_{SIZE}"
out_json = OUT / f"{name}.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Saved!")