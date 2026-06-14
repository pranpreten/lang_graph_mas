"""
make_tradeoff.py - 비용 vs 성능 트레이드오프 그래프 생성.
logs/runs.jsonl 읽어서 results/tradeoff.png 저장.
사용: python make_tradeoff.py
비용축 변경: 아래 COST = "tokens"(기본) / "calls" / "time"
"""
import os, json, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COST = "tokens"   # "tokens" | "calls" | "time"

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "logs", "runs.jsonl")
OUT  = os.path.join(HERE, "results", "tradeoff.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def cost_of(r):
    if COST == "tokens":
        return r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
    if COST == "time":
        return r.get("elapsed_sec", 0)
    return r.get("commander_calls", 0)

YLABEL = {"tokens": "Cost: LLM tokens (prompt + completion)",
          "calls":  "Cost: commander calls",
          "time":   "Cost: time (seconds)"}[COST]

rows = []
for line in open(RUNS, encoding="utf-8", errors="ignore"):
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except Exception:
        pass

agg = defaultdict(lambda: {"score": [], "cost": []})
for r in rows:
    if not r.get("completed") or r.get("score") is None:
        continue
    k = (r["task"], r["level"])
    agg[k]["score"].append(r["score"])
    agg[k]["cost"].append(cost_of(r))

def mean(xs):
    return round(st.mean(xs), 4) if xs else None

data = defaultdict(dict)
for (task, lv), d in agg.items():
    data[task][lv] = (mean(d["score"]), mean(d["cost"]))

TASKS  = [("classification", "#1D9E75"), ("regression", "#BA7517"), ("anomaly", "#D85A30")]
LEVELS = ["L1", "L2", "L3", "L4"]

fig, ax = plt.subplots(figsize=(8.2, 5.4))
for task, color in TASKS:
    if task not in data:
        continue
    xs, ys, labs = [], [], []
    for lv in LEVELS:
        if lv in data[task] and data[task][lv][0] is not None:
            xs.append(data[task][lv][0]); ys.append(data[task][lv][1]); labs.append(lv)
    if not xs:
        continue
    ax.plot(xs, ys, "-o", color=color, linewidth=2.2, label=task)
    for x, y, lb in zip(xs, ys, labs):
        ax.annotate(lb, (x, y), textcoords="offset points", xytext=(7, 5),
                    fontsize=9, color=color, fontweight="bold")

ax.set_xlabel("Performance  (F1 / R2 / AUC  -  higher is better)")
ax.set_ylabel(YLABEL + "  -  lower is better")
ax.set_title("Cost vs Performance trade-off by task")
ax.grid(True, alpha=0.15)
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved:", OUT, "| cost =", COST)

print("")
print("task           level  score   " + COST)
for task, _ in TASKS:
    for lv in LEVELS:
        if lv in data.get(task, {}):
            s, c = data[task][lv]
            print(task.ljust(14), lv, "  score=", s, " cost=", c)
