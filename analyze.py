"""
analyze.py — logs/runs.jsonl 집계 → 안정성·비용 요약 + 레벨별 트레이드오프.

사용: python analyze.py
출력: (1) 레벨×태스크 상세표  (2) 레벨별 요약(트레이드오프)  → results/ 에 CSV 저장
"""
import os, sys, csv, statistics
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config as c
from logger import read_runs


def _mode_freq(values):
    """가장 흔한 값의 비율 (1.0 = 전부 동일 = 완전 일관)."""
    if not values:
        return None
    return Counter(values).most_common(1)[0][1] / len(values)


def _prescription_consistency(action_lists):
    """반복 간 엔진별 조치 일치율 (1.0 = 매 반복 같은 조치 = 안정)."""
    lists = [a for a in action_lists if a]
    if len(lists) < 2:
        return None
    n = min(len(a) for a in lists)
    per_engine = [_mode_freq([a[i] for a in lists]) for i in range(n)]
    return sum(per_engine) / len(per_engine)


def summarize(runs):
    """(레벨, 태스크)별 안정성·비용 집계."""
    groups = defaultdict(list)
    for r in runs:
        groups[(r["level"], r["task"])].append(r)

    rows = []
    for (level, task), g in sorted(groups.items()):
        done = [r for r in g if r.get("completed")]
        scores = [r["score"] for r in done if r.get("score") is not None]
        models = [r["model"] for r in done if r.get("model")]
        actions = [r.get("actions") for r in done]
        preps = [(r.get("preprocessing_mode"), r.get("scaler")) for r in done if r.get("preprocessing_mode")]
        rows.append({
            "level": level, "task": task, "n": len(g),
            "완주율": round(sum(1 for r in g if r.get("completed")) / len(g), 2),
            "점수평균": round(statistics.mean(scores), 3) if scores else None,
            "점수편차": round(statistics.pstdev(scores), 3) if len(scores) > 1 else 0.0,
            "모델일관성": round(_mode_freq(models), 2) if models else None,
            "전처리일관성": round(_mode_freq(preps), 2) if preps else None,
            "처방일관성": (round(_prescription_consistency(actions), 2)
                          if _prescription_consistency(actions) is not None else None),
            "평균재시도": round(statistics.mean([sum((r.get("retry_counts") or {}).values()) for r in g]), 2),
            "커맨더호출": round(statistics.mean([r.get("commander_calls", 0) for r in g]), 1),
            "SLM호출": round(statistics.mean([r.get("slm_calls", 0) for r in g]), 1),
            "토큰": round(statistics.mean([r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in g])),
            "커맨더초": round(statistics.mean([r.get("commander_sec", 0) or 0 for r in g]), 1),
            "SLM초": round(statistics.mean([r.get("slm_sec", 0) or 0 for r in g]), 1),
            "시간": round(statistics.mean([r.get("elapsed_sec", 0) for r in g]), 1),
        })
    return rows


def by_level(rows):
    """레벨별로 태스크 평균 → 트레이드오프(안정성↔비용)."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["level"]].append(r)
    out = []
    for level in c.LEVELS:
        g = groups.get(level, [])
        if not g:
            continue
        def avg(k):
            vals = [r[k] for r in g if r[k] is not None]
            return round(sum(vals) / len(vals), 3) if vals else None
        out.append({
            "level": level,
            "완주율": avg("완주율"), "점수편차": avg("점수편차"),
            "모델일관성": avg("모델일관성"), "전처리일관성": avg("전처리일관성"),
            "처방일관성": avg("처방일관성"), "평균재시도": avg("평균재시도"),
            "커맨더호출": avg("커맨더호출"), "SLM호출": avg("SLM호출"),
            "토큰": avg("토큰"), "커맨더초": avg("커맨더초"), "SLM초": avg("SLM초"),
            "시간": avg("시간"),
        })
    return out


def _print_table(rows, cols):
    print(" | ".join(f"{c_:>8}" for c_ in cols))
    print("-" * (len(cols) * 11))
    for r in rows:
        print(" | ".join(f"{str(r.get(c_,'')):>8}" for c_ in cols))


if __name__ == "__main__":
    runs = read_runs()
    if not runs:
        print("로그가 비었습니다. 먼저 `python runner.py` 로 실험을 돌리세요.")
        sys.exit(0)

    print(f"총 {len(runs)}런 분석\n")
    rows = summarize(runs)
    print("=== 레벨×태스크 상세 ===")
    _print_table(rows, ["level", "task", "n", "완주율", "점수평균", "점수편차", "모델일관성",
                        "전처리일관성", "처방일관성", "평균재시도", "커맨더호출", "SLM호출",
                        "토큰", "커맨더초", "SLM초", "시간"])
    print("\n=== 레벨별 요약 (안정성 ↔ 비용 트레이드오프) ===")
    lv = by_level(rows)
    _print_table(lv, ["level", "완주율", "점수편차", "모델일관성", "전처리일관성", "처방일관성",
                      "평균재시도", "커맨더호출", "SLM호출", "토큰", "커맨더초", "SLM초", "시간"])

    # CSV 저장
    os.makedirs(c.RESULT_DIR, exist_ok=True)
    with open(os.path.join(c.RESULT_DIR, "summary_by_level_task.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(c.RESULT_DIR, "summary_by_level.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lv[0].keys())); w.writeheader(); w.writerows(lv)
    print(f"\n→ CSV 저장: {c.RESULT_DIR}")
