"""
evaluate.py — 방식 A 결과 평가·시각화 (교수님 피드백 반영: 성능 ↔ 비용 트레이드오프).

analyze.py가 뽑은 '레벨×태스크 집계'를 받아서:
  1) 태스크별 '비용 대비 최적 레벨' 자동 판정 (성능이 포화되는 지점 = 더 써도 안 좋아지는 지점)
  2) 트레이드오프 그래프 PNG 생성 — x축=성능, y축=비용, 태스크별 곡선 + 최적레벨 ◎ 표시
  3) 서술형 분석 리포트 results/평가_분석.md ("분류는 L2로 충분, 이상탐지는 L4 필수" 식)

핵심 아이디어:
  레벨이 오를수록(L1→L4) 커맨더 개입↑ → 비용(호출·토큰·시간)↑.
  그 대가로 성능/안정성이 좋아지길 기대하지만, 어느 지점부터는 비용만 늘고 성능은 제자리.
  그 '포화 직전 레벨'이 그 태스크의 비용 대비 최적 레벨이다. 태스크 복잡도마다 다르게 나온다.

사용:
  python evaluate.py                 # 비용축 = 커맨더 호출수 (기본, 가장 깔끔)
  python evaluate.py --cost tokens   # 비용축 = 토큰
  python evaluate.py --cost time      # 비용축 = 실행시간(초)
  python evaluate.py --eps 0.02       # 성능 '포화' 허용오차 (기본 0.02)
"""
import os, sys, argparse
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config as c
from logger import read_runs
from analyze import summarize

import matplotlib
matplotlib.use("Agg")                 # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt

# 태스크별 주지표 이름 (그래프 x축 라벨용, 영문 — 서버 한글폰트 없어도 안 깨짐)
METRIC = {"classification": "F1", "regression": "R2", "anomaly": "AUC"}
# 비용축 선택지: (집계 컬럼명, 그래프 라벨)
COST_FIELD = {
    "calls":  ("커맨더호출", "Cost (commander calls)"),
    "tokens": ("토큰",       "Cost (tokens)"),
    "time":   ("시간",       "Cost (seconds)"),
}
# 태스크 표시 순서·색 (그래프/리포트 공통)
TASK_ORDER = ["classification", "regression", "anomaly"]
TASK_COLOR = {"classification": "#1D9E75", "regression": "#BA7517", "anomaly": "#D85A30"}
MIN_DONE = 0.5                         # 완주율 이 미만인 (레벨,태스크)는 성능 신뢰 못 함 → 최적후보 제외


def rows_by_task(rows):
    """summarize 결과를 {task: {level: row}} 로 재배열."""
    out = {}
    for r in rows:
        out.setdefault(r["task"], {})[r["level"]] = r
    return out


def pick_optimal(task_rows, eps):
    """
    한 태스크의 레벨별 행을 받아 '비용 대비 최적 레벨' 판정.
    규칙: 완주율 OK + 성능 측정된 레벨들 중 최고성능을 기준으로,
          (최고성능 - eps) 이상을 내면서 '비용이 가장 낮은' 레벨을 고른다.
          = 성능을 거의 다 확보하는 가장 싼 레벨.
    반환: (optimal_level, best_level, info_dict) — 후보 없으면 (None, None, {}).
    """
    cand = {lv: r for lv, r in task_rows.items()
            if r.get("점수평균") is not None and r.get("완주율", 0) >= MIN_DONE}
    if not cand:
        return None, None, {}
    best_perf = max(r["점수평균"] for r in cand.values())
    best_level = max(cand, key=lambda lv: cand[lv]["점수평균"])
    order = {lv: i for i, lv in enumerate(c.LEVELS)}      # L1<L2<L3<L4 = 비용 오름차순
    # 성능 포화선(best-eps) 이상을 내는 레벨 중 가장 낮은(싼) 레벨
    reach = [lv for lv in cand if cand[lv]["점수평균"] >= best_perf - eps]
    optimal = min(reach, key=lambda lv: order[lv])
    return optimal, best_level, {"best_perf": best_perf, "n_levels": len(cand)}


def make_plot(by_task, cost_key, path):
    """트레이드오프 산점도: x=성능, y=비용, 태스크별 곡선 + 최적레벨 ◎."""
    cost_col, cost_label = COST_FIELD[cost_key]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for task in TASK_ORDER:
        if task not in by_task:
            continue
        trows = by_task[task]
        opt, _, _ = pick_optimal(trows, EPS)
        xs, ys, labels, opts = [], [], [], []
        for lv in c.LEVELS:                       # L1→L4 순서로 연결
            r = trows.get(lv)
            if not r or r.get("점수평균") is None:
                continue
            xs.append(r["점수평균"]); ys.append(r[cost_col]); labels.append(lv)
            opts.append(lv == opt)
        if not xs:
            continue
        color = TASK_COLOR.get(task, "#555")
        ax.plot(xs, ys, "-", color=color, linewidth=2, zorder=1,
                label=f"{task}  (optimal {opt})" if opt else task)
        for x, y, lb, is_opt in zip(xs, ys, labels, opts):
            if is_opt:                            # 최적레벨 = 큰 빈 원(◎)
                ax.scatter([x], [y], s=200, facecolors="none", edgecolors=color,
                           linewidths=2.5, zorder=3)
            ax.scatter([x], [y], s=42, color=color, zorder=2)
            ax.annotate(lb, (x, y), textcoords="offset points", xytext=(7, 6),
                        fontsize=9, color=color)
    ax.set_xlabel("Performance  (F1 / R2 / AUC  -  higher is better)")
    ax.set_ylabel(f"{cost_label}  -  lower is better")
    ax.set_title("Cost vs Performance trade-off by task  (O = cost-optimal level)")
    ax.grid(True, alpha=0.15)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(by_task, cost_key, plot_name, md_path, n_runs):
    cost_col, cost_label = COST_FIELD[cost_key]
    L = []
    L.append("# 방식 A 평가 — 성능 ↔ 비용 트레이드오프\n")
    L.append(f"_총 {n_runs}런 기준. 비용축 = {cost_col}. 성능 포화 허용오차 eps = {EPS}._\n")
    L.append(f"![트레이드오프]({plot_name})\n")
    L.append("## 태스크별 최적 조율 수준\n")
    for task in TASK_ORDER:
        if task not in by_task:
            continue
        trows = by_task[task]
        opt, best, info = pick_optimal(trows, EPS)
        metric = METRIC.get(task, "score")
        L.append(f"### {task}  (지표: {metric})\n")
        if opt is None:
            L.append("- 완주한 런이 부족해 판정 불가 (분석 실패 다수).\n")
            continue
        ro = trows[opt]
        line = (f"- **최적 레벨: {opt}** — 성능 {ro['점수평균']} 를 {cost_col} {ro[cost_col]}(으)로 달성. "
                f"(완주율 {ro['완주율']}, 점수편차 {ro['점수편차']})")
        L.append(line)
        # 더 올렸을 때 이득이 있는지(=포화 확인) 서술
        order = {lv: i for i, lv in enumerate(c.LEVELS)}
        higher = [lv for lv in trows if order.get(lv, -1) > order.get(opt, -1)
                  and trows[lv].get("점수평균") is not None]
        if higher:
            nxt = min(higher, key=lambda lv: order[lv])
            rn = trows[nxt]
            dperf = round(rn["점수평균"] - ro["점수평균"], 3)
            dcost = round(rn[cost_col] - ro[cost_col], 1)
            L.append(f"- {opt}→{nxt}: 성능 {dperf:+} 변화에 {cost_col} {dcost:+} 추가 "
                     f"→ {'추가 비용만큼의 성능 이득 미미 (포화)' if dperf <= EPS else '아직 성능 이득 존재'}.")
        else:
            L.append(f"- {opt}이 최상위 레벨까지 중 최적 (더 낮출 여지 없음).")
        L.append("")  # 빈 줄
    # 종합 한 줄
    picks = []
    for task in TASK_ORDER:
        if task in by_task:
            opt, _, _ = pick_optimal(by_task[task], EPS)
            if opt:
                picks.append(f"{task}={opt}")
    if picks:
        L.append("## 종합\n")
        L.append("태스크 복잡도에 따라 요구되는 조율 수준이 다름: " + ", ".join(picks) + ".\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", choices=list(COST_FIELD), default="calls")
    ap.add_argument("--eps", type=float, default=0.02)
    a = ap.parse_args()
    EPS = a.eps

    runs = read_runs()
    if not runs:
        print("로그가 비었습니다. 먼저 `python run_experiment.py` 로 실험을 돌리세요.")
        sys.exit(0)

    rows = summarize(runs)
    by_task = rows_by_task(rows)

    os.makedirs(c.RESULT_DIR, exist_ok=True)
    plot_name = f"tradeoff_{a.cost}.png"
    plot_path = os.path.join(c.RESULT_DIR, plot_name)
    md_path = os.path.join(c.RESULT_DIR, "평가_분석.md")
    make_plot(by_task, a.cost, plot_path)
    write_report(by_task, a.cost, plot_name, md_path, len(runs))

    # 콘솔 요약
    print(f"총 {len(runs)}런 → 비용축='{a.cost}', eps={EPS}\n")
    print("태스크별 비용 대비 최적 레벨:")
    for task in TASK_ORDER:
        if task in by_task:
            opt, best, _ = pick_optimal(by_task[task], EPS)
            print(f"  {task:14} 최적={opt}  (최고성능레벨={best})")
    print(f"\n→ 그래프: {plot_path}")
    print(f"→ 분석:  {md_path}")
