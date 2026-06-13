"""
runner.py — 배치 실행기 (방식 A: 전수비교).

레벨 × 태스크 × 반복 을 돌면서 각 런을 타이머로 감싸 logger에 기록.

사용:
  python runner.py                  # 전체 방식 A (config.N_REPEATS=10 → 120런)
  python runner.py --pilot          # 빠른 점검: 각 조건 1회씩 (= 12런)
  python runner.py --repeats 5      # 반복 횟수 지정
  python runner.py --levels L1 L4   # 일부 레벨만
  python runner.py --tasks regression
  python runner.py --dry            # 실행 안 하고 '돌릴 계획'만 출력
"""
import os, sys, time, argparse
ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, os.path.join(ROOT, "nodes"), os.path.join(ROOT, "data")):
    sys.path.insert(0, p)
import config as c
import llm
from graph import build_graph
from state import new_run_state
from logger import log_run, LOG_PATH


def run_batch(levels, tasks, repeats, dry=False, fresh=False):
    os.makedirs(c.LOG_DIR, exist_ok=True)
    if fresh and not dry:
        for pth in (LOG_PATH, llm.TRACE_PATH):   # 요약 로그 + 대화 trace 둘 다 비움
            if os.path.exists(pth):
                open(pth, "w").close()
        print("(--fresh: 기존 로그·trace 비움)")
    total = len(levels) * len(tasks) * repeats
    print(f"방식 A: {levels} × {tasks} × {repeats}회 = 총 {total}런")
    print(f"기록 위치: {LOG_PATH}\n")
    if dry:
        n = 0
        for lv in levels:
            for tk in tasks:
                for r in range(repeats):
                    n += 1
                    print(f"  [{n}/{total}] {lv} {tk} r{r}")
        print("\n(--dry: 실제 실행 안 함)")
        return

    done = 0
    t_start = time.time()
    for level in levels:
        graph = build_graph(level)                 # 레벨당 한 번만 컴파일 (재사용)
        for task in tasks:
            for r in range(repeats):
                done += 1
                run_id = f"{level}-{task}-r{r}"
                init = new_run_state(run_id=run_id, level=level, task=task,
                                     repeat=r, request=c.TASK_SPECS[task]["request"], seed=r)
                llm.set_run_id(run_id)               # trace에 이 런 식별자 기록
                t0 = time.time()
                try:
                    final = graph.invoke(init)
                except Exception as e:             # 런이 통째로 터져도 기록하고 계속
                    final = {**init, "completed": False, "error": f"invoke 실패: {e}"}
                elapsed = time.time() - t0
                log_run(final, elapsed)
                flag = "" if final.get("error") is None else f" ⚠{final.get('error')[:40]}"
                print(f"[{done}/{total}] {level} {task} r{r} | "
                      f"완주={final.get('completed')} 점수={final.get('score')} "
                      f"커맨더={final.get('commander_calls')} {elapsed:.1f}s{flag}")

    print(f"\n완료: {done}런, 총 {time.time()-t_start:.0f}초. → {LOG_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="각 조건 1회씩만 (빠른 점검)")
    ap.add_argument("--repeats", type=int, default=None)
    