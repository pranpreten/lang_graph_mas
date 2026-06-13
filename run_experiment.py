"""
run_experiment.py — 방식 A 실험을 한 번에 실행 (runner → analyze).

  python run_experiment.py          # 전체 120런 + 집계
  python run_experiment.py pilot    # 빠른 점검 12런 + 집계

(--fresh 로 로그·trace 비우고 시작 → 깔끔하게)
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable                              # 현재 파이썬 환경 그대로 사용
pilot = len(sys.argv) > 1 and sys.argv[1] == "pilot"

# 1) 실험 실행
run_cmd = [PY, "runner.py", "--fresh"] + (["--pilot"] if pilot else [])
print(">>", " ".join(run_cmd))
subprocess.run(run_cmd, cwd=HERE, check=True)

# 2) 결과 집계
print("\n>> python analyze.py")
subprocess.run([PY, "analyze.py"], cwd=HERE, check=True)

print("\n끝 → 요약: results/  |  단계별 대화: logs/trace.jsonl")
