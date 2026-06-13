"""
view_trace.py — 한 런의 에이전트 대화를 순서대로 보여준다.
  python view_trace.py            # 어떤 런들이 있는지 목록만 출력
  python view_trace.py <run_id>   # 그 런의 대화 흐름 출력
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as c

path = os.path.join(c.LOG_DIR, "trace.jsonl")
if not os.path.exists(path):
    print("trace 없음. 먼저 실험을 돌리세요 (python run_experiment.py pilot).")
    sys.exit(0)

rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
run_ids = list(dict.fromkeys(r["run_id"] for r in rows))

# 인자 없으면 → 목록만 보여주고 끝
if len(sys.argv) < 2:
    print(f"기록된 런 {len(run_ids)}개:\n")
    for rid in run_ids:
        n = sum(1 for r in rows if r["run_id"] == rid)
        print(f"  {rid}   ({n}개 호출)")
    print("\n자세히 보려면:  python view_trace.py <run_id>")
    print("예:  python view_trace.py", run_ids[0] if run_ids else "L1-regression-r0")
    sys.exit(0)

# 인자 있으면 → 그 런 대화 출력
target = sys.argv[1]
steps = [r for r in rows if r["run_id"] == target]
if not steps:
    print(f"'{target}' 런이 없어요. 인자 없이 치면 목록 나와요: python view_trace.py")
    sys.exit(0)

print(f"=== 런: {target} ===  (총 {len(steps)}개 호출)\n")
for i, s in enumerate(steps, 1):
    who = "🟠커맨더" if s["role"] == "commander" else "🟢SLM"
    sys1 = s["system"].splitlines()[0][:50]
    print(f"[{i}] {who}  ({sys1}…)")
    print(f"    입력: {s['input'][:120].replace(chr(10),' ')}")
    print(f"    응답: {s['response'][:200].replace(chr(10),' ')}")
    print(f"    토큰: in {s['prompt_tokens']} / out {s['completion_tokens']}\n")
