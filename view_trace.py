"""
view_trace.py — 한 런의 에이전트 대화를 '순서대로' 보여준다 (흐름 재구성용).
사용:
  python view_trace.py                  # 첫 번째 런
  python view_trace.py L4-anomaly-r0    # 특정 런
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as c

path = os.path.join(c.LOG_DIR, "trace.jsonl")
if not os.path.exists(path):
    print("trace 없음. 먼저 runner를 돌리세요."); sys.exit(0)

rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
run_ids = list(dict.fromkeys(r["run_id"] for r in rows))
target = sys.argv[1] if len(sys.argv) > 1 else (run_ids[0] if run_ids else None)
steps = [r for r in rows if r["run_id"] == target]

print(f"=== 런: {target} ===  (총 {len(steps)}개 호출)\n")
for i, s in enumerate(steps, 1):
    who = "🟠커맨더" if s["role"] == "commander" else "🟢SLM"
    sys1 = s["system"].splitlines()[0][:50]
    print(f"[{i}] {who}  ({sys1}…)")
    print(f"    입력: {s['input'][:120].replace(chr(10),' ')}")
    print(f"    응답: {s['response'][:200].replace(chr(10),' ')}")
    print(f"    토큰: in {s['prompt_tokens']} / out {s['completion_tokens']}\n")

print(f"가능한 run_id 예시: {run_ids[:5]}")
