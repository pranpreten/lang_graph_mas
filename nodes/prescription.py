"""prescription.py — ④ 처방. 임계치 적용(코드) + 근거(SLM). 프롬프트는 prompts.py."""
import os, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "data")): sys.path.insert(0, p)
import config as c
import llm, prompts
from _parse import extract_json

def _action_from_bands(value, bands):
    for low, high, action in bands:
        if low <= value < high:
            return action
    return bands[-1][2]

def _to_policy_values(task, prediction):
    if task == "regression":     return prediction["pred_rul"]
    if task == "classification": return prediction["proba"]
    if task == "anomaly":        return [1 if s > 0 else 0 for s in prediction["anomaly_score"]]
    raise ValueError(task)

def apply_policy(task, prediction):
    values = _to_policy_values(task, prediction)
    bands = c.PRESCRIPTION_POLICY[task]["bands"]
    return [_action_from_bands(v, bands) for v in values]

def prescription_node(state):
    task = state["task"]; prediction = state.get("prediction") or {}
    if not prediction:
        return {"error": "prescription: 예측값 없음(분석 실패 추정)", "current_stage": "prescription"}
    actions = apply_policy(task, prediction)
    counts = dict(Counter(actions))
    user = f"태스크: {task}\n엔진 100대 조치 분포: {counts}"
    text, ptok, ctok = llm.call_slm(prompts.PRESCRIPTION_SYSTEM, user)
    raw = extract_json(text, {"근거": "(자동) 조치 분포 기반 권고", "urgency": "중간"})
    urgency = raw.get("urgency") if raw.get("urgency") in c.URGENCY_LEVELS else "중간"
    return {
        "prescription": {"actions": actions, "counts": counts, "근거": raw.get("근거", ""), "urgency": urgency},
        "completed": True,
        "slm_calls": 1, "prompt_tokens": ptok, "completion_tokens": ctok,
        "stage_logs": [{"stage": "prescription", "slm_calls": 1, "prompt_tokens": ptok,
                        "completion_tokens": ctok, "note": str(counts)}],
        "current_stage": "prescription",
    }
