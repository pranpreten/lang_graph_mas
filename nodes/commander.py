"""commander.py — 커맨더(Claude) 행동. 시스템 프롬프트는 prompts.py에서 가져옴."""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "data")): sys.path.insert(0, p)
import config as c
import llm
import prompts
from _parse import extract_json


def _log(stage, ptok, ctok, note):
    return {"commander_calls": 1, "prompt_tokens": ptok, "completion_tokens": ctok,
            "stage_logs": [{"stage": f"commander:{stage}", "commander_calls": 1,
                            "prompt_tokens": ptok, "completion_tokens": ctok, "note": note[:80]}]}


def route(state):
    summary = state.get("data_summary", {})
    user = f"요청: {state['request']}\n태스크: {state['task']}\n요약: 피처 {summary.get('n_features','?')}개"
    text, ptok, ctok = llm.call_commander(prompts.ROUTE_SYSTEM, user)
    return {"commander_note": text, **_log("route", ptok, ctok, text)}


def guide(state, stage):
    user = f"태스크: {state['task']}\n현재까지 결정: 전처리={state.get('preprocessing','-')}\n이번 단계: {stage}"
    text, ptok, ctok = llm.call_commander(prompts.guide_system(stage), user)
    return {"commander_input": text, **_log(f"guide:{stage}", ptok, ctok, text)}


def _objective_review(state, stage):
    if stage == "preprocessing":
        if not (state.get("preprocessing") or {}).get("feature_cols"):
            return False, "사용할 피처가 없음"
        return True, ""
    if stage == "analysis":
        if state.get("error"):
            return False, f"학습 실패({state['error']})"
        score = state.get("score")
        if score is None or (isinstance(score, float) and math.isnan(score)):
            return False, "점수가 유효하지 않음(None/NaN)"
        model = state.get("decision", {}).get("model")
        if model not in c.MODEL_CANDIDATES[state["task"]]:
            return False, f"후보 밖 모델({model})"
        metric = c.TASK_SPECS[state["task"]]["metric"]
        thr = c.QUALITY_THRESHOLDS.get(metric)
        if thr is not None and score < thr:
            return False, f"{metric}={score:.3f} < 기준 {thr} → 재학습 필요"
        return True, ""
    return True, ""


def review(state, stage):
    ok, reason = _objective_review(state, stage)
    if not ok:
        return {"last_verdict": "재시도", "last_feedback": reason,
                **_log(f"review:{stage}", 0, 0, f"재시도(객관): {reason}")}
    criteria = prompts.REVIEW_CRITERIA.get(stage, "명백히 이상하면 재시도, 아니면 통과.")
    snapshot = {
        "perception": state.get("data_summary", {}).get("task"),
        "preprocessing": state.get("preprocessing"),
        "analysis": {"model": state.get("decision", {}).get("model"), "score": state.get("score")},
        "prescription": state.get("prescription", {}).get("counts"),
    }.get(stage, "-")
    user = f"태스크: {state['task']}\n'{stage}' 결과: {snapshot}"
    text, ptok, ctok = llm.call_commander(prompts.review_system(stage, criteria), user)
    raw = extract_json(text, {"verdict": "통과", "feedback": ""})
    verdict = raw.get("verdict") if raw.get("verdict") in ("통과", "재시도") else "통과"
    return {"last_verdict": verdict, "last_feedback": raw.get("feedback", ""),
            **_log(f"review:{stage}", ptok, ctok, f"{verdict} {raw.get('feedback','')}")}


def control_ml(state):
    task = state["task"]; candidates = c.MODEL_CANDIDATES[task]
    user = f"태스크: {task}\n후보 모델: {candidates}\n피처 {len(state.get('preprocessing',{}).get('feature_cols',[]))}개"
    text, ptok, ctok = llm.call_commander(prompts.CONTROL_ML_SYSTEM, user)
    raw = extract_json(text, {"model": candidates[0], "hyperparams": {}})
    model_name = raw.get("model") if raw.get("model") in candidates else candidates[0]
    hp = raw.get("hyperparams") if isinstance(raw.get("hyperparams"), dict) else {}
    return {"forced_decision": {"model": model_name, "hyperparams": hp, "decided_by": "commander"},
            **_log("control_ml", ptok, ctok, model_name)}
