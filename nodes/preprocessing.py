"""preprocessing.py — ② 전처리 노드 (SLM). 프롬프트는 prompts.py."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "data")): sys.path.insert(0, p)
import config as c
import llm, prompts
from _parse import extract_json
from _ctx import commander_prefix

_FEATURE_MODES = {"all", "drop_low_variance"}
_SCALERS = {"standard", "minmax", "none"}
_DEFAULT = {"features": "drop_low_variance", "scaler": "standard"}

def preprocessing_node(state):
    summary = state["data_summary"]
    user = (commander_prefix(state) +
            f"피처 목록: {summary['features']}\n"
            f"거의 상수(저분산)인 센서: {summary['low_variance_features']}\n"
            f"태스크: {summary['task']} ({summary['kind']})")
    text, ptok, ctok = llm.call_slm(prompts.PREPROCESSING_SYSTEM, user)
    raw = extract_json(text, _DEFAULT)
    fmode = raw.get("features", _DEFAULT["features"])
    scaler = raw.get("scaler", _DEFAULT["scaler"])
    if fmode not in _FEATURE_MODES: fmode = _DEFAULT["features"]
    if scaler not in _SCALERS: scaler = _DEFAULT["scaler"]
    if fmode == "all":
        feature_cols = list(summary["features"])
    else:
        low = set(summary["low_variance_features"])
        feature_cols = [f for f in summary["features"] if f not in low]
    return {
        "preprocessing": {"features_mode": fmode, "scaler": scaler, "feature_cols": feature_cols},
        "commander_input": "",
        "slm_calls": 1, "prompt_tokens": ptok, "completion_tokens": ctok,
        "stage_logs": [{"stage": "preprocessing", "slm_calls": 1, "prompt_tokens": ptok,
                        "completion_tokens": ctok, "note": f"{fmode}/{scaler}, 피처 {len(feature_cols)}개"}],
        "current_stage": "preprocessing",
    }
