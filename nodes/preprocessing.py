"""
preprocessing.py — ② 전처리(Preprocessing) 노드 (SLM).

하는 일:
  1) 인지가 만든 데이터 요약을 본다.
  2) SLM이 '어떤 피처를 쓸지(상수 센서 제외 여부) + 정규화 방법'을 JSON으로 결정.
  3) 그 결정을 파싱·검증해 실제 사용할 피처 목록까지 확정 → state에 저장.

※ 실제 변환(스케일링·피처 선택 적용)은 분석 노드에서 학습 직전에 한다.
   여기서는 'SLM의 결정'만 확정한다.
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))      # .../nodes
ROOT = os.path.dirname(HERE)                            # .../rxm_experiment
for p in (HERE, ROOT, os.path.join(ROOT, "data")):
    sys.path.insert(0, p)
import config as c
import llm
from _parse import extract_json

# SLM이 고를 수 있는 선택지 (검증용)
_FEATURE_MODES = {"all", "drop_low_variance"}
_SCALERS = {"standard", "minmax", "none"}
_DEFAULT = {"features": "drop_low_variance", "scaler": "standard"}


def preprocessing_node(state):
    summary = state["data_summary"]

    system = (
        "너는 RxM 파이프라인의 '전처리' 에이전트다. 데이터 요약을 보고 전처리 방법을 정하라.\n"
        "반드시 아래 형식의 JSON '하나만' 출력하라(설명 금지):\n"
        '{"features": "drop_low_variance" 또는 "all", "scaler": "standard" 또는 "minmax" 또는 "none"}'
    )
    user = (
        f"피처 목록: {summary['features']}\n"
        f"거의 상수(저분산)인 센서: {summary['low_variance_features']}\n"
        f"태스크: {summary['task']} ({summary['kind']})"
    )
    text, ptok, ctok = llm.call_slm(system, user)

    # SLM 응답 → JSON 결정 (실패 시 기본값)
    raw = extract_json(text, _DEFAULT)
    fmode  = raw.get("features", _DEFAULT["features"])
    scaler = raw.get("scaler",   _DEFAULT["scaler"])
    if fmode not in _FEATURE_MODES:  fmode = _DEFAULT["features"]
    if scaler not in _SCALERS:       scaler = _DEFAULT["scaler"]

    # 실제 사용할 피처 목록 확정
    if fmode == "all":
        feature_cols = list(summary["features"])
    else:  # drop_low_variance
        low = set(summary["low_variance_features"])
        feature_cols = [f for f in summary["features"] if f not in low]

    decision = {"features_mode": fmode, "scaler": scaler, "feature_cols": feature_cols}

    return {
        "preprocessing": decision,
        "slm_calls": 1,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "stage_logs": [{"stage": "preprocessing", "slm_calls": 1,
                        "prompt_tokens": ptok, "completion_tokens": ctok,
                        "note": f"{fmode}/{scaler}, 피처 {len(feature_cols)}개"}],
        "current_stage": "preprocessing",
    }


if __name__ == "__main__":
    # ★ 내 PC/서버에서 실행 ★ (실제 SLM 호출)
    import data_prep as dp
    Xtr, *_ = dp.make_task("regression")
    summary = dp.data_summary(Xtr, "regression")
    state = {"data_summary": summary, "task": "regression"}

    out = preprocessing_node(state)
    print("=== 전처리 노드 단독 실행 ===")
    d = out["preprocessing"]
    print("SLM 결정 — 피처모드:", d["features_mode"], "/ 스케일러:", d["scaler"])
    print("실제 사용 피처:", len(d["feature_cols"]), "개 →", d["feature_cols"][:6], "...")
    print("비용 — 토큰:", out["prompt_tokens"], out["completion_tokens"])
