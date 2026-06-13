"""
prescription.py — ④ 처방(Prescription) 노드.

하는 일:
  1) 분석의 예측값을 정책 임계치(config.PRESCRIPTION_POLICY)에 통과 → 엔진별 조치.
     (임계치는 고정 — SLM이 정하지 않는다. 결정론적.)
  2) SLM이 조치 분포를 보고 '근거 + 전체 긴급도'를 한 번 서술 (추론형, 1회 호출).
  3) 엔진별 조치 + 분포 + 근거를 state에 저장. 여기까지 오면 런 완주.

일관성 측정 대상 = 엔진별 조치 목록 (예측이 흔들리면 조치가 뒤집힘).
"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "data")):
    sys.path.insert(0, p)
import config as c
import llm
from _parse import extract_json


def _action_from_bands(value, bands):
    """값을 정책 밴드에 넣어 조치 문자열로. 밴드: (low, high, action), low<=v<high."""
    for low, high, action in bands:
        if low <= value < high:
            return action
    return bands[-1][2]


def _to_policy_values(task, prediction):
    """분석 예측 → 정책이 보는 값 리스트."""
    if task == "regression":
        return prediction["pred_rul"]                       # RUL 값들
    if task == "classification":
        return prediction["proba"]                          # 고장임박 확률들
    if task == "anomaly":
        # 이상 점수 → 플래그(점수>0 이면 이상=1)
        return [1 if s > 0 else 0 for s in prediction["anomaly_score"]]
    raise ValueError(task)


def apply_policy(task, prediction):
    """엔진별 조치 리스트 반환 (결정론적)."""
    values = _to_policy_values(task, prediction)
    bands = c.PRESCRIPTION_POLICY[task]["bands"]
    return [_action_from_bands(v, bands) for v in values]


def prescription_node(state):
    task = state["task"]
    prediction = state.get("prediction") or {}

    # 분석이 실패해 예측이 없으면 처방 불가 → error
    if not prediction:
        return {"error": "prescription: 예측값 없음(분석 실패 추정)",
                "current_stage": "prescription"}

    actions = apply_policy(task, prediction)
    counts = dict(Counter(actions))

    # SLM: 조치 분포 보고 근거 + 긴급도 (1회). 근거는 자유, urgency는 고정집합.
    system = ("너는 RxM '처방' 에이전트다. 엔진 조치 분포를 보고 정비팀에 전할 "
              "근거를 한두 문장으로 쓰고, 전체 긴급도를 정하라.\n"
              '반드시 JSON 하나만: {"근거": "...", "urgency": "높음" 또는 "중간" 또는 "낮음"}')
    user = f"태스크: {task}\n엔진 100대 조치 분포: {counts}"
    text, ptok, ctok = llm.call_slm(system, user)
    raw = extract_json(text, {"근거": "(자동) 조치 분포 기반 권고", "urgency": "중간"})
    urgency = raw.get("urgency") if raw.get("urgency") in c.URGENCY_LEVELS else "중간"

    prescription = {
        "actions": actions,            # 엔진별 조치 100개 (일관성 측정용)
        "counts": counts,              # 조치 분포
        "근거": raw.get("근거", ""),
        "urgency": urgency,
    }
    return {
        "prescription": prescription,
        "completed": True,             # 여기 도달 = 파이프라인 완주
        "slm_calls": 1, "prompt_tokens": ptok, "completion_tokens": ctok,
        "stage_logs": [{"stage": "prescription", "slm_calls": 1,
                        "prompt_tokens": ptok, "completion_tokens": ctok,
                        "note": str(counts)}],
        "current_stage": "prescription",
    }


if __name__ == "__main__":
    # 임계치 적용(결정론적 핵심)만 테스트 — 분석 예측을 받아 조치로 변환
    sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "data"))
    import data_prep as dp
    import analysis as A
    decisions = {
        "regression":     {"model": "RandomForestRegressor", "hyperparams": {"n_estimators": 100}},
        "classification": {"model": "RandomForestClassifier", "hyperparams": {"n_estimators": 100}},
        "anomaly":        {"model": "IsolationForest", "hyperparams": {"contamination": 0.1}},
    }
    print("=== 처방 임계치 적용 테스트 (실제 예측 → 조치 분포) ===")
    for task, dec in decisions.items():
        Xtr, ytr, Xte, yte = dp.make_task(task)
        std = Xtr.std(numeric_only=True); low = set(std[std < 1e-6].index)
        prep = {"feature_cols": [f for f in Xtr.columns if f not in low], "scaler": "standard"}
        _, _, prediction = A.run_ml(task, dec, prep, (Xtr, ytr, Xte, yte))
        actions = apply_policy(task, prediction)
        print(f"[{task:14s}] 조치 분포:", dict(Counter(actions)))
