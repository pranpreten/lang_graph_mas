"""
analysis.py — ③ 분석(Analysis) 노드.

두 부분:
  · ML 코어  : 모델 레지스트리 + 전처리 적용 + 학습·예측·채점 (sklearn, 모델호출 없음)
  · analysis_node : SLM(또는 L4 커맨더)이 모델을 고르고 → ML 코어 실행 → 결과 저장

ML 코어는 SLM이 필요 없어 단독 테스트 가능. (진짜 ML이 도는 유일한 곳)
"""
import os
import sys
import inspect
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "data")):
    sys.path.insert(0, p)
import config as c
import llm
from _parse import extract_json

# sklearn 모델·도구
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier,
                              IsolationForest)
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, r2_score, mean_squared_error, roc_auc_score

# [조각 1] 모델 레지스트리: SLM이 주는 문자열 → 실제 sklearn 클래스
MODEL_REGISTRY = {
    "LinearRegression": LinearRegression,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "SVC": SVC,
    "IsolationForest": IsolationForest,
    "LocalOutlierFactor": LocalOutlierFactor,
    "OneClassSVM": OneClassSVM,
}


def build_model(name, hyperparams):
    """문자열 모델명 + 하이퍼파라미터 → sklearn 추정기 객체."""
    cls = MODEL_REGISTRY[name]
    params = dict(hyperparams or {})
    sig = inspect.signature(cls).parameters
    if "random_state" in sig:                       # 재현성: 지원 모델엔 시드 고정
        params.setdefault("random_state", c.RANDOM_STATE)
    if name == "LocalOutlierFactor":                # 새 데이터 점수 매기려면 필수
        params.setdefault("novelty", True)
    return cls(**params)


# [조각 2] 전처리 적용: 피처 선택 + 스케일링 (train에 fit, test에 transform)
def apply_preprocessing(X_train, X_test, feature_cols, scaler):
    Xtr = X_train[feature_cols].to_numpy()
    Xte = X_test[feature_cols].to_numpy()
    if scaler == "standard":
        sc = StandardScaler()
    elif scaler == "minmax":
        sc = MinMaxScaler()
    else:
        return Xtr, Xte
    return sc.fit_transform(Xtr), sc.transform(Xte)


# [조각 3] 실제 학습·예측·채점
def run_ml(task, decision, prep, data):
    """반환: (score, detail_dict, prediction_dict)."""
    X_train, y_train, X_test, y_test = data
    Xtr, Xte = apply_preprocessing(X_train, X_test, prep["feature_cols"], prep["scaler"])
    model = build_model(decision["model"], decision.get("hyperparams", {}))

    if task == "regression":
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        score = float(r2_score(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        return score, {"rmse": rmse}, {"pred_rul": pred.tolist()}

    if task == "classification":
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        score = float(f1_score(y_test, pred))
        proba = (model.predict_proba(Xte)[:, 1].tolist()
                 if hasattr(model, "predict_proba") else pred.astype(float).tolist())
        return score, {}, {"pred_class": pred.tolist(), "proba": proba}

    if task == "anomaly":
        model.fit(Xtr)                              # 비지도: 정상 데이터만 (y 없음)
        raw = (model.decision_function(Xte) if hasattr(model, "decision_function")
               else model.score_samples(Xte))
        scores = (-np.asarray(raw)).tolist()        # 높을수록 이상이 되게 부호 반전
        score = float(roc_auc_score(y_test, scores))
        return score, {}, {"anomaly_score": scores}

    raise ValueError(f"알 수 없는 task: {task}")


# ── analysis_node (SLM 결정 → ML 코어) — 실제 SLM 호출은 PC에서 ──
def analysis_node(state):
    task = state["task"]
    prep = state["preprocessing"]
    candidates = c.MODEL_CANDIDATES[task]

    system = ("너는 RxM 파이프라인의 '분석' 에이전트다. 아래 후보 모델 중 하나를 골라 학습 계획을 정하라.\n"
              '반드시 JSON 하나만 출력(설명 금지): {"model": "<후보 중 하나>", "hyperparams": {}}')
    user = (f"태스크: {task}\n후보 모델: {candidates}\n"
            f"피처 {len(prep['feature_cols'])}개, 스케일러={prep['scaler']}")
    text, ptok, ctok = llm.call_slm(system, user)

    raw = extract_json(text, {"model": candidates[0], "hyperparams": {}})
    model_name = raw.get("model")
    if model_name not in candidates:                # 후보 밖이면 기본값 (보호)
        model_name = candidates[0]
    hp = raw.get("hyperparams") if isinstance(raw.get("hyperparams"), dict) else {}
    decision = {"model": model_name, "hyperparams": hp, "decided_by": "slm"}

    # 실제 학습 (실패하면 잡아서 error 기록 → 완주율 측정에 반영)
    import data_prep as dp
    data = dp.make_task(task)
    try:
        score, detail, prediction = run_ml(task, decision, prep, data)
        err = None
    except Exception as e:
        score, detail, prediction, err = None, {}, {}, f"analysis 실패: {e}"

    return {
        "decision": decision, "score": score, "score_detail": detail,
        "prediction": prediction, "error": err,
        "slm_calls": 1, "prompt_tokens": ptok, "completion_tokens": ctok,
        "stage_logs": [{"stage": "analysis", "slm_calls": 1,
                        "prompt_tokens": ptok, "completion_tokens": ctok,
                        "note": f"{model_name} → score={score}"}],
        "current_stage": "analysis",
    }


if __name__ == "__main__":
    # ML 코어만 테스트 (SLM 없이, 모델 하드코딩) — 샌드박스에서 실제 데이터로 검증
    import data_prep as dp
    tests = {
        "regression":     {"model": "RandomForestRegressor", "hyperparams": {"n_estimators": 100}},
        "classification": {"model": "RandomForestClassifier", "hyperparams": {"n_estimators": 100}},
        "anomaly":        {"model": "IsolationForest", "hyperparams": {"contamination": 0.1}},
    }
    print("=== 분석 ML 코어 테스트 (실제 C-MAPSS 데이터) ===")
    for task, decision in tests.items():
        Xtr, ytr, Xte, yte = dp.make_task(task)
        summary = dp.data_summary(Xtr, task)
        low = set(summary["low_variance_features"])
        prep = {"feature_cols": [f for f in summary["features"] if f not in low],
                "scaler": "standard"}
        score, detail, pred = run_ml(task, decision, prep, (Xtr, ytr, Xte, yte))
        print(f"[{task:14s}] {decision['model']:26s} → score={score:.3f}  {detail}")
