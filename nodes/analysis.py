"""analysis.py — ③ 분석. ML 코어 + analysis_node. 프롬프트는 prompts.py."""
import os, sys, inspect
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "data")): sys.path.insert(0, p)
import config as c
import llm, prompts
from _parse import extract_json
from _ctx import commander_prefix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier, IsolationForest)
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (f1_score, r2_score, mean_squared_error, roc_auc_score,
                             precision_score, recall_score, accuracy_score,
                             mean_absolute_error, average_precision_score)
from sklearn.model_selection import GridSearchCV

MODEL_REGISTRY = {
    "LinearRegression": LinearRegression, "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor, "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier, "GradientBoostingClassifier": GradientBoostingClassifier,
    "SVC": SVC, "IsolationForest": IsolationForest,
    "LocalOutlierFactor": LocalOutlierFactor, "OneClassSVM": OneClassSVM,
}

def build_model(name, hyperparams):
    cls = MODEL_REGISTRY[name]; params = dict(hyperparams or {})
    if "random_state" in inspect.signature(cls).parameters:
        params.setdefault("random_state", c.RANDOM_STATE)
    if name == "LocalOutlierFactor":
        params.setdefault("novelty", True)
    return cls(**params)

def apply_preprocessing(X_train, X_test, feature_cols, scaler):
    Xtr = X_train[feature_cols].to_numpy(); Xte = X_test[feature_cols].to_numpy()
    if scaler == "standard": sc = StandardScaler()
    elif scaler == "minmax": sc = MinMaxScaler()
    else: return Xtr, Xte
    return sc.fit_transform(Xtr), sc.transform(Xte)

def run_ml(task, decision, prep, data):
    X_train, y_train, X_test, y_test = data
    Xtr, Xte = apply_preprocessing(X_train, X_test, prep["feature_cols"], prep["scaler"])
    model = build_model(decision["model"], decision.get("hyperparams", {}))
    if task == "regression":
        model.fit(Xtr, y_train); pred = model.predict(Xte)
        return float(r2_score(y_test, pred)), {
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae":  float(mean_absolute_error(y_test, pred)),
        }, {"pred_rul": pred.tolist()}
    if task == "classification":
        model.fit(Xtr, y_train); pred = model.predict(Xte)
        proba = (model.predict_proba(Xte)[:, 1].tolist() if hasattr(model, "predict_proba") else pred.astype(float).tolist())
        return float(f1_score(y_test, pred, zero_division=0)), {
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall":    float(recall_score(y_test, pred, zero_division=0)),
            "accuracy":  float(accuracy_score(y_test, pred)),
        }, {"pred_class": pred.tolist(), "proba": proba}
    if task == "anomaly":
        model.fit(Xtr)
        raw = (model.decision_function(Xte) if hasattr(model, "decision_function") else model.score_samples(Xte))
        scores = (-np.asarray(raw)).tolist()
        return float(roc_auc_score(y_test, scores)), {
            "pr_auc": float(average_precision_score(y_test, scores)),
        }, {"anomaly_score": scores}
    raise ValueError(task)

def tune_hp(task, model_name, prep, data):
    """L4 전용: 후보군 내 하이퍼파라미터를 train CV 그리드서치로 튜닝.
    빈 그리드이거나 이상탐지(비지도 → 라벨 튜닝 시 누수)면 건너뛴다."""
    grid = c.HP_SEARCH_SPACE.get(model_name, {})
    if not grid or task == "anomaly":
        return {}
    X_train, y_train, X_test, y_test = data
    Xtr, _ = apply_preprocessing(X_train, X_test, prep["feature_cols"], prep["scaler"])
    scoring = {"regression": "r2", "classification": "f1"}.get(task)
    try:
        gs = GridSearchCV(build_model(model_name, {}), grid, scoring=scoring, cv=3, n_jobs=-1)
        gs.fit(Xtr, y_train)
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in gs.best_params_.items()}
    except Exception:
        return {}


def analysis_node(state):
    task = state["task"]; prep = state["preprocessing"]; candidates = c.MODEL_CANDIDATES[task]
    import data_prep as dp
    data = dp.make_task(task)
    forced = state.get("forced_decision")
    if forced:
        hp = forced.get("hyperparams") or {}
        if not hp:                              # L4: 커맨더가 후보군 내 HP를 직접 튜닝(그리드서치)
            hp = tune_hp(task, forced["model"], prep, data)
        decision = {"model": forced["model"], "hyperparams": hp,
                    "decided_by": "commander", "tuned": bool(hp)}
        ptok = ctok = 0; slm_n = 0
    else:
        user = (commander_prefix(state) +
                f"태스크: {task}\n후보 모델: {candidates}\n피처 {len(prep['feature_cols'])}개, 스케일러={prep['scaler']}")
        text, ptok, ctok = llm.call_slm(prompts.ANALYSIS_SYSTEM, user)
        raw = extract_json(text, {"model": candidates[0], "hyperparams": {}})
        model_name = raw.get("model") if raw.get("model") in candidates else candidates[0]
        hp = raw.get("hyperparams") if isinstance(raw.get("hyperparams"), dict) else {}
        decision = {"model": model_name, "hyperparams": hp, "decided_by": "slm", "tuned": False}; slm_n = 1
    try:
        score, detail, prediction = run_ml(task, decision, prep, data); err = None
    except Exception as e:
        score, detail, prediction, err = None, {}, {}, f"analysis 실패: {e}"
    return {
        "decision": decision, "score": score, "score_detail": detail,
        "prediction": prediction, "error": err, "commander_input": "",
        "slm_calls": slm_n, "prompt_tokens": ptok, "completion_tokens": ctok,
        "stage_logs": [{"stage": "analysis", "slm_calls": slm_n, "prompt_tokens": ptok,
                        "completion_tokens": ctok, "note": f"{decision['model']}→score={score}"}],
        "current_stage": "analysis",
    }
