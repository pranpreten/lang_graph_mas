"""
data_prep.py — 데이터 준비 모듈 (실험의 '출발선', 고정 부분).

원본 C-MAPSS 파일을 실험에 바로 쓸 형태로 다듬는다.
조율 수준과 무관하게 항상 동일 → L1~L4 비교의 공정성 보장.

함수:
  [v] load_raw()         : train/test/RUL 3파일 → pandas 표
  [v] add_rul(df)        : RUL 라벨 계산해서 컬럼 추가
  [v] make_task(task)    : 회귀 / 분류 / 이상탐지
  [v] data_summary(X,task): 인지 단계가 SLM에 줄 요약
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as c

FEATURE_COLS = [f"op{i}" for i in (1, 2, 3)] + [f"s{i}" for i in range(1, 22)]


def load_raw():
    """원본 C-MAPSS FD001 파일 3개를 읽어 pandas 표로 돌려준다."""
    train = pd.read_csv(c.TRAIN_PATH, sep=r"\s+", header=None, names=c.CMAPSS_COLS)
    test  = pd.read_csv(c.TEST_PATH,  sep=r"\s+", header=None, names=c.CMAPSS_COLS)
    rul   = pd.read_csv(c.RUL_PATH,   sep=r"\s+", header=None, names=["RUL_true"])
    return train, test, rul


def add_rul(df):
    """train 데이터의 각 행에 RUL(잔여수명) 라벨 컬럼을 추가해서 돌려준다."""
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=c.RUL_CAP)
    return df


def _last_cycle_per_engine(df):
    """각 엔진의 '마지막 사이클' 행만 뽑는다 (test 예측 시점 = 엔진당 1행)."""
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def make_task(task):
    """태스크별로 (X_train, y_train, X_test, y_test) 를 만든다.

    공통: test는 엔진당 마지막 행(100개)이 예측 대상.
    정규화·피처 선택은 여기서 안 함 (그건 SLM의 전처리 결정).
    """
    train, test, rul = load_raw()
    train = add_rul(train)

    X_train = train[FEATURE_COLS]
    test_last = _last_cycle_per_engine(test)
    X_test = test_last[FEATURE_COLS]
    rul_true = rul["RUL_true"]

    if task == "regression":
        y_train = train["RUL"]
        y_test  = rul_true.clip(upper=c.RUL_CAP)
        return X_train, y_train, X_test, y_test

    if task == "classification":
        y_train = (train["RUL"] <= c.FAIL_HORIZON).astype(int)
        y_test  = (rul_true     <= c.FAIL_HORIZON).astype(int)
        return X_train, y_train, X_test, y_test

    if task == "anomaly":
        # 비지도: '정상'으로 학습 = 각 엔진 초기 HEALTHY_HEAD(30) 사이클만.
        #   → y_train 없음(None). 고장 라벨을 학습에 쓰지 않는다.
        healthy = train[train["cycle"] <= c.HEALTHY_HEAD]
        X_train = healthy[FEATURE_COLS]
        y_train = None
        # 평가용 라벨: 고장 임박(RUL ≤ 30) 엔진 = 이상(1). AUC 채점용.
        y_test = (rul_true <= c.FAIL_HORIZON).astype(int)
        return X_train, y_train, X_test, y_test

    raise NotImplementedError(f"'{task}' 미구현")


def data_summary(X_train, task):
    """인지(perception) 단계가 SLM에 넘길 '데이터 요약'을 만든다.

    SLM은 raw 수만 행을 다 못 보므로, 스키마+통계 요약만 준다.
    포함: 표 크기, 피처 목록, 타깃 설명, 피처별 기초통계, 저분산(거의 상수) 피처 표시.
    """
    spec = c.TASK_SPECS[task]
    desc = X_train.describe().T                              # 피처별 mean/std/min/max
    low_var = desc.index[desc["std"] < 1e-6].tolist()        # 거의 상수인 센서

    return {
        "task": task,
        "kind": spec["kind"],
        "target_desc": spec.get("target", "없음(비지도)"),
        "metric": spec["metric"],
        "n_samples": int(X_train.shape[0]),
        "n_features": int(X