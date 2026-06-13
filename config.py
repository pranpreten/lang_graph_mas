"""
config.py — 실험의 모든 '고정 상수'를 한 곳에 모은 파일. (C-MAPSS FD001 기준)

설계 원칙(우리가 정한 것):
  · 독립변수는 '조율 수준(L1~L4)' 단 하나. 그 외 모든 것은 여기서 고정한다.
  · 무게중심은 '성능'이 아니라 '안정성·비용'. 성능(F1/R²/AUC)은 보조 지표.
  · 공정성: 모델 후보군은 전 레벨 공통. L4도 이 안에서 '검증·승인'만(모델 발명 금지).
  · 처방 임계치는 SLM이 정하지 않는다. 여기 정책표를 적용만 한다.
"""
import os

# ──────────────────────────────────────────────────────────────
# 1. 실험 규모
# ──────────────────────────────────────────────────────────────
N_REPEATS = 10          # 한 조건당 반복 횟수 (기존 5 → 10: 검정력·분산 추정)
N_PILOT   = 5           # 파일럿: 본 실험 전 분산 확인용

LEVELS = ["L1", "L2", "L3", "L4"]                      # 조율 수준 (독립변수)
TASKS  = ["classification", "regression", "anomaly"]  # 3개 태스크

# 방식 A(전수비교): 4레벨 × 3태스크 × N_REPEATS = 120런
RUN_PLAN_A = {"levels": LEVELS, "tasks": TASKS, "repeats": N_REPEATS}

# ──────────────────────────────────────────────────────────────
# 2. 모델 (전부 버전 핀 고정 — 재현성)
# ──────────────────────────────────────────────────────────────
# 커맨더: 외부 API (강한 추론, 호출 적음). 로컬 실행 부하 0.
COMMANDER_MODEL = "claude-opus-4-8"   # ← 실제 사용 버전으로 핀 고정 (최신·최강)
COMMANDER_TEMPERATURE = 0.0
COMMANDER_THINKING = False            # 재현성: thinking 비활성

# SLM: 로컬(Ollama) 실행자. VRAM엔 1개만 올라가면 됨(4단계 동일 모델 재사용).
SLM_MODEL = "qwen3:8b"   # 서버 GPU에서 실행 (로컬 GTX1650 4GB는 부족 → 서버로)
SLM_OPTIONS = {                       # Ollama 결정성 옵션 (완전 결정론 불가, 노이즈 최소화)
    "temperature": 0.0, "top_p": 0.0, "top_k": 1,
    "seed": 42, "num_ctx": 8192,
}
SLM_VARIANTS_OPTIONAL = ["phi4-mini", "qwen3:8b", "qwen3:30b"]  # (선택) SLM 역량 보조비교

# ──────────────────────────────────────────────────────────────
# 3. 공통 모델 후보군 — 공정성의 핵심
#    모든 레벨(L1~L4)은 '이 안에서만' 모델을 고른다. L4도 발명 금지.
# ──────────────────────────────────────────────────────────────
MODEL_CANDIDATES = {
    "classification": ["LogisticRegression", "RandomForestClassifier",
                       "GradientBoostingClassifier", "SVC"],
    "regression":     ["LinearRegression", "RandomForestRegressor",
                       "GradientBoostingRegressor"],
    "anomaly":        ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"],
}

HP_SEARCH_SPACE = {
    "RandomForestClassifier":     {"n_estimators": [100, 300], "max_depth": [None, 10]},
    "GradientBoostingClassifier": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "SVC":                        {"C": [1.0, 10.0], "kernel": ["rbf"]},
    "LogisticRegression":         {"C": [1.0, 10.0], "max_iter": [1000]},
    "RandomForestRegressor":      {"n_estimators": [100, 300], "max_depth": [None, 10]},
    "GradientBoostingRegressor":  {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "LinearRegression":           {},
    "IsolationForest":            {"contamination": [0.1]},
    "LocalOutlierFactor":         {"contamination": [0.1], "novelty": [True]},
    "OneClassSVM":                {"nu": [0.05]},
}

RANDOM_STATE = 42        # sklearn 학습 결정론 고정

# ──────────────────────────────────────────────────────────────
# 4. C-MAPSS 데이터 / 태스크 정의
# ──────────────────────────────────────────────────────────────
# 원본 컬럼: unit, cycle, op1~3, s1~s21 (공백구분, 헤더없음)
CMAPSS_COLS = (["unit", "cycle"] + [f"op{i}" for i in (1, 2, 3)]
               + [f"s{i}" for i in range(1, 22)])

RUL_CAP      = 125   # RUL 상한 cap (C-MAPSS 문헌 관례)
FAIL_HORIZON = 30    # 분류: RUL ≤ 30사이클이면 '곧 고장(1)'
HEALTHY_HEAD = 30    # 이상탐지: 각 엔진 초기 30사이클을 '정상'으로 보고 학습

# 요청 문장은 모든 레벨에서 동일 (what만, how는 시스템이 정함)
TASK_SPECS = {
    "regression": {
        "request": "엔진들의 잔여수명(RUL)을 예측하는 모델을 만들어줘",
        "target":  "RUL",
        "metric":  "r2",            # 보조로 RMSE도 기록
        "kind":    "supervised",
    },
    "classification": {
        "request": "엔진들이 곧(30사이클 내) 고장 날지 판별해줘",
        "target":  "fail_soon",     # RUL ≤ FAIL_HORIZON → 1
        "metric":  "f1",
        "kind":    "supervised",
    },
    "anomaly": {
        "request": "엔진 센서 데이터에서 이상 징후를 찾아줘",
        "target":  "is_anomaly",    # 평가에만 사용 (학습엔 정상 데이터만)
        "metric":  "roc_auc",
        "kind":    "unsupervised",
    },
}

# ──────────────────────────────────────────────────────────────
# 5. 조율 수준 정책 — '단일 마스터 그래프 + 정책 주입' 의 핵심
#    레벨마다 그래프를 새로 짜지 않는다. 같은 그래프에서 이 플래그만 읽어 분기.
#      route      : 시작 시 작업 분배 (전 레벨 공통, 항상 1회)
#      pre        : 각 단계 '시작' 시 커맨더 가이드·지시
#      post       : 각 단계 '종료' 시 커맨더 검토·승인(통과/재시도)
#      control_ml : 분석 단계에서 커맨더가 모델·HP 직접 지정(공통 후보군 내)
# ──────────────────────────────────────────────────────────────
INTERVENTION_POLICY = {
    "L1": {"route": True, "pre": False, "post": False, "control_ml": False},  # ≈1회
    "L2": {"route": True, "pre": False, "post": True,  "control_ml": False},  # ≈5회
    "L3": {"route": True, "pre": True,  "post": True,  "control_ml": False},  # ≈9회 (기반 논문[3])
    "L4": {"route": True, "pre": True,  "post": True,  "control_ml": True},   # ≈15회+
}

MAX_RETRIES = 2          # post에서 '재시도' 판정 시 같은 단계 최대 재실행 횟수
STAGES = ["perception", "preprocessing", "analysis", "prescription"]  # 고정 순서

# ──────────────────────────────────────────────────────────────
# 6. 처방 정책표 — 임계치 고정. SLM은 '적용'만(추론형: 근거는 자유, 조치는 고정집합).
#    (값은 문헌 관례 기반. 전 레벨 동일 고정이 철칙.)
# ──────────────────────────────────────────────────────────────
PRESCRIPTION_POLICY = {
    "regression": {              # 입력: 예측 RUL [사이클]
        "type": "rul_cycles",
        "bands": [
            (0,   30,   "즉시정비"),
            (30,  125,  "정비계획수립"),
            (125, 1e9,  "정상운영"),
        ],
    },
    "classification": {          # 입력: 고장임박 확률 [0,1]
        "type": "probability",
        "bands": [
            (0.70, 1.01, "즉시점검"),
            (0.30, 0.70, "다음정비때점검"),
            (0.00, 0.30, "정상운영"),
        ],
    },
    "anomaly": {                 # 입력: 이상 플래그 (1=이상, 0=정상)
        "type": "anomaly_flag",
        "bands": [
            (1, 2, "점검"),
            (0, 1, "정상운영"),
        ],
    },
}

PRESCRIPTION_ACTIONS = {         # 일관성 측정은 'action' 라벨 기준
    "regression":     ["즉시정비", "정비계획수립", "정상운영"],
    "classification": ["즉시점검", "다음정비때점검", "정상운영"],
    "anomaly":        ["점검", "정상운영"],
}
URGENCY_LEVELS = ["높음", "중간", "낮음"]

# ──────────────────────────────────────────────────────────────
# 7. 경로
# ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train_FD001.txt")
TEST_PATH  = os.path.join(DATA_DIR, "test_FD001.txt")
RUL_PATH   = os.path.join(DATA_DIR, "RUL_FD001.txt")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
RESULT_DIR = os.path.join(BASE_DIR, "results")
