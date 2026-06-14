"""
state.py — 그래프 전체를 흐르는 '공유 상태(State)' 정의.

한 번의 런(파이프라인 1회 실행) 동안 이 객체 하나가
  인지 → 전처리 → 분석 → 처방
노드를 지나며 채워진다. 커맨더 개입·SLM 실행·비용·로그가 모두 여기 누적된다.

LangGraph 규칙:
  · State는 TypedDict.
  · 여러 노드가 '누적'해야 하는 필드(로그·비용)는 Annotated + reducer 사용.
"""
from __future__ import annotations
from typing import TypedDict, Annotated, Any, Optional
import operator


def _add_int(a: int, b: int) -> int:      # 누적 reducer: 각 노드가 더한 값을 합산
    return (a or 0) + (b or 0)


class Decision(TypedDict, total=False):
    """분석 단계에서 내려진 ML 결정 (일관성 측정 대상)."""
    model: str                 # 고른 모델명 (MODEL_CANDIDATES 내)
    features: list[str]        # 사용한 피처
    hyperparams: dict[str, Any]
    decided_by: str            # "slm" | "commander"  (L4는 commander)


class StageLog(TypedDict, total=False):
    """단계별 실행 흔적 (비용 분해·디버깅용)."""
    stage: str
    commander_calls: int
    slm_calls: int
    prompt_tokens: int
    completion_tokens: int
    retries: int
    latency_sec: float
    note: str


class RunState(TypedDict, total=False):
    # ── 입력 (런 시작 시 고정) ──────────────────────────────
    run_id: str
    level: str                 # "L1"~"L4"  (독립변수)
    task: str                  # "classification" | "regression" | "anomaly"
    repeat: int                # 반복 회차
    request: str               # 요청 문장 (모든 레벨 동일)
    seed: int

    # ── 파이프라인 중간 산출물 ──────────────────────────────
    data_summary: dict[str, Any]   # 인지: 스키마·통계 (SLM에 주는 요약, raw 아님)
    preprocessing: dict[str, Any]  # 전처리: 결측·스케일·피처 결정
    decision: Decision             # 분석: 모델·피처·HP 결정
    prediction: dict[str, Any]     # 분석: sklearn 학습 결과(예측값/점수)
    prescription: dict[str, Any]   # 처방: {근거, action, urgency}

    # ── 결과 지표 ───────────────────────────────────────────
    score: Optional[float]         # 주 지표 (F1/R²/AUC) — 보조 해석용
    score_detail: dict[str, Any]   # RMSE 등 부가 점수
    completed: bool                # 완주 여부 (안정성 지표)
    error: Optional[str]           # 실패 시 사유

    # ── 비용 (안정성·비용 1차 지표) ─────────────────────────
    commander_calls: Annotated[int, _add_int]
    slm_calls: Annotated[int, _add_int]
    prompt_tokens: Annotated[int, _add_int]
    completion_tokens: Annotated[int, _add_int]

    # ── 커맨더 개입 채널 (LangGraph가 유지하도록 명시 — 없으면 노드 출력이 버려짐) ──
    commander_note: str            # route 분배 메모
    commander_input: str           # 가이드/검토 피드백 → 다음 SLM 프롬프트에 주입(L3+)
    forced_decision: dict[str, Any]  # L4 control_ml이 지정한 모델·HP
    last_verdict: str              # 검토 판정: 통과|재시도
    last_feedback: str             # 검토 사유
    retry_counts: dict[str, int]   # 단계별 재시도 횟수

    # ── 로그 누적 ───────────────────────────────────────────
    stage_logs: Annotated[list[StageLog], operator.add]
    current_stage: str             # 현재 처리 중 단계 (retry 라우팅용)


def new_run_state(run_id: str, level: str, task: str, repeat: int,
                  request: str, seed: int) -> RunState:
    """런 1회분 초기 상태 생성."""
    return RunState(
        run_id=run_id, level=level, task=task, repeat=repeat,
        request=request, seed=seed,
        commander_calls=0, slm_calls=0, prompt_tokens=0, completion_tokens=0,
        stage_logs=[], completed=False, error=None, score=None, score_detail={},
    )
