"""
commander.py — 커맨더(LLM) 행동 모음.

레벨 정책에 따라 호출되는 커맨더의 4가지 행동:
  route(state)        : 시작 시 작업 분배 ("이건 RUL 회귀다")  [전 레벨 공통]
  guide(state, stage) : 단계 시작 가이드 (pre)               [L3·L4]
  review(state, stage): 단계 종료 검토·승인/재시도 판정 (post) [L2·L3·L4]
  control_ml(state)   : 분석 단계 모델·HP 직접 지정          [L4만]
"""
# TODO: B3에서 구현
