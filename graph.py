"""
graph.py — LangGraph 조립.

단일 마스터 그래프 + 정책 주입:
  · 4단계를 pre→agent→post 래퍼로 감싸 노드 등록
  · pre/post는 INTERVENTION_POLICY[level] 읽어 커맨더 호출 or 통과
  · post '재시도' 판정 시 같은 단계로 되돌리는 conditional_edge
  · checkpointer로 실패 복구(완주율 측정)

채울 함수: build_graph()  → compiled graph
"""
# TODO: B4에서 구현
