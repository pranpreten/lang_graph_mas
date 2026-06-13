"""
graph.py — LangGraph 조립. 레벨별로 '명시적으로' 분리 (읽기 쉬움).

build_graph_L1 / L2 / L3 / L4  — 각 함수가 그 레벨의 노드·엣지를 전부 나열.
build_graph(level) 로 골라서 부른다.

노드 동작을 찍어내는 작은 헬퍼(재사용):
  _make_guide_node(stage)  : 그 단계용 가이드 노드
  _make_review_node(stage) : 검토 노드 (review + 재시도 예산 + 피드백 주입)
  _make_router(stage, nxt) : 검토 후 분기 판단 (재시도→단계 / 통과→nxt)
"""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, os.path.join(ROOT, "nodes"), os.path.join(ROOT, "data")):
    sys.path.insert(0, p)

from langgraph.graph import StateGraph, START, END
import config as c
from state import RunState, new_run_state
import commander
from perception import perception_node
from preprocessing import preprocessing_node
from analysis import analysis_node
from prescription import prescription_node


# ── 노드 동작 헬퍼 ─────────────────────────────────────────────
def _make_guide_node(stage):
    def _g(state): return commander.guide(state, stage)
    return _g

def _make_review_node(stage):
    """검토 노드: review + 재시도 예산(MAX_RETRIES) + 재시도 시 피드백 주입."""
    def _r(state):
        upd = commander.review(state, stage)
        count = state.get("retry_counts", {}).get(stage, 0)
        if upd.get("last_verdict") == "재시도" and count >= c.MAX_RETRIES:
            upd["last_verdict"] = "통과"
            upd["last_feedback"] = f"(재시도 {c.MAX_RETRIES}회 초과 → 진행)"
        elif upd.get("last_verdict") == "재시도":
            nc = dict(state.get("retry_counts", {})); nc[stage] = count + 1
            upd["retry_counts"] = nc
            upd["commander_input"] = upd.get("last_feedback", "")
        return upd
    return _r

def _make_router(stage, nxt):
    def _route(state):
        return stage if state.get("last_verdict") == "재시도" else nxt
    return _route


# ── L1: route → 4단계 직선 (검토·가이드 없음) ──────────────────
def build_graph_L1():
    g = StateGraph(RunState)
    g.add_node("route", commander.route)
    g.add_node("perception", perception_node)
    g.add_node("preprocessing", preprocessing_node)
    g.add_node("analysis", analysis_node)
    g.add_node("prescription", prescription_node)

    g.add_edge(START, "route")
    g.add_edge("route", "perception")
    g.add_edge("perception", "preprocessing")
    g.add_edge("preprocessing", "analysis")
    g.add_edge("analysis", "prescription")
    g.add_edge("prescription", END)
    return g.compile()


# ── L2: 각 단계 뒤 검토 + 재시도 루프 ──────────────────────────
def build_graph_L2():
    g = StateGraph(RunState)
    g.add_node("route", commander.route)
    g.add_node("perception", perception_node)
    g.add_node("preprocessing", preprocessing_node)
    g.add_node("analysis", analysis_node)
    g.add_node("prescription", prescription_node)
    g.add_node("review_perception", _make_review_node("perception"))
    g.add_node("review_preprocessing", _make_review_node("preprocessing"))
    g.add_node("review_analysis", _make_review_node("analysis"))
    g.add_node("review_prescription", _make_review_node("prescription"))

    g.add_edge(START, "route")
    g.add_edge("route", "perception")

    g.add_edge("perception", "review_perception")
    g.add_conditional_edges("review_perception", _make_router("perception", "preprocessing"),
                            {"perception": "perception", "preprocessing": "preprocessing"})

    g.add_edge("preprocessing", "review_preprocessing")
    g.add_conditional_edges("review_preprocessing", _make_router("preprocessing", "analysis"),
                            {"preprocessing": "preprocessing", "analysis": "analysis"})

    g.add_edge("analysis", "review_analysis")
    g.add_conditional_edges("review_analysis", _make_router("analysis", "prescription"),
                            {"analysis": "analysis", "prescription": "prescription"})

    g.add_edge("prescription", "review_prescription")
    g.add_conditional_edges("review_prescription", _make_router("prescription", END),
                            {"prescription": "prescription", END: END})
    return g.compile()


# ── L3: 각 단계 앞 가이드 + 뒤 검토 ────────────────────────────
def build_graph_L3():
    g = StateGraph(RunState)
    g.add_node("route", commander.route)
    g.add_node("guide_perception", _make_guide_node("perception"))
    g.add_node("guide_preprocessing", _make_guide_node("preprocessing"))
    g.add_node("guide_analysis", _make_guide_node("analysis"))
    g.add_node("guide_prescription", _make_guide_node("prescription"))
    g.add_node("perception", perception_node)
    g.add_node("preprocessing", preprocessing_node)
    g.add_node("analysis", analysis_node)
    g.add_node("prescription", prescription_node)
    g.add_node("review_perception", _make_review_node("perception"))
    g.add_node("review_preprocessing", _make_review_node("preprocessing"))
    g.add_node("review_analysis", _make_review_node("analysis"))
    g.add_node("review_prescription", _make_review_node("prescription"))

    g.add_edge(START, "route")
    g.add_edge("route", "guide_perception")

    g.add_edge("guide_perception", "perception")
    g.add_edge("perception", "review_perception")
    g.add_conditional_edges("review_perception", _make_router("perception", "guide_preprocessing"),
                            {"perception": "perception", "guide_preprocessing": "guide_preprocessing"})

    g.add_edge("guide_preprocessing", "preprocessing")
    g.add_edge("preprocessing", "review_preprocessing")
    g.add_conditional_edges("review_preprocessing", _make_router("preprocessing", "guide_analysis"),
                            {"preprocessing": "preprocessing", "guide_analysis": "guide_analysis"})

    g.add_edge("guide_analysis", "analysis")
    g.add_edge("analysis", "review_analysis")
    g.add_conditional_edges("review_analysis", _make_router("analysis", "guide_prescription"),
                            {"analysis": "analysis", "guide_prescription": "guide_prescription"})

    g.add_edge("guide_prescription", "prescription")
    g.add_edge("prescription", "review_prescription")
    g.add_conditional_edges("review_prescription", _make_router("prescription", END),
                            {"prescription": "prescription", END: END})
    return g.compile()


# ── L4: L3 + 분석 앞 control_ml(커맨더가 모델 지정) ────────────
def build_graph_L4():
    g = StateGraph(RunState)
    g.add_node("route", commander.route)
    g.add_node("guide_perception", _make_guide_node("perception"))
    g.add_node("guide_preprocessing", _make_guide_node("preprocessing"))
    g.add_node("guide_analysis", _make_guide_node("analysis"))
    g.add_node("guide_prescription", _make_guide_node("prescription"))
    g.add_node("control_ml", commander.control_ml)          # ★ L4 전용
    g.add_node("perception", perception_node)
    g.add_node("preprocessing", preprocessing_node)
    g.add_node("analysis", analysis_node)
    g.add_node("prescription", prescription_node)
    g.add_node("review_perception", _make_review_node("perception"))
    g.add_node("review_preprocessing", _make_review_node("preprocessing"))
    g.add_node("review_analysis", _make_review_node("analysis"))
    g.add_node("review_prescription", _make_review_node("prescription"))

    g.add_edge(START, "route")
    g.add_edge("route", "guide_perception")

    g.add_edge("guide_perception", "perception")
    g.add_edge("perception", "review_perception")
    g.add_conditional_edges("review_perception", _make_router("perception", "guide_preprocessing"),
                            {"perception": "perception", "guide_preprocessing": "guide_preprocessing"})

    g.add_edge("guide_preprocessing", "preprocessing")
    g.add_edge("preprocessing", "review_preprocessing")
    g.add_conditional_edges("review_preprocessing", _make_router("preprocessing", "guide_analysis"),
                            {"preprocessing": "preprocessing", "guide_analysis": "guide_analysis"})

    # 분석만 다름: 가이드 → control_ml → 분석 → 검토
    g.add_edge("guide_analysis", "control_ml")
    g.add_edge("control_ml", "analysis")
    g.add_edge("analysis", "review_analysis")
    g.add_conditional_edges("review_analysis", _make_router("analysis", "guide_prescription"),
                            {"analysis": "analysis", "guide_prescription": "guide_prescription"})

    g.add_edge("guide_prescription", "prescription")
    g.add_edge("prescription", "review_prescription")
    g.add_conditional_edges("review_prescription", _make_router("prescription", END),
                            {"prescription": "prescription", END: END})
    return g.compile()


BUILDERS = {"L1": build_graph_L1, "L2": build_graph_L2,
            "L3": build_graph_L3, "L4": build_graph_L4}


def build_graph(level):
    """레벨에 맞는 그래프를 골라서 만든다."""
    return BUILDERS[level]()


if __name__ == "__main__":
    level = next((a for a in sys.argv[1:] if a in BUILDERS), "L1")
    graph = build_graph(level)
    print(f"✓ {level} 컴파일 OK")
    print("  노드:", [n for n in graph.get_graph().nodes.keys() if not n.startswith("__")])
    task = next((a for a in sys.argv[1:] if a in c.TASKS), "regression")
    if "--run" in sys.argv:
        init = new_run_state(run_id=f"demo-{level}-{task}", level=level, task=task,
                             repeat=0, request=c.TASK_SPECS[task]["request"], seed=42)
        r = graph.invoke(init)
        print(f"\n=== {level} 한 바퀴 ===")
        print("완주:", r.get("completed"), "| 점수:", r.get("score"), "| 모델:", r.get("decision", {}).get("model"))
        print("처방:", r.get("prescription", {}).get("counts"))
        print("커맨더 호출:", r.get("commander_calls"), "| SLM:", r.get("slm_calls"), "| 재시도:", r.get("retry_counts", {}))
