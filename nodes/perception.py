"""
perception.py — ① 인지(Perception) 노드 (SLM).

하는 일:
  1) 이 태스크의 데이터를 불러와 '요약'을 만든다 (data_prep.data_summary).
  2) SLM에게 그 요약을 보여주고 "무슨 데이터/태스크인지" 파악하게 한다.
  3) 요약 + SLM 관찰 + 비용(토큰·호출)을 state에 기록해 돌려준다.

노드 = state를 받아 → 일하고 → '바뀐 부분만' dict로 돌려주는 함수.
"""
import os
import sys
import json
from functools import lru_cache

# config, llm, data_prep 를 불러오기 위한 경로 설정
HERE = os.path.dirname(os.path.abspath(__file__))      # .../nodes
ROOT = os.path.dirname(HERE)                            # .../rxm_experiment
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))
import config as c
import llm
import data_prep as dp


@lru_cache(maxsize=None)
def load_task_data(task):
    """태스크 데이터를 불러온다. (캐시: 같은 태스크 반복 로드 방지)
    무거운 X/y 는 state에 안 넣고, 필요할 때 이 함수로 가져온다."""
    return dp.make_task(task)


def perception_node(state):
    task = state["task"]

    # 1) 데이터 불러오고 요약 만들기
    X_train, y_train, X_test, y_test = load_task_data(task)
    summary = dp.data_summary(X_train, task)

    # 2) SLM에게 데이터 파악 요청
    system = ("너는 스마트 제조 RxM 파이프라인의 '인지' 에이전트다. "
              "주어진 데이터 요약을 보고 어떤 데이터이고 무슨 태스크인지 한두 문장으로 파악하라.")
    user = (f"요청: {state['request']}\n"
            f"태스크 종류: {task}\n"
            f"데이터 요약: {json.dumps(summary, ensure_ascii=False)}")
    observation, ptok, ctok = llm.call_slm(system, user)

    # 3) 바뀐 부분만 돌려준다 (LangGraph가 기존 state에 합쳐줌)
    return {
        "data_summary": summary,
        "slm_calls": 1,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "stage_logs": [{"stage": "perception", "slm_calls": 1,
                        "prompt_tokens": ptok, "completion_tokens": ctok,
                        "note": observation[:80]}],
        "current_stage": "perception",
    }


if __name__ == "__main__":
    # 노드를 단독으로 한 번 돌려본다 (그래프 없이, 더미 SLM으로)
    test_state = {"task": "regression",
                  "request": c.TASK_SPECS["regression"]["request"]}
    out = perception_node(test_state)
    print("=== 인지 노드 단독 실행 (더미) ===")
    print("요약 — 피처 수:", out["data_summary"]["n_features"],
          "| 저분산 센서:", out["data_summary"]["low_variance_features"])
    print("SLM 관찰:", out["stage_logs"][0]["note"])
    print("비용 — SLM 호출:", out["slm_calls"], "/ 토큰:", out["prompt_tokens"], out["completion_tokens"])
    print("현재 단계:", out["current_stage"])
