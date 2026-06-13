"""perception.py — ① 인지 노드 (SLM). 프롬프트는 prompts.py."""
import os, sys, json
from functools import lru_cache
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "data")): sys.path.insert(0, p)
import config as c
import llm, prompts
import data_prep as dp
from _ctx import commander_prefix

@lru_cache(maxsize=None)
def load_task_data(task):
    return dp.make_task(task)

def perception_node(state):
    task = state["task"]
    X_train, y_train, X_test, y_test = load_task_data(task)
    summary = dp.data_summary(X_train, task)
    user = (commander_prefix(state) +
            f"요청: {state['request']}\n태스크 종류: {task}\n"
            f"데이터 요약: {json.dumps(summary, ensure_ascii=False)}")
    observation, ptok, ctok = llm.call_slm(prompts.PERCEPTION_SYSTEM, user)
    return {
        "data_summary": summary, "commander_input": "",
        "slm_calls": 1, "prompt_tokens": ptok, "completion_tokens": ctok,
        "stage_logs": [{"stage": "perception", "slm_calls": 1, "prompt_tokens": ptok,
                        "completion_tokens": ctok, "note": observation[:80]}],
        "current_stage": "perception",
    }

if __name__ == "__main__":
    s = {"task": "regression", "request": c.TASK_SPECS["regression"]["request"]}
    print(perception_node(s)["data_summary"]["n_features"], "피처")
