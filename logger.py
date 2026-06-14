"""logger.py — 런 1개 결과를 JSONL 한 줄로 기록."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as c
import llm

LOG_PATH = os.path.join(c.LOG_DIR, "runs.jsonl")


def log_run(state, elapsed_sec, path=LOG_PATH):
    """런 결과(state) + 실행시간 → 기록 dict 만들어 파일에 append."""
    dec = state.get("decision", {})
    rec = {
        "run_id": state.get("run_id"),
        "level": state.get("level"),
        "task": state.get("task"),
        "repeat": state.get("repeat"),
        "seed": state.get("seed"),
        "model": dec.get("model"),
        "decided_by": dec.get("decided_by"),
        "tuned": dec.get("tuned", False),
        "score": state.get("score"),
        "score_detail": state.get("score_detail", {}),
        "completed": state.get("completed", False),
        "error": state.get("error"),
        "commander_calls": state.get("commander_calls", 0),
        "slm_calls": state.get("slm_calls", 0),
        "preprocessing_mode": (state.get("preprocessing") or {}).get("features_mode"),
        "scaler": (state.get("preprocessing") or {}).get("scaler"),
        "prompt_tokens": state.get("prompt_tokens", 0),
        "completion_tokens": state.get("completion_tokens", 0),
        "elapsed_sec": round(elapsed_sec, 2),
        "commander_sec": round(llm.TIMING.get("commander_sec", 0.0), 2),
        "slm_sec": round(llm.TIMING.get("slm_sec", 0.0), 2),
        "retry_counts": state.get("retry_counts", {}),
        "actions": state.get("prescription", {}).get("actions"),   # 처방 일관성용
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def read_runs(path=LOG_PATH):
    """기록된 런들을 리스트로 읽어온다 (분석용)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    # 가짜 state로 기록 동작만 확인 (LLM 불필요)
    fake = {
        "run_id": "test-1", "level": "L2", "task": "regression", "repeat": 0, "seed": 42,
        "decision": {"model": "RandomForestRegressor", "decided_by": "slm"},
        "score": 0.81, "score_detail": {"rmse": 17.2}, "completed": True, "error": None,
        "commander_calls": 5, "slm_calls": 4, "prompt_tokens": 1200, "completion_tokens": 300,
        "retry_counts": {"analysis": 1},
        "prescription": {"actions": ["정비계획수립", "즉시정비", "정상운영"]},
    }
    os.makedirs(c.LOG_DIR, exist_ok=True)
    test_path = os.path.join(c.LOG_DIR, "_test_runs.jsonl")
    rec = log_run(fake, elapsed_sec=12.34, path=test_path)
    print("기록된 한 줄:")
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    print("\n다시 읽기:", len(read_runs(test_path)), "건")
