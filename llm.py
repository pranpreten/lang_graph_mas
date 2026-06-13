"""
llm.py — 커맨더(Claude Opus 4.8)·SLM(Ollama) 호출 + 전체 trace 기록.

호출할 때마다 (system / 입력 / 응답 / 토큰)을 logs/trace.jsonl 에 한 줄씩 남긴다.
→ "각 에이전트가 단계별로 뭐라고 했는지" 전부 보존 (흐름 재구성·검증·논문 부록용).

준비물 (내 PC/서버):
  pip install anthropic ollama python-dotenv
  .env 에 ANTHROPIC_API_KEY,  Ollama 실행 + `ollama pull qwen3:8b`
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as c

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

TRACE_PATH = os.path.join(c.LOG_DIR, "trace.jsonl")
TRACE = True                      # 전체 대화 기록 on/off
_CTX = {"run_id": None}           # 현재 런 식별 (runner가 설정)


def set_run_id(run_id):
    _CTX["run_id"] = run_id


def _trace(role, system, user, response, ptok, ctok):
    if not TRACE:
        return
    os.makedirs(c.LOG_DIR, exist_ok=True)
    rec = {"run_id": _CTX["run_id"], "role": role, "ts": round(time.time(), 3),
           "system": system, "input": user, "response": response,
           "prompt_tokens": ptok, "completion_tokens": ctok}
    with open(TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


_anthropic_client = None
def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic()
    return _anthropic_client


def call_commander(system, user_msg, max_tokens=1024):
    """커맨더 LLM(Claude Opus 4.8) 호출. (Opus 4.8은 temperature 미지원)"""
    client = _get_anthropic()
    resp = client.messages.create(
        model=c.COMMANDER_MODEL, max_tokens=max_tokens,
        system=system, messages=[{"role": "user", "content": user_msg}],
    )
    text = resp.content[0].text
    ptok, ctok = resp.usage.input_tokens, resp.usage.output_tokens
    _trace("commander", system, user_msg, text, ptok, ctok)
    return text, ptok, ctok


def call_slm(system, user_msg):
    """SLM(Ollama, qwen3:8b) 호출."""
    import ollama
    try:
        resp = ollama.chat(
            model=c.SLM_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user_msg}],
            options=c.SLM_OPTIONS,
            think=False,                     # qwen3 '사고' 끄기 → 속도↑·토큰↓·재현성↑
        )
    except TypeError:
        # 구버전 ollama-python(think 미지원) → 프롬프트에 /no_think 로 대체
        resp = ollama.chat(
            model=c.SLM_MODEL,
            messages=[{"role": "system", "content": system + " /no_think"},
                      {"role": "user", "content": user_msg}],
            options=c.SLM_OPTIONS,
        )
    text = resp["message"]["content"]
    ptok = resp.get("prompt_eval_count", 0); ctok = resp.get("eval_count", 0)
    _trace("slm", system, user_msg, text, ptok, ctok)
    return text, ptok, ctok


if __name__ == "__main__":
    set_run_id("smoke-test")
    print("커맨더:", call_commander("너는 커맨더다.", "한 문장 자기소개")[0])
    print("SLM   :", call_slm("너는 분석 에이전트다.", "한 문장 자기소개")[0])
    print(f"\n→ trace 기록: {TRACE_PATH}")
