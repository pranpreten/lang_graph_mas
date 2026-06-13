"""
llm.py — 커맨더(Claude)·SLM(Ollama) 호출 + trace 기록 + 타임아웃.
타임아웃 걸어서 한 호출이 멈춰도 무한 대기 안 하고 에러 → runner가 실패로 기록하고 다음 런 진행.
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
TRACE = True
_CTX = {"run_id": None}
COMMANDER_TIMEOUT = 120      # 초
SLM_TIMEOUT = 180            # 초


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
        _anthropic_client = Anthropic(timeout=COMMANDER_TIMEOUT)
    return _anthropic_client


_ollama_client = None
def _get_ollama():
    global _ollama_client
    if _ollama_client is None:
        import ollama
        _ollama_client = ollama.Client(timeout=SLM_TIMEOUT)
    return _ollama_client


def call_commander(system, user_msg, max_tokens=1024):
    """커맨더 LLM(Claude Opus 4.8) 호출."""
    resp = _get_anthropic().messages.create(
        model=c.COMMANDER_MODEL, max_tokens=max_tokens,
        system=system, messages=[{"role": "user", "content": user_msg}],
    )
    text = resp.content[0].text
    ptok, ctok = resp.usage.input_tokens, resp.usage.output_tokens
    _trace("commander", system, user_msg, text, ptok, ctok)
    return text, ptok, ctok


def call_slm(system, user_msg):
    """SLM(Ollama) 호출. qwen3 사고(thinking) 끄기 + 출력 상한. 타임아웃 적용."""
    client = _get_ollama()
    # qwen3 soft switch: think 파라미터를 서버가 무시해도 /no_think 토큰은 항상 먹힘.
    # 현재 턴(user 메시지)에 넣는 게 가장 확실해서 system·user 양쪽에 박는다.
    user_msg = user_msg + " /no_think"
    kwargs = dict(model=c.SLM_MODEL,
                  messages=[{"role": "system", "content": system + " /no_think"},
                            {"role": "user", "content": user_msg}],
                  options=c.SLM_OPTIONS)
    try:
        resp = client.chat(think=False, **kwargs)
    except TypeError:
        resp = client.chat(**kwargs)          # 구버전: /no_think 토큰만으로 처리
    text = resp["message"]["content"]
    ptok = resp.get("prompt_eval_count", 0); ctok = resp.get("eval_count", 0)
    _trace("slm", system, user_msg, text, ptok, ctok)
    return text, ptok, ctok


if __name__ == "__main__":
    set_run_id("smoke-test")
    print("커맨더:", call_commander("너는 커맨더다.", "한 문장 자기소개")[0])
    print("SLM   :", call_slm("너는 분석 에이전트다.", "한 문장 자기소개")[0])
