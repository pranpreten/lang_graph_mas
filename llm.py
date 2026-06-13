"""
llm.py — 커맨더(Claude Opus 4.8)·SLM(Ollama) 실제 호출 래퍼.
노드들은 이 두 함수만 부른다 (SDK를 직접 만지지 않음).

준비물 (내 PC):
  · pip install anthropic ollama python-dotenv
  · .env 파일에 ANTHROPIC_API_KEY 작성
  · Ollama 실행 + `ollama pull qwen3:8b`

반환 형식(둘 다 동일): (응답텍스트, prompt_tokens, completion_tokens)
※ Opus 4.8은 temperature 파라미터를 지원하지 않음 → 커맨더 호출에서 제외.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as c

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

_anthropic_client = None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic()
    return _anthropic_client


def call_commander(system, user_msg, max_tokens=1024):
    """커맨더 LLM(Claude Opus 4.8) 호출. 반환: (텍스트, 입력토큰, 출력토큰).
    ※ Opus 4.8은 temperature 미지원 → 넘기지 않는다."""
    client = _get_anthropic()
    resp = client.messages.create(
        model=c.COMMANDER_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return resp.content[0].text, resp.usage.input_tokens, resp.usage.output_tokens


def call_slm(system, user_msg):
    """SLM(Ollama, qwen3:8b) 호출. 반환: (텍스트, 입력토큰, 출력토큰)."""
    import ollama
    resp = ollama.chat(
        model=c.SLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user_msg}],
        options=c.SLM_OPTIONS,
    )
    return (resp["message"]["content"],
            resp.get("prompt_eval_count", 0),
            resp.get("eval_count", 0))


if __name__ == "__main__":
    print("커맨더 테스트...")
    t, pi, po = call_commander("너는 커맨더다.", "한 문장으로 자기소개 해줘.")
    print("  →", t, "| 토큰:", pi, po)
    print("SLM 테스트...")
    t, pi, po = call_slm("너는 분석 에이전트다.", "한 문장으로 자기소개 해줘.")
    print("  →", t, "| 토큰:", pi, po)
    print("\n🎉 둘 다 응답하면 실제 연결 성공!")
