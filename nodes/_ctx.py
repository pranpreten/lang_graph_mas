"""_ctx.py — 커맨더 지침/피드백(commander_input)을 SLM 프롬프트에 주입하는 헬퍼."""

def commander_prefix(state):
    """커맨더가 남긴 지침/피드백이 있으면 프롬프트 앞에 붙일 문자열 반환."""
    ci = state.get("commander_input", "")
    return f"[커맨더 지침] {ci}\n" if ci else ""
