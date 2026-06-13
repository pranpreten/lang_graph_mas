"""
verify_api.py — 커맨더 API(Claude) 연결 확인용. ★내 PC에서 실행★

준비:
  1) pip install anthropic python-dotenv
  2) .env 파일에 ANTHROPIC_API_KEY 작성
실행:
  python verify_api.py
"""
import os

# .env 로드
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    print("⚠ python-dotenv 없음 →  pip install python-dotenv  (또는 시스템 환경변수 사용)")

MODEL = "claude-opus-4-8"

key = os.environ.get("ANTHROPIC_API_KEY")
if not key or key.startswith("sk-ant-여기에"):
    print("❌ ANTHROPIC_API_KEY가 없거나 placeholder입니다. .env에 실제 키를 넣으세요.")
    raise SystemExit(1)
print("✓ API 키 감지됨 (앞 12자):", key[:12], "...")

try:
    from anthropic import Anthropic
except ImportError:
    print("❌ anthropic 패키지 없음 →  pip install anthropic")
    raise SystemExit(1)

client = Anthropic()
resp = client.messages.create(
    model=MODEL, max_tokens=50,
    messages=[{"role": "user", "content": "한 문장으로 자기소개 해줘."}],
)
print("✓ 모델 응답:", resp.content[0].text)
print("✓ 토큰 — 입력:", resp.usage.input_tokens, "출력:", resp.usage.output_tokens)
print("\n🎉 커맨더 API 연결 성공!")
