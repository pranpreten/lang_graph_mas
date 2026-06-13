"""
_parse.py — SLM 응답에서 구조화된 JSON 결정을 뽑아내는 공용 도우미.

SLM은 깔끔한 JSON을 늘 주지 않으므로(설명을 덧붙이거나 형식이 어긋남),
느슨하게 파싱하고 실패하면 기본값(default)을 돌려준다.
→ 파이프라인이 안 죽게 해서 '완주율'을 지킨다.
"""
import json
import re


def extract_json(text, default):
    """text 안에서 첫 JSON 객체를 찾아 dict로 반환. 실패하면 default 복사본 반환."""
    # 1) 통째로 JSON 인지
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 2) 텍스트 속 { ... } 덩어리 추출
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # 3) 다 실패 → 기본값 (파이프라인 보호)
    return dict(default)


if __name__ == "__main__":
    # 간단 자가 테스트 (모델 없이 동작)
    cases = [
        '{"scaler": "standard", "features": "drop_low_variance"}',     # 깔끔
        '좋아요! 결정은 이렇습니다: {"scaler":"minmax"} 입니다.',        # 설명 섞임
        '도저히 모르겠음',                                              # 망함 → 기본값
    ]
    default = {"features": "drop_low_variance", "scaler": "standard"}
    for t in cases:
        print(repr(t[:30]), "→", extract_json(t, default))
