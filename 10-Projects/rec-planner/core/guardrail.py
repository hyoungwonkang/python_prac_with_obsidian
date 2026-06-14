"""입력 검열 워커 (HARNESS 골격 4: 전담 워커 분리 / 골격 6: API 경계 매핑).

travel-rag의 `GuardrailBlockedError`(Bedrock guardrail) 경량 대체.
- 여기서는 **결정적 denylist**로 구현 → anthropic 없이도 항상 작동·테스트 가능.
- 실서비스라면 모델 기반 guardrail(Haiku 분류기/Bedrock guardrail)로 교체.
검열 단계를 파이프라인 본체와 분리한 것 자체가 체크리스트의 요점.
"""
from __future__ import annotations


class GuardrailBlockedError(Exception):
    """입력이 정책상 차단됨. API 경계에서 400으로 매핑된다."""

    def __init__(self, category: str, message: str = "요청을 처리할 수 없습니다."):
        self.category = category
        super().__init__(message)


# 데모용 결정적 denylist (실서비스는 모델 기반 guardrail 권장)
_BLOCKED: dict[str, tuple[str, ...]] = {
    "self_harm": ("자살", "자해"),
    "weapon": ("폭탄", "총기 제조", "사제 폭탄"),
    "illegal": ("마약 제조", "해킹 방법"),
}


def screen_input(query: str) -> None:
    """요청을 검열. 위반 시 GuardrailBlockedError, 통과 시 None.

    나들이 추천과 무관한 위험 요청을 입력 경계에서 차단한다 (Step 0).
    """
    text = (query or "").lower()
    for category, patterns in _BLOCKED.items():
        for p in patterns:
            if p in text:
                raise GuardrailBlockedError(category)
