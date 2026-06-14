"""LLM 워커 (HARNESS 골격 4: 다중 워커 조율).

- analyze_intent: 경량 워커 Claude Haiku 4.5 — 빠르고 싸고 결정적(temperature=0.1).
- generate_plan: 메인 워커 Claude Opus 4.8 — adaptive thinking + effort.
  (Opus 4.8은 temperature/top_p/budget_tokens 제거됨 → 보내면 400)

각 워커는 실패 시 결정적 폴백으로 강등 (골격 6: graceful degradation).
"""
from __future__ import annotations

import json
import os

from .schema import Intent

_HAIKU = "claude-haiku-4-5"   # 경량 워커
_OPUS = "claude-opus-4-8"     # 메인 워커


def _client():
    """anthropic 클라이언트 (지연 임포트 — 검증 루프 테스트는 anthropic 불필요)."""
    import anthropic
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ---- Step 1: 의도 분석 (경량 워커) ----

_INTENT_SYS = (
    "너는 나들이 추천 요청을 분석하는 분류기다. "
    "사용자 요청에서 다음을 추출해 JSON만 출력하라(설명 금지):\n"
    '{"keywords": [...], "include_categories": [...], "exclude_categories": [...], '
    '"indoor_only": true|false|null, "style_hint": "..."}\n'
    "카테고리 예: 문화/야외/테마파크/체험/관광/실내놀이. "
    "실내 선호가 명확하면 indoor_only=true, 야외 선호면 false, 무관하면 null."
)


def analyze_intent(query: str) -> Intent:
    """Haiku로 의도 분석. 실패 시 키워드 휴리스틱으로 강등."""
    try:
        resp = _client().messages.create(
            model=_HAIKU,
            max_tokens=384,
            temperature=0.1,  # Haiku 4.5는 sampling 파라미터 허용
            system=_INTENT_SYS,
            messages=[{"role": "user", "content": query}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        data = _safe_json(text)
        return Intent(**{k: data[k] for k in Intent.model_fields if k in data})
    except Exception:
        return _fallback_intent(query)


def _fallback_intent(query: str) -> Intent:
    """LLM 실패 시 정규식/키워드 기반 휴리스틱 (골격 6: 3단 강등의 마지막)."""
    indoor = None
    if "실내" in query:
        indoor = True
    elif "야외" in query or "바깥" in query:
        indoor = False
    return Intent(keywords=[w for w in query.split() if len(w) > 1][:5],
                  indoor_only=indoor, style_hint=query[:80])


# ---- Step 3: 플랜 생성 (메인 워커) ----

def build_plan_prompt(query: str, intent: Intent, candidates: list[dict],
                      days: int, adults: int, children: int) -> str:
    """후보 카탈로그 + 요청을 플랜 생성 프롬프트로 조립."""
    lines = [f'- itemId={c["id"]} | {c["name"]} | {c["category"]} | '
             f'{"실내" if c["indoor"] else "야외"} | 성인 {c["adultPrice"]}원'
             for c in candidates]
    return (
        f"요청: {query}\n"
        f"인원: 성인 {adults}, 아동 {children} / 일수: {days}일\n"
        f"스타일 힌트: {intent.style_hint}\n\n"
        f"아래 후보에서만 골라 {days}일치 일정을 짜라. "
        f"반드시 목록의 itemId만 사용하고, 하루 2~3개를 추천하라.\n"
        f"후보:\n" + "\n".join(lines) + "\n\n"
        '출력은 JSON만:\n'
        '{"title": "...", "itinerary": [{"day": 1, "activities": [{"itemId": 101}]}]}'
    )


def generate_plan(user_message: str, attempt: int = 0) -> str:
    """Opus로 플랜 생성. 응답 텍스트(JSON 추정)를 그대로 반환.

    검증 루프(plan.py)가 파싱·검증·재시도를 담당하므로 여기선 생성만.
    """
    resp = _client().messages.create(
        model=_OPUS,
        max_tokens=2048,
        thinking={"type": "adaptive"},          # Opus 4.8: adaptive only
        output_config={"effort": "medium"},     # 재시도 시 호출부에서 조정 가능
        messages=[{"role": "user", "content": user_message}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "")


def _safe_json(text: str) -> dict:
    try:
        from .validator import parse_llm_response
        out = parse_llm_response(text)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}
