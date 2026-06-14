"""파이프라인 통합 테스트 (HARNESS 골격 1·5·6). 의존성 0 — LLM은 stub 주입.

실 anthropic/fastapi 없이 Step 0~4 전체 흐름을 검증한다:
guardrail → intent → filter → verify→fix 루프 → 폴백.
"""
import json
from types import SimpleNamespace

from core import catalog, pipeline
from core.guardrail import GuardrailBlockedError

# 공통 stub: 의도(전체 통과), 프롬프트(무의미 문자열)
_INTENT = SimpleNamespace(include_categories=[], exclude_categories=[],
                          indoor_only=None, style_hint="")
def _intent(_q): return _INTENT
def _prompt(*a, **k): return "PROMPT"


def _two_real_ids():
    ids = list(catalog.valid_item_ids())
    return ids[0], ids[1]


def _plan_json(days_items: list[list[int]]) -> str:
    """[[id,id],[id]] → 일자별 플랜 JSON 문자열."""
    itin = [{"day": i + 1, "activities": [{"itemId": x} for x in day]}
            for i, day in enumerate(days_items)]
    return "```json\n" + json.dumps({"title": "t", "itinerary": itin}) + "\n```"


def test_happy_path():
    a, b = _two_real_ids()
    gen = lambda msg, attempt=0: _plan_json([[a, b], [a]])
    out = pipeline.run_pipeline("실내 위주", days=2, adults=1, children=0,
                                analyze_intent=_intent, build_prompt=_prompt, generate_plan=gen)
    assert out["error"] is None
    assert len(out["plan"]["itinerary"]) == 2


def test_verify_fix_loop_corrects_violation():
    """1차: 1일치(2일 요청 위반) → 2차: 2일치(통과). 루프가 교정해야 한다."""
    a, b = _two_real_ids()
    calls = {"n": 0}
    def gen(msg, attempt=0):
        calls["n"] += 1
        return _plan_json([[a]]) if calls["n"] == 1 else _plan_json([[a], [b]])
    out = pipeline.run_pipeline("x", days=2, adults=1, children=0,
                                analyze_intent=_intent, build_prompt=_prompt, generate_plan=gen)
    assert calls["n"] == 2, "위반 시 재호출되어야 한다"
    assert out["error"] is None
    assert len(out["plan"]["itinerary"]) == 2


def test_broken_json_then_recovers():
    a, b = _two_real_ids()
    calls = {"n": 0}
    def gen(msg, attempt=0):
        calls["n"] += 1
        return "JSON 아님 죄송" if calls["n"] == 1 else _plan_json([[a], [b]])
    out = pipeline.run_pipeline("x", days=2, adults=1, children=0,
                                analyze_intent=_intent, build_prompt=_prompt, generate_plan=gen)
    assert out["error"] is None
    assert len(out["plan"]["itinerary"]) == 2


def test_all_broken_falls_back():
    """3회 모두 깨진 JSON → 결정적 폴백 (error 채워짐, 일수는 맞음)."""
    gen = lambda msg, attempt=0: "전부 깨짐"
    out = pipeline.run_pipeline("x", days=2, adults=1, children=0,
                                analyze_intent=_intent, build_prompt=_prompt, generate_plan=gen)
    assert out["error"] is not None, "폴백 시 안내 메시지가 있어야 한다"
    assert len(out["plan"]["itinerary"]) == 2


def test_hallucinated_id_removed_integration():
    a, _ = _two_real_ids()
    gen = lambda msg, attempt=0: _plan_json([[a, 999999], [a]])  # 999999 환각
    out = pipeline.run_pipeline("x", days=2, adults=1, children=0,
                                analyze_intent=_intent, build_prompt=_prompt, generate_plan=gen)
    flat = [act["itemId"] for d in out["plan"]["itinerary"] for act in d["activities"]]
    assert 999999 not in flat, "환각 itemId는 파이프라인을 통과하면 안 된다"


def test_guardrail_blocks_at_step0():
    gen = lambda msg, attempt=0: _plan_json([[1]])
    try:
        pipeline.run_pipeline("자살 방법", days=1, adults=1, children=0,
                              analyze_intent=_intent, build_prompt=_prompt, generate_plan=gen)
    except GuardrailBlockedError:
        return
    assert False, "검열 위반은 Step 0에서 차단되어야 한다"
