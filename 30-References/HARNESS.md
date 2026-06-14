# LLM Harness Engineering — 범용 설계 가이드

> **하네스(Harness)란?** LLM처럼 *불안정하고 비결정적인 엔진*을, 신뢰할 수 있는 시스템으로 감싸는 **절차적·구조적 골격**.
> 사용자는 "추천 한 줄"만 보지만, 그 뒤에서 명령어 파이프라인 · 핸드오프 · 검증 루프 · 장애 폴백이 돌아간다.
> 모델 품질(어떤 모델을 쓰는가)이 *프롬프트 엔지니어링*이라면, 그 모델을 빈틈없이 부려쓰는 *방법(How)* 이 **하네스 엔지니어링**이다.

이 문서는 두 부분으로 구성된다:

1. **범용 템플릿** — 모든 LLM 애플리케이션에 적용되는 6대 구성요소 + 체크리스트. 새 프로젝트에 그대로 복사해 쓴다.
2. **레퍼런스 예시** — 이 저장소(`travel-rag`)가 각 구성요소를 어떻게 구현했는지 `파일:줄` 근거와 함께 제시. 새 프로젝트에서는 이 블록만 교체한다.

> 📌 **새 프로젝트에 재사용하는 법**: 이 파일을 새 서비스 디렉토리에 복사 → 각 섹션의 `### 📍 레퍼런스 예시 (travel-rag)` 블록을 비우거나 자기 코드로 교체 → 상단 정의와 체크리스트는 그대로 유지.

---

## 0. 왜 필요한가 — 엔진은 거짓말을 한다

LLM은 다음을 *언제든* 한다:

- 존재하지 않는 ID·필드를 **환각**한다
- 산술을 **틀린다** (합계, 박수 계산)
- 요청한 JSON 포맷을 **깬다** (마크다운 펜스, 설명 문구 첨가)
- 같은 입력에 **다른 출력**을 낸다 (비결정성)
- 외부 호출(모델 API·DB·큐)이 **죽는다**

하네스의 전제는 단 하나: **"엔진의 출력을 절대 신뢰하지 않는다. 검증층이 보정한다."**

---

## 1. 제어 흐름 — 명시적 단계 파이프라인

### 원칙
- 작업을 `입력 → 단계1 → 단계2 → … → 출력`의 **명시적 단계**로 쪼갠다.
- 각 단계는 **앞 단계의 산출물만** 입력으로 받는다 (숨은 전역 상태 금지).
- 단계 경계마다 로깅 → 어디서 실패했는지 즉시 추적 가능.

### ✅ 체크리스트
- [ ] 진입점 함수에 `Step 1 / Step 2 …` 주석 또는 단계 구분이 있는가?
- [ ] 각 단계 산출물이 다음 단계로 명시적으로 전달되는가?
- [ ] 단계별 진입/완료 로그가 있는가?

### 📍 레퍼런스 예시 (travel-rag)
- **최상위 파이프라인** — `app/routers/chat.py:28` `recommend_chat()`
  `Step 1` 자연어 파싱 → `Step 2` `TravelQuery` 구조화 → `Step 3` `recommend()` 위임
- **추천 하위 파이프라인** — `app/routers/recommend.py:72` `recommend()`
  `Step 1` 의도분석(`:81`) → `Step 2` 검색텍스트 구성(`:91`) → `Step 3` 임베딩(`:98`) → `Step 4` ES 하이브리드 검색(`:101`) → `Step 5` 실시간 SKU 조회(`:129`) → `Step 6` LLM 플랜 생성+검증(`:151`)

---

## 2. 컨텍스트 관리 — 단계 간 핸드오프 계약

### 원칙
- 단계 사이에 비정형 데이터를 그냥 흘리지 말고, **구조화된 계약(타입/스키마)** 으로 넘긴다.
- 핸드오프 객체에는 "결정"과 "제약(위험)"을 함께 담는다.

### ✅ 체크리스트
- [ ] 단계 간 전달 데이터가 dataclass / Pydantic / TypedDict 등 **타입**으로 정의돼 있는가?
- [ ] 여러 출처의 신호를 합칠 때 **병합 규칙**이 한 곳에 모여 있는가?

### 📍 레퍼런스 예시 (travel-rag)
- **핸드오프 계약** — `recommend/intent_analyzer.py:114` `@dataclass SearchIntent`
  Haiku가 분석한 의도(`search_keywords`, `category_boosts`, `style_hint`, `excluded_*`)를 구조화. `style_hint`는 플랜 생성 프롬프트로, `excluded_*`는 검색 필터로 흘러간다.
- **병합 규칙 단일화** — `app/routers/recommend.py:59` `_merge_exclusions()`
  API 입력 필드와 LLM이 추론한 제약을 한 곳에서 병합 (카테고리/장르/상품ID/SKU).

---

## 3. 상태 영속화 — 크래시 후 재개(resume)

### 원칙
- 진행 상태를 **외부에 영속화**(DB/큐 오프셋/Redis)하여, 프로세스가 죽어도 이어서 처리한다.
- 개별 작업 실패를 **격리**해 전체가 멈추지 않게 한다.

### ✅ 체크리스트
- [ ] 장기 실행 워커가 죽으면 마지막 처리 지점부터 재개되는가? (오프셋 커밋 등)
- [ ] 개별 항목 실패가 루프 전체를 죽이지 않고 격리되는가?
- [ ] 외부 연결 장애 시 재연결 루프가 있는가?

### 📍 레퍼런스 예시 (travel-rag)
- **재개 + 실패 격리 루프** — `services/kafka_consumer.py:64`
  ```python
  while True:
      try:
          consumer = _KafkaConsumer(..., enable_auto_commit=True, auto_offset_reset="latest")
          for msg in consumer:
              try: reindex_product(...)        # 개별 메시지 실패 격리
              except Exception as e: logger.warning(...)
      except Exception as e:                    # 연결 장애 → 재개
          logger.warning("... retrying in 10s"); time.sleep(10)
  ```
  오프셋 커밋 = 진행 상태 영속화, 바깥 `while/except` = 재연결 resume, 안쪽 `try/except` = 단건 격리.
- (BE 전반) CQRS: command side가 DB 소유 → Kafka 이벤트 발행 → query side가 Redis 프로젝션으로 복원. 자세한 내용은 `../../CLAUDE.md` 참고.

---

## 4. 다중 "워커" 조율 — 역할별 엔진 배정

### 원칙
- 작업 성격에 맞는 엔진(모델/서비스)에 **역할을 분담**한다: 가볍고 빠른 일 ↔ 무겁고 정교한 일.
- 입력 검열·후처리 등 **전담 워커**를 분리한다.

### ✅ 체크리스트
- [ ] 싸고 빠른 작업과 비싸고 정교한 작업에 **다른 모델/설정**을 쓰는가?
- [ ] 결정성이 필요한 단계는 `temperature`를 낮췄는가?
- [ ] 입력 검열(guardrail) 워커가 분리돼 있는가?

### 📍 레퍼런스 예시 (travel-rag)
- **경량 워커(Haiku) 강제 배정** — `recommend/intent_analyzer.py:26` `_HAIKU_MODEL_ID`
  의도 분석은 `LLM_MODEL_ID`와 무관하게 항상 Haiku + `max_tokens=384, temperature=0.1` (빠르고 싸고 결정적).
- **무거운 워커(메인 모델)** — `app/routers/recommend.py:177` 플랜 생성 `invoke_llm(..., max_tokens=9999)`, 1차 `temp=0.7` → 재시도 시 `0.2`로 보수화.
- **검열 워커** — `common/bedrock.py:11` `GuardrailBlockedError`, `services/query_parser.py:94`에서 차단 시 발생.

---

## 5. 검증 루프 — verify → 자동 fix → 무한루프 방지

> **하네스의 심장.** LLM 출력을 받자마자 검증하고, 위반이 있으면 *교정 지시와 함께 재호출*하되, **횟수 상한**으로 무한루프를 막고, 끝내 실패하면 *결정적 폴백*으로 안전한 결과를 보장한다.

### 원칙
1. **거부** — 유효하지 않은 출력(환각 ID, 깨진 JSON)을 버린다.
2. **교정** — 신뢰 불가 값(합계 등)은 코드가 **항상 재계산**한다.
3. **재시도** — 위반 사항을 fix-up 메시지로 만들어 다시 호출한다.
4. **상한** — `max_attempts`로 루프를 끊는다 (`max_fix_loops`).
5. **폴백** — 최종 실패 시 결정적 보정으로 최소한의 유효 결과를 낸다.

### ✅ 체크리스트
- [ ] LLM 출력의 모든 참조(ID 등)가 실제 데이터에 존재하는지 검증하는가?
- [ ] 합계/계산값을 LLM 출력 그대로 쓰지 않고 **코드가 재계산**하는가?
- [ ] 재시도 횟수 상한(`max_attempts`)이 있는가?
- [ ] 상한 도달 시 결정적 폴백 경로가 있는가?
- [ ] 출력 개수 상한(cap)으로 폭주를 막는가?

### 📍 레퍼런스 예시 (travel-rag)
- **검증·교정 본체** — `recommend/validator.py`
  - `:73` 존재하지 않는 `productId/skuId` 참조 활동 **제거**
  - `:81` STAY 가격을 실제 SKU가 × 박수로 **교정**
  - `:115` `totalPrice` **항상 재계산** — *"LLM 산술 오류 방지"*
  - `:88` 같은 날짜 day_plan 병합, `:98` 중복 productId 제거
  - `:166` `MAX_PLANS_PER_TIER = 3` 초과분 절단, `:163` 활동 0개 플랜 폐기
- **verify→fix 루프 + 상한** — `app/routers/recommend.py:173`
  ```python
  max_attempts = 3                               # ← max_fix_loops
  for attempt in range(max_attempts):
      response_text = invoke_llm(...)            # temp: 1차 0.7 → 이후 0.2
      try: parsed = parse_llm_response(...)
      except: # JSON 깨짐 → fix-up 지시 붙여 재시도 (:184)
      validated = validate_tier_response(...)    # 거부+교정 (:198)
      violations = check_plan_constraints(...) + check_plans_diversity(...)  # (:209)
      if not violations: break                   # 통과 시 종료 (:221)
      fixup = build_fixup_message(...)           # 위반을 교정 지시로 (:237)
      user_message = base + fixup                # 다음 attempt에 주입
  # 상한 도달 시 최종 폴백: prune_flights + totalPrice 재계산 (:241)
  ```

---

## 6. 장애 처리 — graceful degradation 캐스케이드

### 원칙
- 실패를 **단계적으로 강등**한다: 정밀 → 근사 → 휴리스틱 → 명확한 에러.
- 사용자에게는 **행동 가능한 에러 메시지**를, 시스템에는 **분류된 예외**를 남긴다.

### ✅ 체크리스트
- [ ] 외부 호출 실패 시 폴백 경로가 있는가? (정밀→근사→최소)
- [ ] 비정형 출력 파싱이 여러 전략을 순차 시도하는가?
- [ ] API 경계에서 예외를 적절한 상태코드로 매핑하는가? (검열 400, 검증실패 422 등)

### 📍 레퍼런스 예시 (travel-rag)
- **의도 분석 3단 강등** — `recommend/intent_analyzer.py:225`
  Haiku 실패 → `_fallback_from_dict()` 사전 기반(`:71`) → `_fallback_negations()` 정규식 휴리스틱(`:99`).
- **JSON 파싱 3전략 캐스케이드** — `recommend/validator.py:5` `parse_llm_response()`
  ① 마크다운 코드펜스 → ② 직접 파싱 → ③ 괄호 중첩 카운팅 추출 → 최종 실패 시에만 `raise`.
- **API 경계 매핑** — `app/routers/chat.py:38`
  `GuardrailBlockedError → 400`, 파싱/검증 실패 → `422` + 사용자 안내 메시지.
- **LLM 호출 전면 실패** — `app/routers/recommend.py:250`
  guardrail → `400`, 그 외 → 티어별 빈 결과 + `error` 필드로 부분 응답.

---

## 부록 A — 새 프로젝트 부트스트랩 체크리스트

새 LLM 서비스를 만들 때, 위 6개를 이 순서로 갖춘다:

1. **[제어 흐름]** 진입점을 `Step N` 주석이 달린 명시적 파이프라인으로 작성.
2. **[핸드오프]** 단계 간 전달 데이터를 dataclass/Pydantic 타입으로 정의.
3. **[검증 루프]** *가장 먼저 만들 것* — LLM 출력 검증·교정 함수 + `max_attempts` 루프. 이게 없으면 데모는 되도 운영은 안 된다.
4. **[장애 처리]** 모든 LLM/외부 호출에 폴백 경로. 파싱은 다전략.
5. **[워커 분담]** 가벼운 분류/추출은 작은 모델, 무거운 생성은 큰 모델.
6. **[상태 영속화]** 비동기/장기 작업이 있으면 큐 오프셋·재연결 루프.

> 우선순위 팁: **3(검증 루프)과 4(장애 처리)가 하네스의 80%다.** 1·2는 가독성, 5·6은 규모 확장. 검증 없는 LLM 앱은 하네스가 아니라 그냥 데모다.

## 부록 B — 안티패턴 (하지 말 것)

- ❌ LLM이 준 합계·ID를 **그대로 신뢰**해서 응답에 넣기 → 반드시 재계산/대조.
- ❌ 재시도에 **상한 없음** → 비용 폭주·무한루프.
- ❌ `except: pass`로 에러 **조용히 삼키기** → 최소 `logger.warning`. (현 코드 audit: `recommend/constraints.py:89,101`이 약한 케이스)
- ❌ 단계 간 데이터를 **dict 그대로** 흘리기 → 타입 계약으로.
- ❌ 모든 작업에 **같은 큰 모델** → 분류/추출은 경량 모델로 비용↓·속도↑.

---

*이 문서는 `travel-rag`를 레퍼런스 구현으로 삼은 범용 하네스 가이드다. 새 서비스에 복사 후 `### 📍 레퍼런스 예시` 블록만 교체해 사용한다.*
