# llm-ch7-instruct

[[llm-from-scratch]] 마스터 플랜의 **Phase 7 / 7장** 정본 노트.
교재: Sebastian Raschka, *밑바닥부터 만들면서 배우는 LLM* — 7장. 지시를 따르도록 미세 튜닝하기.

## 개요

instruction 데이터셋과 프롬프트 템플릿으로 모델이 지시를 따르도록 미세 튜닝하고, 정성·자동 평가로 마무리하는 마지막 장.

## 체크리스트

- [x] 7.1 instruction 데이터셋 구성 — 다운로드+분할 (`instruction_dataset_finetune.py`, rasbt 1100건)
- [~] 7.2 입력 포맷팅(프롬프트 템플릿) `format_input` 적용 (Alpaca식)
- [ ] 7.3 `InstructionDataset`(토큰화) + **사용자 정의 콜레이트 함수**(배치 시 패딩·타깃·-100)
- [ ] 7.4 instruction fine-tuning 학습
- [ ] 7.5 정성 평가 + 간단한 자동 평가 루프
- [ ] 7.6 회고 — 사전훈련 vs 분류 FT vs 지시 FT의 차이 노트화

## 학습 메모 (실습으로 익힌 것)

### 데이터 준비 (`instruction_dataset_finetune.py`)
- 다운로드: rasbt 레포 `instruction-data.json`(1100건). `import urllib.request` 필요 — `import urllib`만 하면 `urllib.request` 미로드(AttributeError). 실제 겪음.
- `format_input`: Alpaca식 프롬프트 템플릿("Below is an instruction..." + `### Instruction` + 선택적 `### Input`). 응답은 `### Response`로 별도 결합.
- 분할: 단순 **순서 슬라이싱** train 85%/test 10%/val 나머지. ch6은 `random_split`(섞음) — ch7은 순차 분할(데이터가 이미 다양). val은 `[...:]`로 나머지 전부(int 버림 누락 방지).

### 콜레이트 함수 = 배치 시점 처리 (단계 2.x) — ch6 DataLoader로 부족
- 데이터 준비(다운로드·format·분할)는 **콜레이트 전 단계**. `InstructionDataset`(토큰화)이 단계 1, **콜레이트 함수**가 단계 2.x.
- ch6은 입력만 패딩하면 끝(라벨=숫자). ch7은 입력·타깃 둘 다 자유 텍스트라 콜레이트에서 ①배치 내 최장 길이 패딩 ②**타깃=입력 한 토큰 이동** ③패딩 자리 **-100**(손실 제외)까지 직접 처리 → 그래서 사용자 정의 콜레이트.

### 타깃 = 입력 한 토큰 이동 = 다음 토큰 예측 (사전훈련과 동일)
- `입력[A B C D]` → `타깃[B C D E]`(한 칸 밀림). 각 위치에서 "바로 다음 토큰" 맞히기 = ch2 슬라이딩 윈도·ch5 사전훈련과 **같은 메커니즘**.
- "한 토큰 이동"(학습 방식) ≠ "GPT가 토큰 하나씩 다룸"(causal/자기회귀 구조)이지만 **원인-결과**: 이렇게 학습해서 → 자기회귀 생성이 가능.

### 분류 FT(6장) vs 명령어 FT(7장)
| | 분류(ch6) | 명령어(ch7) |
|---|---|---|
| 출력 | 고정 클래스(0/1) | 자유 텍스트 생성 |
| head | 768→2 **교체** | 768→50257 **유지** |
| 쓰는 출력 | 마지막 토큰 argmax | 토큰 하나씩(자기회귀) |
| 손실 | 마지막 토큰 CE | 모든 위치 다음 토큰 CE |
| 비유 | 객관식 | 서술형 |

### 배치 처리가 빠른 이유 (GPU 병렬)
- 여러 샘플을 한 텐서로 묶어 넣으면 GPU **코어 수천 개**가 동시 계산 → 1개 처리 시간 ≈ 배치 처리 시간(노는 코어를 채움). 버스 빈 좌석 채우기 비유.
- **코어 ≠ 메모리**(독립 자원, 비례 안 함): 코어=계산 일꾼, 메모리=데이터 창고. 배치 키우면 노는 코어 채워 빨라지지만 **메모리 초과(OOM)** 면 멈춤 → 둘의 균형점에서 batch_size 결정.

## 검증

미세튜닝 전/후 지표 비교 — instruction 응답 품질 개선.

## 관련 노트

- [[llm-from-scratch]] — 마스터 플랜 (인덱스)
- [[llm-ch6-classify]] — 6장 분류 FT (출력·head·손실 대비)
- [[../../30-References/bert-vs-gpt2-classification]] — 트랜스포머·어텐션 개념 정리
- [[../../30-References/pytorch-env-hybrid]] — 환경 정본
