# llm-ch6-classify

[[llm-from-scratch]] 마스터 플랜의 **Phase 6 / 6장** 정본 노트.
교재: Sebastian Raschka, *밑바닥부터 만들면서 배우는 LLM* — 6장. 분류를 위해 미세 튜닝하기.

## 개요

사전훈련된 모델에 분류 head를 붙이고 일부 레이어를 freeze해 분류 태스크(spam)로 미세 튜닝하는 단계.

## 체크리스트

- [x] 6.1 분류 데이터셋 준비 (`dataset_finetuning.py` + `data_loader.py`)
- [ ] 6.2 모델에 분류 head 추가 (GPT-2 out_head 768→2 교체)
- [ ] 6.3 일부 레이어 freeze 전략
- [ ] 6.4 fine-tuning 학습 루프 + 평가 지표 (accuracy)
- [ ] 6.5 결과 정리

## 학습 메모 (실습으로 익힌 것)

### 데이터 준비 2단계 (모델 훈련 아님 — 손질일 뿐)
1. **`dataset_finetuning.py`** = 데이터셋 만들기. **데이터 다운로드**(SMS zip, UCI) → 라벨 인코딩 `.map({"ham":0,"spam":1})` → **클래스 균형화**(다수 ham을 spam 수로 언더샘플) → **70:10:20 분할** → CSV 저장. 모델·학습 루프 없음.
2. **`data_loader.py`** = CSV를 모델 입력으로. `SpamDataset`(토크나이즈+패딩) → `DataLoader`(배치). 여전히 훈련 아님.
3. (예정) **훈련 스크립트** = 모델 로드 + head 교체 + 학습 루프 → 여기서만 `loss.backward()`.

> "finetuning"이 이름에 있어도 dataset_finetuning은 **파인튜닝용 데이터 손질**이지 훈련이 아님.

### 슬라이딩 윈도(2장) vs 패딩(6장) — 길이 통일 방법의 차이
- 2장: 긴 글 하나를 `max_length`로 **잘라** 청크 생성 → 자를 때부터 균일 → 배치 공짜.
- 6장: SMS는 **원래 따로 있던 개별 문자**(샘플), 길이 제각각 → 자르면 의미 깨짐 → **패딩**(짧은 뒤를 `<|endoftext|>`=50256으로 채움)으로 길이 통일.

### 라벨링 ↔ 분류 = 동전의 양면
- 라벨링 = 정답(ham/spam) 달기 / 분류 = 그 정답 맞히기. 같은 ham/spam을 앞뒤로 다룸.
- 분류는 **지도학습**이라 라벨 없으면 학습 불가 → 라벨링이 전제조건.

### `SpamDataset` 핵심 (`data_loader.py`)
- 토크나이저: **tiktoken `gpt2`**, pad_token_id **50256**(`<|endoftext|>`).
- `max_length`: train의 **가장 긴 문장 길이**(예 120)로 잡고, val/test도 그 값에 맞춤.
- `_longest_encoded_length`: 지역변수 `max_length`로 통일해야 함 (`self.`와 섞으면 *마지막 문장 길이*를 반환하는 버그 — 실제 겪음).
- `__getitem__` → `(토큰텐서 (max_length,), 라벨 스칼라 텐서)`. **클래스 레이블 배열**은 CSV `Label`열 → `__getitem__`이 텐서화 → DataLoader가 batch_size개 stack → shape `(batch,)`.
- DataLoader 출력: 입력 `(batch, max_length)` / 레이블 `(batch,)`, 같은 순서로 짝 유지(shuffle해도).

### test셋·validation_frac
- train(학습) / val(중간 점검·과적합 확인) / **test(최종 1회 채점, 학습 미사용)**.
- `random_split(df, 0.7, 0.1)`: train_frac 0.7, **validation_frac 0.1**, test는 나머지(0.2, 자동).

### 두 트랙 구분 (헷갈림 주의)
| | 교재 GPT-2 트랙 (이 장) | 업무 BERT R&D 트랙 |
|---|---|---|
| 모델 | GPT-2 (직접 구축 + **OpenAI** 가중치) | BERT (`bert-base-uncased`, **구글**) |
| 토크나이저 | tiktoken gpt2 (pad 50256) | BertTokenizer ([CLS]/[SEP]) |
| 분류 | 마지막 토큰 출력 → head 50257→2 | `[CLS]` 벡터 → 분류층 |
| 데이터 준비 | `dataset_finetuning.py` 공유 | 〃 (같은 코드 재사용) |

## 별도 트랙 접점 — BERT 실습은 업무 R&D로 선행/병행됨

원래 계획은 이 장 완료 직후 BERT 실습 진입이었으나, 업무 요청으로 **BERT 분류 R&D를 먼저 수행**(HF `BertForSequenceClassification`, SMS 스팸, test acc ~97%). → [[../../30-References/rnd-bert-labeling-test-plan]]. GPT-2 head 교체 학습과 비교 대상이 이미 확보됨.

## 검증

미세튜닝 전/후 지표 비교 — 분류 정확도 상승.

## 관련 노트

- [[llm-from-scratch]] — 마스터 플랜 (인덱스)
- [[../../30-References/rnd-bert-labeling-test-plan]] — BERT 분류 R&D(업무 산출물, HF). 같은 스팸분류를 BERT로 실증.
- [[../../30-References/bert_ocr_practice_plan]] — BERT·OCR 라이브러리 실습 트랙
- [[../../30-References/pytorch-env-hybrid]] — 환경 정본
