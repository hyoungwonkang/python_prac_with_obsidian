# llm-ch6-classify

[[llm-from-scratch]] 마스터 플랜의 **Phase 6 / 6장** 정본 노트.
교재: Sebastian Raschka, *밑바닥부터 만들면서 배우는 LLM* — 6장. 분류를 위해 미세 튜닝하기.

## 개요

사전훈련된 모델에 분류 head를 붙이고 일부 레이어를 freeze해 분류 태스크(spam)로 미세 튜닝하는 단계.

## 체크리스트

- [x] 6.1 분류 데이터셋 준비 (`dataset_finetuning.py` + `data_loader.py`)
- [x] 6.2 모델에 분류 head 추가 (`heads.py`, out_head 768→2)
- [x] 6.3 일부 레이어 freeze 전략 (마지막 블록+final_norm+head만 학습)
- [x] 6.4 fine-tuning 학습 루프 + 평가 (`model_finetune.py`, **로컬 1.88분, val 97.5%**)
- [ ] 6.8 학습된 모델로 실제 스팸 분류 추론

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

### GPT-2 분류 파인튜닝 (6.4~6.7) — `heads.py`·`model_finetune.py`
- **out_head 교체**: 768→50257(생성) → 768→**2**(정상/스팸). 새 `nn.Linear`는 requires_grad=True 기본 → 자동 학습 대상.
- **마지막 토큰으로 분류**: `model(x)[:, -1, :]` → `(batch, 2)`. GPT는 왼→오라 **마지막 토큰이 문장 전체를 봄**(BERT의 `[CLS]` 대신). 출력 `(b, seq, 2)` 중 마지막만 사용.
- **freeze 전략**: 전체 동결 → **마지막 트랜스포머 블록 + final_norm + 새 head만 해제**. 백본 보존, 끝부분만 미세조정.
- **순서 함정**: 가중치 로드 → 동결 → out_head 교체. (교체를 먼저 하면 `load_weights_into_gpt`가 out_head=wte(50257) 대입에서 **shape mismatch** 크래시.)

### 정확도 vs 크로스 엔트로피 (대리 손실)
- 진짜 목표=정확도지만 **미분 불가**(argmax=계단함수, 기울기 0/끊김) → 경사하강법 못 씀.
- **크로스 엔트로피**(매끄럽고 미분 가능)를 **대리(surrogate)** 로 최소화 → 정답 확률↑ → 정확도 간접 최대화.
- 학습 전 baseline 손실 ≈ **ln(2)=0.693** = "50:50 찍기"(정확도 ~50%). **낮은 손실 ≠ 좋음**(확신 없이 어정쩡).
- 사전훈련 백본은 초기 손실이 오히려 **높음**(강한 활성화 → 랜덤 head가 자신있게 틀림) → 그러나 학습 후 훨씬 우수.

### 프롬프팅 baseline (파인튜닝 전, 6.x)
- 파인튜닝 전 GPT-2에게 "yes/no로 답해" 명령 프롬프트로 분류 시도 → **작은 모델이라 부정확** → 파인튜닝 필요성을 보여주는 도입.

### no_grad vs requires_grad / autograd 디폴트
- `requires_grad=False`: **특정 파라미터 영구 동결**(freeze).
- `torch.no_grad()`: **코드 블록 임시로** 기울기 추적 끔(평가·추론 → 메모리·속도↑).
- autograd 디폴트: 전역 grad 모드 ON(`torch.is_grad_enabled()`) + 모델 파라미터 `requires_grad=True` 기본. (내가 만든 `torch.tensor`는 False)

### ★ 로컬 실행 돌파 — HF 가중치 어댑터 (`hf_weight_adapter.py`)
- GPT-2 가중치 로드가 그동안 **Colab 전용**이었던 유일 이유 = `gpt_download.py`의 **TF import**(로컬 mutex 크래시).
- 해결: **HF `GPT2LMHeadModel.from_pretrained("gpt2")`(PyTorch)** 로 받아 → 같은 `params` 형식으로 변환 → **기존 `load_weights_into_gpt` 재사용**. TF 불필요. (HF Conv1D 가중치 방향이 TF 형식과 동일해서 가능)
- 결과: **로컬 M4 Max에서 전체 파인튜닝 완주 — 1.88분, val/test acc ~97%** (교재 M3 Air 6분의 ~3배 빠름). → GPT-2 트랙도 BERT처럼 **완전 로컬화**.

## 별도 트랙 접점 — BERT 실습은 업무 R&D로 선행/병행됨

원래 계획은 이 장 완료 직후 BERT 실습 진입이었으나, 업무 요청으로 **BERT 분류 R&D를 먼저 수행**(HF `BertForSequenceClassification`, SMS 스팸, test acc ~97%). → [[../../30-References/rnd-bert-labeling-test-plan]]. GPT-2 head 교체 학습과 비교 대상이 이미 확보됨.

## 검증

미세튜닝 전/후 지표 비교 — 분류 정확도 상승.

## 관련 노트

- [[llm-from-scratch]] — 마스터 플랜 (인덱스)
- [[../../30-References/rnd-bert-labeling-test-plan]] — BERT 분류 R&D(업무 산출물, HF). 같은 스팸분류를 BERT로 실증.
- [[../../30-References/bert_ocr_practice_plan]] — BERT·OCR 라이브러리 실습 트랙
- [[../../30-References/pytorch-env-hybrid]] — 환경 정본
