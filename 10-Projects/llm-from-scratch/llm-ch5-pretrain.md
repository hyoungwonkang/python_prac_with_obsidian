# llm-ch5-pretrain

[[llm-from-scratch]] 마스터 플랜의 **Phase 5 / 5장** 정본 노트.
교재: Sebastian Raschka, *밑바닥부터 만들면서 배우는 LLM* — 5장. 레이블이 없는 데이터를 활용한 사전 훈련.

## 개요

cross-entropy/perplexity 기반 학습 루프를 구현해 소규모 코퍼스로 사전훈련하고, OpenAI 공개 GPT-2 가중치를 자체 구조에 매핑한다. 실제 학습은 Colab T4가 메인.

## 체크리스트

- [~] 5.1 cross-entropy loss 계산 — `text_model.py`(수동 + `F.cross_entropy`). perplexity 미작성
- [x] 5.2 학습 루프 (forward → loss → backward → optimizer.step) — `train_gpt.py` 작성 완료(실행 보류)
- [ ] 5.3 작은 코퍼스로 1 epoch 학습 (로컬에서 검증)
- [ ] 5.4 학습률 워밍업·코사인 스케줄링
- [ ] 5.5 체크포인트 저장/로드 (`weights_only=True`)
- [ ] 5.6 OpenAI 공개 GPT-2 가중치 로드 → 자체 구조에 매핑
- [ ] 5.7 **Colab T4에서 소규모 사전훈련 실행** (로컬은 코드 디버그)
- [ ] (별도) top-k·temperature 샘플링 (4.5에서 이월)

## 학습 정리 (2026-06-24)

### 텍스트 생성 사이클
`텍스트 →[encode]→ 토큰ID →[model]→ 로짓 →[argmax]→ 토큰ID →[decode]→ 텍스트`.
- 로짓 = `out_head`(768→50257)가 낸 **어휘별 점수 벡터**(확률분포 직전; softmax 거치면 확률).
- argmax = **최댓값의 위치(=토큰ID)** 선택 = greedy. softmax는 단조(순서 보존)라 argmax 앞에선 생략 가능.
- 헬퍼: `text_to_token_ids`/`token_ids_to_text`(입출구), `generate_text_simple`(자기회귀 반복).

### 손실(loss) = 크로스 엔트로피
- loss = "예측이 정답에서 얼마나 틀렸나"를 나타내는 **한 숫자**(낮을수록 좋음). 학습 = loss 최소화.
- 정답이 원-핫이라 크로스 엔트로피 = $-\log(\text{모델이 정답 토큰에 매긴 확률})$ 의 평균. 정답 확신↑→loss↓, 자신만만한 오답→∞.
- `torch.nn.functional.cross_entropy(logits_flat, targets_flat)` — ⚠️ **logits를 직접** 받음(내부에서 softmax+log+음수+평균). `logits.flatten(0,1)`/`targets.flatten()`로 (배치·시퀀스) 펼쳐 입력.

### 훈련 손실 vs 검증 손실
- 같은 loss 도구를 **대상 데이터만 바꿔** 적용: train(학습·grad/step) / val(감시·grad 없음).
- train↓인데 val↑ = **과적합** 신호 → early stopping. The Verdict(작음)는 빨리 과적합, tinyshakespeare(큼)는 느림.
- 보통 `calc_loss_loader`로 로더 전체 배치 평균을 train/val 각각 계산.

### 데이터셋 2종 (로컬 확보)
| 파일 | 토큰 | 용도 |
|---|---|---|
| `the-verdict.txt` | 5,145 | 책 수치 재현·빠른 검증 |
| `tinyshakespeare.txt` | 338,025 | 실제 학습 체감 |
> `.gitignore`에 두 파일 + `mlruns/` 제외 (코드만 git 추적). 다운로드: the-verdict=rasbt 저장소, tinyshakespeare=karpathy char-rnn.

### 학습 루프 `train_gpt.py` (실행 보류)
- `calc_loss_batch`/`calc_loss_loader`/`evaluate_model`/`train_model_simple`(교재 5.2).
- 루프: `zero_grad → loss = calc_loss_batch → backward → step`, `eval_freq`마다 train/val loss 출력·기록.
- **MLflow 통합**(아래) + AdamW(lr 4e-4, weight_decay 0.1), device 자동(M4 Max=mps).

### MLflow 실험 추적 통합
- MLflow = 실험 기록장. 3동사: `log_params`(설정)·`log_metric`(성능, step별 곡선)·`log_model`(모델).
- 매핑: `GPT_CONFIG`+학습설정 → log_params / train·val loss → log_metric(step=global_step) / `GPTModel` → log_model.
- 추적 데이터는 `file:./mlruns`(gitignore). UI: `mlflow ui`. 개념 정본 = [[../../30-References/mlops]].

## 검증

사전훈련 loss 곡선이 우하향. 공개 GPT-2 가중치 로드 후 텍스트 생성이 의미 있는 출력.
MLflow UI에서 train/val loss 곡선이 그려지고 run 간 비교 가능.

## 관련 노트

- [[llm-from-scratch]] — 마스터 플랜 (인덱스)
- [[../../30-References/pytorch-env-hybrid]] — 환경 정본
- [[../../30-References/mlops]] — MLOps·MLflow 정본 (5.2 학습 루프에 MLflow 통합)
