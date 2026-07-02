# MLflow 파라미터·메트릭 용어집 (주간보고용)

> 이 vault의 MLflow 실험들(UI: http://127.0.0.1:5000)에 실제로 기록되는 **모든 파라미터·메트릭·태그**를
> 코드에서 추출해 정리한 용어집. 주간보고 때 "spam_recall이 뭐지?" 하고 헤매지 않기 위한 위키.
>
> MLflow 사용법 자체(왜 log_params vs log_metric인지, step의 의미, mlruns 경로 함정)는
> [[mlflow-llm-tracking]] 참조. 이 문서는 **단어의 뜻**에 집중.

## 0. 어느 실험에 뭐가 기록되나

| Experiment | 기록하는 스크립트 | 내용 |
|---|---|---|
| `bert-spam-classification` | `30-References/rnd-bert-labeling-test/export/eval_mlflow_spam.py` | 저장된 BERT 가중치를 test셋으로 평가만 (재학습 없음) |
| `gpt2-spam-classify` | `10-Projects/llm-from-scratch/model_finetune.py` | GPT-2를 스팸 분류기로 파인튜닝 (6장) |
| `gpt2-from-scratch` | `10-Projects/llm-from-scratch/train_with_mlflow.py` | GPT-2 밑바닥부터 사전학습 (5장) |
| `gpt2-finetune` | `10-Projects/llm-from-scratch/finetune_gpt2_mlflow.py` | 공개 GPT-2 가중치 로드 후 이어서 학습 (Colab) |

---

## 1. MLflow 자체 용어 (UI에서 보이는 구조)

| 용어 | 뜻 |
|---|---|
| **Experiment(실험)** | run들을 묶는 폴더. 예: `bert-spam-classification` |
| **Run** | 스크립트 1회 실행 = 기록 1건. run 이름(예: `en-bert-base-uncased`)으로 구분 |
| **Parameter(파라미터)** | 실행 **전에 내가 정한 설정값**. 한 run에 1개 값 고정 (예: `batch_size=8`) |
| **Metric(메트릭)** | 실행 **결과로 나온 측정값**. 숫자이고 step별로 여러 번 기록 가능 → 곡선이 됨 (예: `train_loss`) |
| **Tag(태그)** | run에 붙이는 라벨성 메모. 예: `stage=eval-only` (재학습 아니고 평가만 했다는 표시) |
| **Artifact(아티팩트)** | run에 첨부한 파일. 예: `classification_report.txt`, 저장된 모델 |
| **step** | 메트릭을 기록한 시점 번호(몇 번째 배치/에폭인지). x축이 되어 곡선을 그림 |

> 한 줄 요약: **파라미터 = 입력(설정), 메트릭 = 출력(성적표)**.

---

## 2. 파라미터 용어집

### 2-1. 데이터·평가 설정 (`bert-spam-classification`)

| 파라미터 | 뜻 |
|---|---|
| `model_name` | HuggingFace 모델 이름. `bert-base-uncased`(영어) / `klue/bert-base`(한국어) |
| `weights_file` | 로드한 학습 완료 가중치 파일. `spam_bert.pt` / `spam_klue.pt` |
| `test_csv` | 평가에 쓴 테스트 데이터 파일 경로 |
| `test_size` | 테스트 데이터 **건수**. 메트릭의 신뢰도 판단 기준 (건수가 적으면 수치 변동이 큼) |
| `max_length` | 문장을 토큰으로 자를 최대 길이. 이보다 긴 문자는 잘림 |
| `batch_size` | 한 번에 모델에 넣는 데이터 개수. 크면 빠르지만 메모리 많이 씀 |
| `device` | 연산 장치. `cpu` / `cuda`(NVIDIA GPU) / `mps`(Mac GPU) |

### 2-2. 모델 구조 (GPT-2 실험들의 `GPT_CONFIG`/`BASE_CONFIG`)

| 파라미터 | 뜻 |
|---|---|
| `vocab_size` | 어휘 사전 크기 = 모델이 아는 토큰 종류 수 (GPT-2: 50257) |
| `context_length` | 모델이 한 번에 볼 수 있는 최대 토큰 수 (문맥 창) |
| `emb_dim` | 임베딩 차원 = 토큰 1개를 표현하는 숫자 벡터의 길이 (768) |
| `n_heads` | 어텐션 헤드 수. 문맥을 몇 갈래 관점으로 나눠 보는지 (12) |
| `n_layers` | 트랜스포머 블록(층) 수. 깊을수록 크고 느리고 표현력↑ (12) |
| `drop_rate` | 드롭아웃 비율. 과적합 방지로 학습 중 뉴런을 랜덤하게 끄는 비율 (0.1 = 10%) |
| `qkv_bias` | 어텐션의 Q/K/V 계산에 bias 항을 쓸지 여부 (공개 GPT-2 가중치 로드 시 True 필수) |

### 2-3. 학습 하이퍼파라미터 (GPT-2 실험들의 `TRAIN`)

| 파라미터 | 뜻 |
|---|---|
| `learning_rate` (`lr`) | 학습률. 한 스텝에 가중치를 얼마나 크게 고칠지. 파인튜닝은 작게(5e-5) — 사전지식 보존 |
| `weight_decay` | 가중치가 너무 커지지 않게 누르는 규제(정규화). 과적합 방지 |
| `num_epochs` | 에폭 수 = 전체 학습 데이터를 처음부터 끝까지 몇 바퀴 도는지 |
| `train_ratio` | 전체 데이터 중 학습용 비율 (0.90 = 90% 학습, 10% 검증) |
| `eval_freq` | 몇 배치(step)마다 중간 평가를 해서 loss를 기록할지 |
| `eval_iter` | 중간 평가 때 몇 개 배치만 샘플로 loss를 잴지 (전체를 다 재면 느려서) |
| `dataset` | 학습에 쓴 데이터 파일 이름 (예: `tinyshakespeare.txt`) |

---

## 3. 메트릭 용어집

### 3-1. 스팸 분류 평가 (`bert-spam-classification`) ← 주간보고 핵심

라벨 규칙: **0 = ham(정상 메시지), 1 = spam**. `spam_*` 메트릭은 전부 **"spam(=1)을 기준으로"** 계산한 값.

먼저 혼동행렬(confusion matrix) 4칸 — 모든 지표의 재료:

|  | 예측: ham | 예측: spam |
|---|---|---|
| **실제: ham** | `cm_tn` (True Negative) — 정상을 정상이라 함 ✅ | `cm_fp` (False Positive) — **정상인데 스팸이라 오판** ⚠️ |
| **실제: spam** | `cm_fn` (False Negative) — **스팸인데 놓침** ⚠️ | `cm_tp` (True Positive) — 스팸을 스팸이라 잡음 ✅ |

| 메트릭 | 식 | 뜻 (보고서 문장으로) |
|---|---|---|
| `test_accuracy` | (tp+tn) / 전체 | 전체 중 맞힌 비율. **주의: 스팸이 드물면 다 ham이라 해도 높게 나옴** → 이것만 보고하면 안 됨 |
| `spam_precision` | tp / (tp+fp) | **"스팸이라고 판정한 것 중 진짜 스팸 비율."** 낮으면 → 정상 메시지를 스팸함으로 보내는 오탐(false alarm)이 많다 |
| `spam_recall` | tp / (tp+fn) | **"실제 스팸 중 잡아낸 비율."** 낮으면 → 스팸이 필터를 뚫고 들어온다(놓침) |
| `spam_f1` | 2·P·R / (P+R) | precision과 recall의 조화평균. 둘 다 좋아야 높아짐 → **한 숫자로 보고할 때 이걸 씀** |
| `cm_tn`/`cm_fp`/`cm_fn`/`cm_tp` | (위 표) | 혼동행렬 4칸의 실제 건수. 비율이 아니라 개수 |

> **precision vs recall 트레이드오프**: 필터를 빡빡하게 하면 recall↑(스팸 다 잡음) 대신 precision↓(정상도 스팸 처리).
> 스팸 필터에서는 보통 **fp(정상을 스팸 처리)가 fn(스팸 놓침)보다 치명적** → precision을 더 챙겨보는 경우가 많음.

### 3-2. 학습 곡선 (GPT-2 실험들, step별 기록 → UI에서 그래프)

| 메트릭 | 뜻 |
|---|---|
| `train_loss` | 학습 데이터에 대한 손실(loss). 모델 예측이 정답과 얼마나 다른지 — **낮을수록 좋음** |
| `val_loss` | 검증(validation) 데이터에 대한 loss. **학습에 안 쓴 데이터**로 잰 실력 |
| `train_acc` / `val_acc` | 에폭별 학습/검증 정확도 (분류 실험 `gpt2-spam-classify`만) |

> 읽는 법: `train_loss`만 내려가고 `val_loss`가 다시 올라가면 → **과적합(overfitting)** — 암기 시작한 것.
> 두 곡선이 같이 내려가면 정상적으로 배우는 중.

### 3-3. 최종 요약값 (run 끝날 때 1회 기록)

| 메트릭 | 뜻 |
|---|---|
| `final_train_loss` / `final_val_loss` | 학습 종료 시점의 마지막 train/val loss. run끼리 표로 비교할 때 씀 |
| `final_train_acc` / `final_val_acc` | 학습 종료 시점의 학습/검증 정확도 |
| `final_test_acc` | **테스트셋** 정확도. 학습·검증 어디에도 안 쓴 데이터라 **최종 성적표로 보고할 값** |
| `train_minutes` | 학습에 걸린 시간(분) |

> train / val / test 구분: **train = 공부한 문제집, val = 공부 중 모의고사, test = 수능**.
> 보고할 최종 성능은 test 기준(`final_test_acc`, `test_accuracy`)을 쓰는 게 원칙.

---

## 4. 태그·아티팩트 (bert-spam-classification)

| 항목 | 뜻 |
|---|---|
| tag `stage=eval-only` | 이 run은 재학습이 아니라 저장된 가중치 평가만 했다는 표시 |
| tag `weights` | 어떤 .pt 가중치 파일을 평가했는지 |
| artifact `classification_report.txt` | sklearn classification_report 전문 — ham/spam 클래스별 precision·recall·f1 표 |
| artifact `gpt2_spam_classifier` 등 | `mlflow.pytorch.log_model`로 저장한 모델 본체 |

---

## 5. 주간보고 한 줄 템플릿

> "영어 BERT(bert-base-uncased) 스팸 분류기를 test셋 N건으로 평가한 결과,
> **accuracy 0.98, spam recall 0.95**(실제 스팸의 95%를 탐지),
> **spam precision 0.97**(스팸 판정 중 97%가 진짜 스팸), **F1 0.96**.
> 오탐(fp) X건, 놓침(fn) Y건."

숫자는 MLflow UI → experiment `bert-spam-classification` → 해당 run의 Metrics 탭에서 그대로 읽으면 됨.
