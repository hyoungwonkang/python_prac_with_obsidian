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
| `test_size` | 테스트 데이터 **총 건수**. 메트릭의 신뢰도 판단 기준 (건수가 적으면 수치 변동이 큼) |
| `max_length` | 문장을 토큰으로 자를 최대 길이. 이보다 긴 문자는 잘림 |
| `batch_size` | 한 번에 모델에 넣는 **묶음 크기**. 크면 빠르지만 메모리 많이 씀 |

> **test_size vs batch_size**: 시험지 총 1,000장(test_size)을 32장씩 묶어(batch_size) 채점하는 구조.
> test_size는 데이터의 속성 → 메트릭의 **신뢰도**를 좌우. batch_size는 실행 설정 → 평가에서는 속도·메모리만 바뀌고 결과 수치는 동일.
> 단, **학습**에서는 batch_size가 가중치 업데이트 단위라 최종 성능에 영향 → 비교 실험 대상 하이퍼파라미터.
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

## 5. 헷갈렸던 것 Q&A (2026-07-02 학습 기록)

실제로 헷갈렸던 질문 순서 그대로. 위 표가 안 읽힐 때 여기부터 다시 볼 것.

### Q1. recall이라는 단어 자체가 무슨 뜻인가

영어 recall = "회수하다" (자동차 결함 리콜과 같은 단어). **"찾아야 할 것들을 얼마나 빠짐없이 회수했는가"** 라는 감각.
실제 스팸 100건 중 95건을 잡으면 recall = 0.95, 놓친 5건이 fn. **분모가 "실제 스팸 전체"** 라는 게 핵심 —
그래서 recall이 낮다 = 스팸이 필터를 뚫고 받은편지함에 들어온다.

### Q2. 비슷한 탐지를 두 번 재는 것 같은데 왜 precision/recall 둘로 나눴나

한쪽만 보면 모델이 "치팅"해도 못 잡기 때문. 극단 예시 (정상 900 + 스팸 100):

- **전부 스팸이라고 하는 모델**: recall = 100/100 = **1.00** (만점!) 그러나 precision = 100/1000 = 0.10.
  recall만 보고하면 "탐지율 100%"인데 실제론 정상 900건이 전부 스팸함행 → 못 쓰는 모델.
- **가장 확실한 1건만 잡는 모델**: precision = 1/1 = **1.00** (만점!) 그러나 recall = 1/100 = 0.01.
  스팸 99%가 통과 → 필터가 일을 안 함. 참고로 이 모델의 accuracy는 901/1000 = 0.90으로 높게 나옴
  → **클래스 불균형에서 accuracy 하나만 보고하면 안 되는 이유**.

즉 같은 탐지를 두 번 재는 게 아니라 **서로 반대 방향의 실패를 하나씩 감시**: recall = 놓침 감시, precision = 오탐 감시.

### Q3. "스팸을 스팸이라 처리 / 스팸을 통과시키지 않음"이 precision/recall인가

**아니다 — 그 두 문장은 같은 말이고 둘 다 recall 쪽.** precision은 스팸이 아니라 **정상 메일 쪽**을 보는 지표.
precision 정의("스팸 판정 중 진짜 스팸 비율")에 '스팸'이 들어가서 스팸 잡는 얘기처럼 들리지만,
이 비율을 깎는 원인은 **정상 메일을 잘못 잡은 fp** → 실질 의미는 "정상 메일 보호".

> 외우기: **recall = 스팸을 놓쳤나? / precision = 정상을 건드렸나?**

### Q4. 그럼 수치를 올리려면 "정상을 스팸 처리 안 함(precision) / 스팸을 스팸이라 판단(recall)"이 맞나

**맞다.** 식에서 그대로 보임 — 분자는 둘 다 tp로 같고, **분모의 실수 종류만 다름**:

```
precision = tp / (tp + fp)   ← fp(정상인데 스팸 처리)가 늘면 깎임
recall    = tp / (tp + fn)   ← fn(스팸인데 통과시킴)이 늘면 깎임
```

숫자 예시 (정상 900 + 스팸 100, tp=90 fn=10 fp=30):
- precision = 90/(90+30) = 0.75 → 스팸함 120통 중 30통이 억울한 정상 메일. 이 30을 줄이는 게 precision 올리기.
- recall = 90/(90+10) = 0.90 → 스팸 10통이 받은편지함으로 새어 들어옴. 이 10을 줄이는 게 recall 올리기.

단, 둘은 트레이드오프: 필터 기준을 조이면 fn↓(recall↑) 대신 fp↑(precision↓), 풀면 반대.
모델이 진짜 좋아진다 = 트레이드오프를 넘어 **fp·fn을 둘 다 줄이는 것** → 한 숫자 확인은 `spam_f1`.

### Q5. FP(false positive)에서 false가 무슨 뜻인가

- **positive/negative** = **모델의 판정** (positive = "찾으려는 것(스팸=1)이다", negative = "아니다(정상=0)")
- **true/false** = **그 판정이 맞았나/틀렸나**

그래서 FP = "틀린 스팸 판정" = 정상 메일에 누명. **FN은 "negative가 틀림" = 정상이라 했는데 사실 스팸 = 놓친 스팸**
(false를 "실제가 negative"로 읽으면 헷갈리니 주의 — false는 판정의 정오).

positive가 스팸인 이유: positive는 좋다는 뜻이 아니라 **"검사가 찾으려는 대상"** (코로나 검사 '양성'과 같은 용법).
스팸 분류기의 목적이 스팸 찾기이므로 스팸=1=positive.

연결: **precision을 깎는 fp = 틀린 스팸 판정(누명) / recall을 깎는 fn = 틀린 정상 판정(놓침)**.

---

## 6. 주간보고 한 줄 템플릿

> "영어 BERT(bert-base-uncased) 스팸 분류기를 test셋 N건으로 평가한 결과,
> **accuracy 0.98, spam recall 0.95**(실제 스팸의 95%를 탐지),
> **spam precision 0.97**(스팸 판정 중 97%가 진짜 스팸), **F1 0.96**.
> 오탐(fp) X건, 놓침(fn) Y건."

숫자는 MLflow UI → experiment `bert-spam-classification` → 해당 run의 Metrics 탭에서 그대로 읽으면 됨.
