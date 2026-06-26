# llm-ch5-pretrain

[[llm-from-scratch]] 마스터 플랜의 **Phase 5 / 5장** 정본 노트.
교재: Sebastian Raschka, *밑바닥부터 만들면서 배우는 LLM* — 5장. 레이블이 없는 데이터를 활용한 사전 훈련.

## 개요

cross-entropy/perplexity 기반 학습 루프를 구현해 소규모 코퍼스로 사전훈련하고, OpenAI 공개 GPT-2 가중치를 자체 구조에 매핑한다. 실제 학습은 Colab T4가 메인.

## 체크리스트

> 번호 = 교재 절 번호. (학습률 워밍업·코사인 감쇠·gradient clipping 등은 5장 본문이 아니라 **부록 D** — 별도.)

- [~] 5.1 텍스트 생성·손실 평가 (cross-entropy, train/val loss) — `text_model.py`. perplexity 미작성
- [x] 5.2 LLM 훈련 루프 (forward → loss → backward → optimizer.step) — `train_gpt.py`
- [x] 5.3 디코딩 전략 (temperature·top-k + 통합 `generate`) — `decoding.py`
- [x] 5.4 모델 가중치 저장/로드 (체크포인트) — `decoding.py` (`torch.save`/`load_state_dict`)
- [x] 5.5 OpenAI 공개 GPT-2 가중치 로드 → 자체 구조 매핑 — `weight_load.py` (**Colab 성공**)
- [ ] (실행) Colab T4에서 소규모 사전훈련/파인튜닝

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
- AdamW(lr 4e-4, weight_decay 0.1), device 자동(M4 Max=mps).
> MLflow 실험 추적은 교재 외 별도 트랙 → [[../../30-References/mlflow-practice/quickstart-notes]] 참조.

### 에포크(epoch)
- **훈련 데이터 전체를 빠짐없이 한 번씩 다 학습한 한 바퀴.** 배치 단위 step들이 모여 1 에포크. 여러 에포크 반복으로 loss↓ (과도하면 과적합).
- 코드: 바깥 `for epoch` 1회 = 1 에포크, 안쪽 `for batch in train_loader` = step 순회.

### train/val 손실 그래프 (matplotlib)
- `train_model_simple`이 모은 `train_losses`/`val_losses`를 `plot_losses`로 시각화.
- 완성 요소: `legend`, `MaxNLocator(integer=True)`(에포크 정수 눈금), `ax1.twiny()`로 "본 토큰 수" 위쪽 축, `tight_layout`.
- ⚠️ **CLI에선 `plt.show()`로 창이 안 뜸 → `plt.savefig("loss-plot.png")` 필수.** (`loss-plot.png` gitignore)

### 디코딩 전략 (5.3) — `decoding.py` (2026-06-25)

**토큰 생성 단계**: LLM은 자기회귀로 **한 토큰씩** 생성. 한 단계 = "현재까지 토큰 → model → 다음 토큰 1개 → 이어붙임". `max_new_tokens` = 단계 수. 직전 출력이 다음 입력에 포함됨.

**greedy의 한계**: `generate_text_simple`은 매 단계 `argmax`(1등만) → **결정적** → 같은 입력엔 항상 같은 출력. 재현엔 좋지만 다양성 0. → 그래서 샘플링 도입.

- **temperature scaling**: `logits / T` 후 softmax. T<1=뾰족(보수적), T>1=평평(과감), T=0=greedy. 분포 모양을 조절해 1등 아닌 토큰도 가끔 선택.
- **top-k sampling**: `torch.topk`로 상위 k개만 남기고 나머지 로짓을 `-inf`로 마스킹 → softmax → 후보를 k개로 제한.
- **`torch.multinomial(probs, num_samples=1)`**: 확률에 비례한 **랜덤 추출**(argmax 대체). 매번 다름 — `torch.manual_seed`로 고정 가능.
- **통합 `generate(... temperature, top_k, eos_id)`**: top_k면 마스킹 → temperature>0이면 multinomial 샘플링, 아니면 argmax → eos면 중단. **`generate_text_simple`의 상위호환**(temperature=0·top_k=None이면 greedy와 동일).

**비교 관점**: greedy("출력 테스트") vs 샘플링("출력 텍스트")을 같은 입력으로 나란히 출력 → 디코딩 전략이 결과를 바꾼다. 단 **미학습 모델이면 둘 다 무의미** → 품질 비교는 학습 후에만 의미(동작 차이만 지금 확인 가능).

> 흔한 버그: `multinomial`을 `multimodal`로 오타 / `generate` 본문 들여쓰기(temperature 블록이 top_k 블록 안으로) / `return`이 for 루프 안 → 1토큰만 생성. `next_token_logits`는 vocab 크기(9)와 길이 일치 필요.

### 5.4 모델 가중치 저장/복원(체크포인트) (2026-06-25)
- **state_dict** = 학습된 값의 스냅샷(모델=가중치/편향, 옵티마이저=모멘텀/분산/step).
- 저장: `torch.save(model.state_dict(), "model.pth")` / 둘 다: `torch.save({"model_state_dict":..., "optimizer_state_dict":...}, ...)`.
- **복원** = 새로 만든(랜덤) 모델·옵티마이저에 저장값을 **`load_state_dict`로 덮어써** 저장 시점 상태로 되돌림. `torch.load`(디스크→메모리) → `load_state_dict`(객체에 주입).
- 추론만이면 모델만, **학습 이어가기(resume)면 옵티마이저까지** 복원해야 모멘텀 유지돼 매끄러움.
- 이유: 재학습 비용 회피 / resume / 배포. (흔한 오타: `load_state_dict`를 `load_stat_dict`로)

### AdamW (옵티마이저)
- `optimizer.step()`의 업데이트 알고리즘. SGD + **모멘텀(1차)** + **파라미터별 적응 보폭(2차)** = Adam, 거기에 **분리된 weight decay**(=W, 과적합 방지) = AdamW. 트랜스포머 표준.
- 비용: 파라미터당 추가 상태 2개(모멘텀·분산) → 학습 메모리 ↑.

### LLM 사전훈련이 비싼 이유
- 비용 ≈ **파라미터 × 토큰 × 에포크 × (학습은 forward+backward+optimizer로 ×3)** — 곱셈이라 작은 값도 곱하면 폭발(124M·5천토큰·10에포크 ≈ 수십조 FLOPs).
- 학습은 추론보다 비쌈: backward·옵티마이저 상태·활성값 저장. → 대형 모델은 **공개 가중치 재사용**(5.6)이 현실적.

### 5.5 OpenAI GPT-2 가중치 로드 + 환경 전환 (TF → Colab)
- `download_and_load_gpt2(model_size="124M", models_dir="gpt2")` → `settings`·`params` 반환 → 본인 `GPTModel`에 매핑. (흔한 오타: `models_dir`를 `model_dir`로 → `TypeError: unexpected keyword argument`)
- **`settings` vs `params`**: `settings`=하이퍼파라미터(작은 dict, n_layer·n_head·n_embd 등). **`params`=실제 가중치 텐서 딕셔너리**(거대).
  - params **키**=가중치 이름/위치(`wte`토큰임베딩·`wpe`위치임베딩·`blocks`블록들·`g`/`b`최종LayerNorm), **값**=텐서.
  - `params.keys()`로 **구조만**(읽기 쉬움), `print(params)`는 1.2억 값 전부(방대). `params["wte"].shape` → `(50257, 768)`. 보통 `.shape`로 크기만 확인.
- 파일: `weight_load.py`(gpt_download를 rickiepark/rasbt repo에서 받아 호출).
- ⚠️ **로컬(시스템 파이썬 3.9) TF가 import 단계에서 크래시**(`mutex lock failed`) → **Colab으로 전환**(TF 정상). `gpt_download.py`가 상단에서 `import tensorflow` 하기 때문 = `from gpt_download import ...`(import) 단계에서 죽음. **로컬 ❌ / Colab ✅ 확정**(Colab 에러가 import 이후 함수호출에서 난 게 증거). 정본 [[../../30-References/pytorch-env-hybrid]] 참조.

#### `assign` + `load_weights_into_gpt` (가중치 매핑)
- **`assign(left, right)`**: ① `left`(내 모델 자리)·`right`(GPT-2 가중치) **shape 일치 검사**(안 맞으면 에러) → ② `right`를 `nn.Parameter`로 변환. 손수 짠 매핑의 **크기 틀림 실수를 잡는 안전장치**.
- **왜 틀릴 수 있나**: 매핑이 **수작업**(OpenAI 이름 `c_attn`/`wte` ↔ 내 모델 `W_query`/`tok_emb`). 전치·QKV분할·인덱스 착각 가능 → assign이 크기 불일치는 잡음(Q↔K 같은 동일크기 실수는 못 잡음).
- **`load_weights_into_gpt(gpt, params)`** 핵심 변환:
  - 임베딩(`wpe`/`wte`) 그대로, 나머지 Linear는 **`.T` 전치**(GPT-2는 PyTorch와 행/열 반대로 저장).
  - **QKV**: `c_attn` 한 덩어리(768×2304)를 `np.split(...,3,axis=-1)`로 Q·K·V(각 768×768) 분리.
  - **weight tying**: `out_head`는 `params["wte"]` 재사용.
  - ⚠️ `qkv_bias=True` 필요(GPT-2 어텐션에 bias 있음 → `W_query.bias` 등 존재해야 복사됨).

#### ✅ 5.5 성공 (2026-06-25)
- 미학습(랜덤)=무의미 토큰 → **GPT-2 로드 후 = 의미 있는 영어 문장** 생성 확인 (예: "Every effort moves you as far as the hand can go..."). 일관된 영어 = **매핑 정확**의 증거.
- temperature 높으면(1.5) 문법 OK·내용 자유분방. 재학습 없이 좋은 모델 확보 = 사전훈련 비용 회피.

#### 파인튜닝 (GPT-2 가중치에서 추가 훈련)
- GPT-2 로드한 `gpt`를 the-verdict로 **계속 훈련 = 파인튜닝**(랜덤 시작과 달리 **시작 loss 낮음**, 작은 데이터라 과적합 빠름). 가중치 로드만 TF(Colab), **훈련은 순수 PyTorch**(로컬도 가능).
- 주의: 모델 `context_length=1024`지만 데이터로더 `max_length`는 ≤1024(작은 데이터엔 256 권장).

### Colab 운영 메모 (로컬 코드 옮길 때)
- **cwd = `/content`**(루트 `/` 아님). import·`open()`은 cwd 기준 → 필요한 파일을 cwd에 둬야 함.
- import 문에 **경로를 못 박음**(SyntaxError). 폴더는 `%cd` 또는 `sys.path.append(폴더)`로 알려주고 import는 모듈명만(`from previous_5 import ...`).
- `git clone`은 **추적 파일만** 가져옴 → `gpt_download.py`(미추적)·`the-verdict.txt`(gitignore)는 **따로 다운로드**. device는 자동 `cuda` 선택(수정 불필요), `plt.show()`는 인라인.

## 검증

사전훈련 loss 곡선이 우하향. 공개 GPT-2 가중치 로드 후 텍스트 생성이 의미 있는 출력.

## 관련 노트

- [[llm-from-scratch]] — 마스터 플랜 (인덱스)
- [[../../30-References/pytorch-env-hybrid]] — 환경 정본
