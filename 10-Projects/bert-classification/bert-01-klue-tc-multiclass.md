# bert-01 — KLUE-TC 다중 클래스 분류

[[../bert-classification]] **Phase 1** 정본. 시작 2026-07-07.
스팸 2-class([[../../30-References/rnd-bert-labeling-test-plan|업무 R&D]])를 **7클래스로 확장** — 코드 골격 재사용, `num_labels`만 변경한다는 계획의 실행.

## 목표

KLUE-TC(ynat, 뉴스 제목 주제 분류 7클래스)로 klue/bert-base를 파인튜닝해
**다중 클래스 분류의 차이점**(num_labels·다중 클래스 지표)을 체득한다.
검증 3종: ① 미니 데이터 end-to-end 로그 ② MLflow run(한글 키) ③ 이 노트.

## 체크리스트

- [x] 코드 작성 — `finetune_klue_tc.py` (spam 골격 재사용: TcDataset·evaluate·학습루프 동일, 변경점은 num_labels 7·KLUE-TC 로드·macro F1)
- [x] 스모크 (TC_SUBSET=500, 1ep) — 1차에서 **mlflow 3.14 파일스토어 거부 발견** → sqlite 전환 후 end-to-end 통과
- [x] 본 실행 (TC_SUBSET=10000, 2ep) — **accuracy 0.8425 · macro F1 0.8313** (1.9분, MPS) + MLflow run `7class-10000건`
- [x] 학습 메모 정리 (아래)

> **✅ Phase 1 완료 (2026-07-07).** 검증 3종 충족: 실행 로그 · MLflow run(한글 키, sqlite) · 이 노트.

## 스팸 대비 변경점 (계획 그대로 확인)

| | 스팸 (2-class) | KLUE-TC (7-class) |
|---|---|---|
| 분류층 | 768→2 | **768→7** (`num_labels=len(label_names)`) |
| 데이터 | UCI CSV 직접 다운로드·분할 | HF `klue/klue` config `ynat` — **분할 제공** (test 라벨 비공개→validation 평가, NER과 동일) |
| 입력 | SMS 본문 (MAX_LENGTH 128) | 뉴스 **제목** (짧음 → MAX_LENGTH 64로 절약) |
| 지표 | accuracy + spam P/R/F1 | accuracy + **클래스별 P/R/F1 + macro F1** |
| MLflow | bert-spam-classification (file store) | `bert-klue-tc` (**sqlite** — 아래 함정 참조) |

7클래스: IT과학·경제·사회·생활문화·세계·스포츠·정치 (label 0~6).

## 실행 기록

- 2026-07-07 스모크 1차: `mlflow.set_experiment`에서 예외 — **mlflow 3.14가 파일 스토어(file:./mlruns)를 유지보수 모드로 전환, 기본 거부**. ml-env(3.1.4)에선 되던 방식 — 상사가 강조한 "의존성 버전 문제"의 실사례. 조치: `sqlite:///mlflow.db`로 전환 (권장 방식).
- 스모크 2차 (500건·1ep): end-to-end 통과. acc 0.45 — 극소량 학습이라 당연히 낮음. 일부 클래스 미예측 → `UndefinedMetricWarning` (아래 메모).
- **본 실행 (10,000건·2ep, MPS 1.9분): accuracy 0.8425 · macro F1 0.8313.** 클래스별 F1 0.71(IT과학)~0.91(스포츠). epoch1→2에서 val_acc 0.809→0.843 (진행 정상). KLUE-TC 공식 베이스라인(전체 45k 학습 시 ~0.86)에 근접 — 1/4 데이터로 이 수준이면 파이프라인 유효. MLflow run `7class-10000건` (sqlite, 한글 키).

## 학습 메모 (다중 클래스에서 새로 배운 것만)

- **num_labels 한 줄 확장이 실제로 성립**: 스팸 코드에서 실질 변경은 `num_labels=7`·데이터 로드부·지표뿐 — "head의 칸 수만 문제에 맞춘다"(kickoff의 분류 FT 본질)가 실증됨.
- **macro vs weighted 평균** (다중 클래스 신개념): macro = 클래스를 **동등하게** 평균(소수 클래스 IT과학의 0.71이 1/7 그대로 반영) / weighted = **건수 가중**(다수 클래스 사회 854건이 지배). accuracy(0.8425)와 macro F1(0.8313)의 간극 = 소수 클래스가 평균을 끌어내린 크기. **불균형에서 성능을 정직하게 보려면 macro** — 스팸 때 "정확도가 불균형에 속는다" 교훈의 다중 클래스판.
- **클래스 불균형의 다른 대처**: 스팸 땐 언더샘플링으로 균형화했지만, 여기선 원 분포 그대로 학습(사회 854 vs IT과학 95)하고 **지표(macro)로 감시** — 균형화가 유일한 답이 아님.
- **UndefinedMetricWarning의 뜻**: 모델이 한 번도 예측하지 않은 클래스는 정밀도 분모(뽑은 수)가 0 → 0/0 정의 불가 → sklearn이 0으로 처리하며 경고. 극소량 학습(스모크)에서 자연 발생 — NER 스모크의 seqeval 경고와 동일 현상.
- **mlflow 3.14 함정**: 파일 스토어가 유지보수 모드(예외 발생) → `sqlite:///mlflow.db` 권장. 같은 코드가 라이브러리 버전에 따라 죽고 사는 실사례 — requirements 버전 고정 정책의 근거.

## 심화 Q&A (2026-07-07~08, 실험으로 검증한 것)

### 분류층의 해부
- **실체 = Linear(768→7) 한 장**: W[7×768] + b[7] = **5,383눈금** (전체 1.1억의 0.005%). 계산 = 클래스별 "자(W의 행)" 7개를 [CLS] 좌표에 내적 + 출발선(b) — 점수 = x·w + b라는 초등 산수의 768차원판.
- **768→7 압축의 정체 = 내적의 합(Σ)**: 자 하나가 곱 768개를 **전부 더해 숫자 1개로 요약** → 자 7개 = 로짓 7칸. 5,383은 추려지는 대상이 아니라 **기계의 부품** (변환되는 건 입력 768→출력 7).
- **bias의 두 역할**: ① 클래스별 출발선 — 다수 클래스(사회 854건)의 사전 성향이 학습으로 새겨짐 ② 판정 경계선을 원점에서 밀어내는 평행이동 (y=ax+b의 그 b).
- **값의 출처 실험** (두 번 로드 비교): 같은 `from_pretrained(num_labels=7)` 호출 두 번 → W_query는 **항상 동일**(파일에서 로드) / classifier는 **매번 다름**(난수). 즉 num_labels는 **모양(칸 수)만 지시**, 값은 백지 — KLUE 파일에 분류층이 없다는 LOAD REPORT `MISSING`의 실증. `torch.manual_seed`가 필요한 이유이기도. 정의 실코드: transformers `modeling_bert.py:1087` (NER 13칸은 1265행 — 같은 한 줄). 학습된 값의 거처: `bert_klue_tc.pt`의 `classifier.weight`.

### 가중치와 기울기 (loss→backward→step의 실체)
- **W_query의 3시점**: ① 실행 시 HF 캐시 safetensors에서 키 이름(`...layer.0.attention.self.query.weight`)으로 로드(실측) ② 그 값의 원산지 = KLUE 팀이 난수에서 MLM 사전학습으로 구움 ③ 우리 파인튜닝이 사본을 미세 수정 → .pt로. "남이 만든 눈금을 받아, 조금 고쳐, 내 파일로"가 전이학습의 물리 실체.
- **기울기(gradient) ≠ 가중치**: `backward()`는 가중치를 안 건드리고 **가중치마다 짝꿍 텐서 `.grad`** 에 [방향(부호)+민감도(크기)]를 씀 → `step()`이 `w −= LR×grad`로 이동. 미니 실측 검산: 0.0669 + 0.01×20.02 = 0.2671 ✓. loss **숫자 1개**가 역전파로 **1.1억 개 기울기**(책임 배분)로 펼쳐짐. 이름 유래 = 언덕의 기울기 → 경사하강.
- **LR 2e-5(0.00002) = 아주 작은 보폭**: 사전학습 지식(1.1억 눈금)을 보존하며 살짝만 새기기 위함 — BERT 파인튜닝 표준 대역(2e-5~5e-5). 너무 크면 지식 파괴·발산, 너무 작으면 수렴 못 함.
- **파라미터 실측**: klue/bert-base = **110,617,344개** (임베딩 2,500만 + 인코더 8,500만 = 층당 709만 — Q/K/V/출력 4×768² ≈ 236만 + FFN 472만). 파일 크기 검산: 개수×4바이트(fp32) ≈ 442MB ✓. **768 = base "등급"의 hidden_size**(BERT 계열 상수 아님 — large는 1024; GPT-2 emb_dim과 동일 개념; 12헤드×64; 체크포인트 안에서는 불변).

### 데이터·평가의 정밀 이해
- **ynat = KLUE-TC의 공식 config명** (연합뉴스 **제목** 주제 분류). MAX_LENGTH 64의 근거 실측: 제목 평균 15.4토큰·최대 26·**64 초과 0%** → 자름 없이 어텐션 비용(길이²) 1/4 절약 — "데이터 길이 분포를 보고 천장을 정한다" 실천.
- **리더보드와 validation의 역할 대행**: KLUE test는 **문제지만 공개, 정답지(라벨)는 금고** — "마지막에 한 번만" 규율의 시스템화(오염 원천 차단). 정답 없는 세트로는 채점(gold vs pred 대조)이 불성립 → **라벨 있는 validation이 최종 시험지 역할을 겸직** (별개 세트의 역할 대행이지 test=validation이 아님). 준시험지 단서: 공식 성적은 리더보드 제출 채점이 정도.
- **테스트 라벨 ⊂ 테스트셋**: 셋 = (문제, 정답) 쌍의 묶음, 라벨 = 그중 정답지 부분 — "라벨만 잠근다"는 제3의 배포 방식이 리더보드를 가능케 함.
- **3렌즈 지표 체계**: accuracy(빠른 요약, 다수 클래스에 지배됨) / 클래스별 P·R·F1(진단표 — IT과학 0.71 최약, 스포츠 0.91 최강) / macro F1(동등평균 정직 요약, KLUE 공식 지표). **acc−macro 간극(우리 0.011) = 소수 클래스 희생의 신호** — 클수록 경고. macro의 재료 7개 = 각 주제의 F1(성적표 f1-score 열).

## 다음 (Phase 2 후보)

- Korean HateSpeech (도메인 유사) — 마스터 노트 Phase 2
- 전체 데이터(45k)·에폭 확대해 베이스라인(~0.86) 도달 시도 (선택)

## 관련 노트

- [[../bert-classification]] — 마스터 (Phase 1 정의)
- [[bert-00-kickoff]] — 전환 정리 (head 교체=num_labels 논리)
- [[../../30-References/mlflow-practice/mlflow-terms-glossary]] — macro 평균 등 지표
- [[../../30-References/rnd-detection-models/00-학습메모]] — klue/klue 로드·KLUE test 라벨 비공개 등 선행 확인 사항
