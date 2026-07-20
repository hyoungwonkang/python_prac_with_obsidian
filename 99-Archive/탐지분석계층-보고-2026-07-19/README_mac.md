# README (macOS) — 탐지 계층 R&D 통합 보고 패키지

> **Windows는 `README_windows.md` 참조.** 클린 macOS 기준(Apple Silicon 검증 — M4 Max, GPU=MPS 자동 사용).
> 명령은 **터미널**에서 이 폴더(`탐지계층-R&D-보고/`)로 이동 후 실행.

**추출 계층**(OCR: 이미지→텍스트)과 범용 **1차 탐지 계층**(텍스트 분류·PII·NER·이미지 탐지·상황 판단)을
공개 데이터로 로컬에서 end-to-end 재현한 R&D 묶음입니다. 특정 도메인이 아니라 **도메인 교체 가능한 범용 계층**이 목적이며,
적용처는 아동·청소년 성착취물 대응, 불법스포츠도박 등 복수에 걸칩니다.

## 산출물 한눈에

| # | 내용 | 핵심 결과 | 폴더 |
|---|---|---|---|
| 1 | 학습 데이터 산출물(확장성) | 규약만 맞추면 학습→산출물→재사용이 반복되는 환경 | `rnd-dataset-artifacts` |
| 2 | 분류 잘하는 법 | Rule vs BERT vs 하이브리드 — 단일 승자 없음, 업무별 선택지 | `rnd-rule-vs-bert` |
| 3 | 통합 UXUI | 모듈을 탭 5개 데모로 통합 — OCR 추출(②) 탭 포함 (신규 학습 0) | `rnd-uxui-demo` |
| 4 | CLIP 상황 판단 | KoCLIP+한국어 프롬프트 — 재학습 없이 분류 체계 교체 | `rnd-clip` |
| 5 | YOLO 직접 라벨링 | 라벨링→학습→탐지 실증 (mAP50 0.995) | `rnd-dataset-artifacts` |
| 6 | OCR 추출 계층 | Easy·Paddle 동일 잣대 비교 — deskew+Paddle CER 2.4%, ②추출 품질=③탐지 성능 실증 | `rnd-ocr` |
| 7 | OpenCV 실습 | 프레임 추출·전처리·**비식별화(블러)** — 영상→이미지 다리, 검수 비식별본 원칙 | `rnd-detection-models-2` |

진입점은 **통합 데모**(`rnd-uxui-demo`) — 위 모듈을 한 화면에서 시연.
`rnd-detection-models`(기존 NER·YOLO·PII)는 데모가 재사용하므로 함께 포함됩니다.

## 1. 가상환경 + 의존성 (버전 고정)

```bash
brew install python@3.12                # 없으면 (Homebrew: https://brew.sh)
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv rnd-env
source rnd-env/bin/activate              # 프롬프트 앞에 (rnd-env) 표시되면 성공
python -m pip install --upgrade pip
pip install -r requirements.txt          # 반드시 requirements.txt로 (버전 고정 조합)
```

설치 검증 (MPS=Apple GPU 사용 가능 여부까지):
```bash
python -c "import torch, transformers, datasets, ultralytics, ko_pii, gradio; print('OK', torch.__version__, '| MPS', torch.backends.mps.is_available())"
# 기대: OK 2.12.1 | MPS True
```
- **Python 3.12 필수** — 3.10·3.11은 설치 자체가 실패(numpy 2.5.1이 ≥3.12 요구, 실측). 첫 실행 시 모델 자동 다운로드(klue/bert-base·KoCLIP 등 약 1.5GB).
- 장치는 코드가 자동 감지 — 맥은 MPS(Apple GPU) 사용.

## 2. 가중치 — 배포본에 따라 다름

이 패키지는 두 형태로 배포됩니다. `rnd-dataset-artifacts/export/artifacts/ko-spam-full/model.pt`가 **있으면 전체본**(바로 실행), 없으면 **경량본**입니다.

- **전체본**: 스팸·NER·YOLO 가중치 포함 → 아무 것도 안 해도 §3으로.
- **경량본(가중치 미포함)**: 데모 실행 **전 1회** 아래로 가중치 2개를 생성하세요. 학습 데이터·자동 다운로드가 준비돼 있어 명령만 실행하면 됩니다.
  ```bash
  # (1) 스팸 분류 가중치 — 동봉 데이터로 학습 (~수 분)
  cd rnd-dataset-artifacts/export
  DATA=datasets/ko-spam NAME=ko-spam-full EPOCHS=2 python train_text.py
  # → artifacts/ko-spam-full/ 에 3종 세트 생성

  # (2) NER 가중치 — KLUE 데이터 자동 다운로드 후 학습 (~십수 분, 인터넷 필요)
  cd ../../rnd-detection-models/export
  NER_SUBSET=6000 EPOCHS=3 python finetune_ner.py     # → ner_klue.pt 생성
  cd ../..
  ```
  - 두 명령은 각각 `가중치 저장`류 메시지가 나오고 파일이 생기면 성공.
  - 급하면 스모크로 더 빠르게(정확도는 낮음): 스팸 `EPOCHS=1`, NER `NER_SUBSET=200 EPOCHS=1`.
  - **가중치 없이도 PII·이미지 분석·이미지 검색 탭은 즉시 동작** — 먼저 둘러본 뒤 위 학습을 돌려도 됩니다.
- KoCLIP·YOLO 기본 모델은 두 배포본 모두 첫 실행 시 자동 다운로드.
- OCR(EasyOCR) 모델도 첫 실행 시 자동 다운로드. 비교 엔진 PaddleOCR은 **격리 venv**(의존성 충돌 방지) — `3_사용법.md` 7) 참조.

## 3. 통합 데모 실행

```bash
cd rnd-uxui-demo/export
python app.py                            # → 브라우저 http://127.0.0.1:7860
YOLO_PT=<best.pt 경로> python app.py       # 직접 학습한 YOLO 모델로 교체
```
탭 5개: 텍스트 분석 / 검수·라벨링 / OCR 추출(② — deskew 전처리 자동 + Easy·Paddle 교차 검증) / 이미지 분석 / 이미지 검색·일괄.
개별 모듈 실행법은 `3_사용법.md`.

## 4. 폴더 지도 & 문서 읽는 순서

```
탐지계층-R&D-보고/
├─ README_mac.md / README_windows.md   ← 시작점 (OS별)
├─ 6_보고서.md / .pdf                    ← 공식 보고서 (수행기간·결과·기대효과)
├─ 5_도식도.md / .pdf                    ← 아키텍처·데이터 순환·처리 흐름 그림
├─ 1_연구문서.md   2_소스코드.md   3_사용법.md   4_가이드.md
├─ requirements.txt                      ← 전체 의존성 (한 파일)
└─ rnd-*/                                ← 모듈 코드 (dataset-artifacts·rule-vs-bert·clip·uxui-demo·detection-models·detection-models-2·ocr)
```
- **급하면**: `6_보고서` + `5_도식도` 두 개면 충분 (둘 다 PDF 있음 — 뷰어 없이 열람·인쇄 가능).
- **더 깊이**: `1_연구문서` → `4_가이드` → `3_사용법` → `2_소스코드`.

## 5. 문제 해결 (macOS)

| 증상 | 원인·조치 |
|---|---|
| `No matching distribution found for numpy==2.5.1` | Python이 3.12 미만 — venv를 3.12로 다시 생성 |
| `No module named 'ko_pii'` 등 | venv 활성화 후 `pip install -r requirements.txt` 재확인 |
| KLUE 로드 실패 (`Invalid HF URI 'klue'`) | `datasets`가 5.0.0인지 확인 (`pip show datasets`) — 구버전 미지원 |
| 스팸·NER 탭 "가중치 없음" | 가벼운 배포본 — 가중치 포함본을 쓰거나 §2대로 생성 (PII·이미지 탭은 동작) |
| Apple Silicon 연산 오류 | 코드에 `PYTORCH_ENABLE_MPS_FALLBACK=1` 내장. 결과가 이상하면 소량을 CPU와 비교 (MPS 조용한 오답 사례 있음) |
| 데모 포트 충돌(7860) | `app.py` 마지막 줄 `server_port` 변경 |
| 다운로드 실패 | 네트워크에서 huggingface.co, github.com 허용 확인 |
