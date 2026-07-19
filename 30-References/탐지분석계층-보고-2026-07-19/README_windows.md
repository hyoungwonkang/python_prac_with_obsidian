# README (Windows) — 탐지 계층 R&D 통합 보고 패키지

> **macOS는 `README_mac.md` 참조.** 클린 Windows 기준.
> 명령은 **PowerShell**에서 이 폴더(`탐지계층-R&D-보고\`)로 이동 후 실행.

**추출 계층**(OCR: 이미지→텍스트)과 범용 **1차 탐지 계층**(텍스트 분류·PII·NER·이미지 탐지·상황 판단)을
공개 데이터로 로컬에서 end-to-end 재현한 R&D 묶음입니다. 특정 도메인이 아니라 **도메인 교체 가능한 범용 계층**이 목적이며,
적용처는 아동·청소년 성착취물 대응, 불법스포츠도박 등 복수에 걸칩니다.

## 산출물 한눈에

| # | 내용 | 핵심 결과 | 폴더 |
|---|---|---|---|
| 1 | 학습 데이터 산출물(확장성) | 규약만 맞추면 학습→산출물→재사용이 반복되는 환경 | `rnd-dataset-artifacts` |
| 2 | 분류 잘하는 법 | Rule vs BERT vs 하이브리드 — 단일 승자 없음, 업무별 선택지 | `rnd-rule-vs-bert` |
| 3 | 통합 UXUI | 모듈 5개를 탭 4개 데모로 통합 (신규 학습 0) | `rnd-uxui-demo` |
| 4 | CLIP 상황 판단 | KoCLIP+한국어 프롬프트 — 재학습 없이 분류 체계 교체 | `rnd-clip` |
| 5 | YOLO 직접 라벨링 | 라벨링→학습→탐지 실증 (mAP50 0.995) | `rnd-dataset-artifacts` |
| 6 | OCR 추출 계층 | Easy·Paddle 동일 잣대 비교 — deskew+Paddle CER 2.4%, ②추출 품질=③탐지 성능 실증 | `rnd-ocr` |

진입점은 **통합 데모**(`rnd-uxui-demo`) — 위 모듈을 한 화면에서 시연.
`rnd-detection-models`(기존 NER·YOLO·PII)는 데모가 재사용하므로 함께 포함됩니다.

## 0. 사전 요구

- **Python 3.12 필수** (3.10·3.11은 numpy 2.5.1 설치 실패 — 실측): https://www.python.org/downloads/ 에서 설치, **"Add python.exe to PATH" 체크**. 확인: `py -3.12 --version`
- 인터넷 연결 (첫 실행 시 모델 자동 다운로드 약 1.5GB)

## 1. 가상환경 + 의존성 (버전 고정)

```powershell
py -3.12 -m venv rnd-env
.\rnd-env\Scripts\Activate.ps1           # 프롬프트 앞에 (rnd-env) 표시되면 성공
python -m pip install --upgrade pip
pip install -r requirements.txt          # 반드시 requirements.txt로 (버전 고정 조합)
# NVIDIA GPU가 있으면 (없으면 CPU로 자동 동작):
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu124
```
- ⚠️ Activate가 막히면: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 후 재시도.

설치 검증 (CUDA=NVIDIA GPU 사용 가능 여부까지):
```powershell
python -c "import torch, transformers, datasets, ultralytics, ko_pii, gradio; print('OK', torch.__version__, '| CUDA', torch.cuda.is_available())"
# 기대: OK 2.12.1 | CUDA True(GPU 있을 때) / False(CPU만)
```
- 장치는 코드가 자동 감지 — NVIDIA GPU면 CUDA, 없으면 CPU.

## 2. 가중치 — 배포본에 따라 다름

이 패키지는 두 형태로 배포됩니다. `rnd-dataset-artifacts\export\artifacts\ko-spam-full\model.pt`가 **있으면 전체본**(바로 실행), 없으면 **경량본**입니다.

- **전체본**: 스팸·NER·YOLO 가중치 포함 → 아무 것도 안 해도 §3으로.
- **경량본(가중치 미포함)**: 데모 실행 **전 1회** 아래로 가중치 2개를 생성하세요 (PowerShell `$env:` 형식).
  ```powershell
  # (1) 스팸 분류 가중치 — 동봉 데이터로 학습 (~수 분)
  cd rnd-dataset-artifacts\export
  $env:DATA="datasets/ko-spam"; $env:NAME="ko-spam-full"; $env:EPOCHS="2"; python train_text.py
  # → artifacts\ko-spam-full\ 에 3종 세트 생성

  # (2) NER 가중치 — KLUE 데이터 자동 다운로드 후 학습 (~십수 분~시간, 인터넷 필요. CPU면 오래 걸림)
  cd ..\..\rnd-detection-models\export
  $env:NER_SUBSET="6000"; $env:EPOCHS="3"; python finetune_ner.py     # → ner_klue.pt 생성
  cd ..\..
  ```
  - 두 명령은 각각 `가중치 저장`류 메시지가 나오고 파일이 생기면 성공.
  - 급하면 스모크로 더 빠르게(정확도는 낮음): 스팸 `$env:EPOCHS="1"`, NER `$env:NER_SUBSET="200"; $env:EPOCHS="1"`.
  - **가중치 없이도 PII·이미지 분석·이미지 검색 탭은 즉시 동작** — 먼저 둘러본 뒤 위 학습을 돌려도 됩니다.
- KoCLIP·YOLO 기본 모델은 두 배포본 모두 첫 실행 시 자동 다운로드.
- OCR(EasyOCR) 모델도 첫 실행 시 자동 다운로드. 비교 엔진 PaddleOCR은 **격리 venv**(의존성 충돌 방지) — `3_사용법.md` 7) 참조 (Windows는 `py -3.12 -m venv %USERPROFILE%\ocr-env`).

## 3. 통합 데모 실행

```powershell
cd rnd-uxui-demo\export
python app.py                            # → 브라우저 http://127.0.0.1:7860
$env:YOLO_PT="<best.pt 경로>"; python app.py   # 직접 학습한 YOLO 모델로 교체
```
탭 4개: 텍스트 분석 / 검수·라벨링 / 이미지 분석 / 이미지 검색·일괄.
개별 모듈 실행법은 `3_사용법.md` (환경변수는 PowerShell `$env:` 형식으로).

## 4. 폴더 지도 & 문서 읽는 순서

```
탐지계층-R&D-보고\
├─ README_mac.md / README_windows.md   ← 시작점 (OS별)
├─ 6_보고서.md / .pdf                    ← 공식 보고서 (수행기간·결과·기대효과)
├─ 5_도식도.md / .pdf                    ← 아키텍처·데이터 순환·처리 흐름 그림
├─ 1_연구문서.md   2_소스코드.md   3_사용법.md   4_가이드.md
├─ requirements.txt                      ← 전체 의존성 (한 파일)
└─ rnd-*\                                ← 모듈 코드 (dataset-artifacts·rule-vs-bert·clip·uxui-demo·detection-models·ocr)
```
- **급하면**: `6_보고서` + `5_도식도` 두 개면 충분 (둘 다 PDF 있음 — 뷰어 없이 열람·인쇄 가능).
- **더 깊이**: `1_연구문서` → `4_가이드` → `3_사용법` → `2_소스코드`.

## 5. 문제 해결 (Windows)

| 증상 | 원인·조치 |
|---|---|
| `Activate.ps1` 실행 차단 | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `No matching distribution found for numpy==2.5.1` | Python이 3.12 미만 — venv를 3.12로 다시 생성 |
| `No module named 'ko_pii'` 등 | venv 활성화 후 `pip install -r requirements.txt` 재확인 |
| KLUE 로드 실패 (`Invalid HF URI 'klue'`) | `datasets`가 5.0.0인지 확인 (`pip show datasets`) — 구버전 미지원 |
| GPU가 안 잡힘 | 기본 torch는 CPU 빌드 — §1의 CUDA 설치 명령(`--index-url .../cu124`) 사용 |
| 스팸·NER 탭 "가중치 없음" | 가벼운 배포본 — 가중치 포함본을 쓰거나 §2대로 생성 (PII·이미지 탭은 동작) |
| 데모 포트 충돌(7860) | `app.py` 마지막 줄 `server_port` 변경 |
| 다운로드 실패 | 방화벽·프록시에서 huggingface.co, github.com 허용 |
