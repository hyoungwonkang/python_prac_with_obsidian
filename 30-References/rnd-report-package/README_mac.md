# README (macOS) — 탐지 계층 R&D 통합 보고 패키지

> **Windows는 `README_windows.md` 참조.** 클린 macOS 기준(Apple Silicon 검증 — M4 Max, GPU=MPS 자동 사용).
> 명령은 **터미널**에서 이 폴더(`탐지계층-R&D-보고/`)로 이동 후 실행.

범용 **1차 탐지 계층**(텍스트 분류·PII·NER·이미지 탐지·상황 판단)을 공개 데이터로 로컬에서
end-to-end 재현한 R&D 묶음입니다. 특정 도메인이 아니라 **도메인 교체 가능한 범용 계층**이 목적이며,
적용처는 아동·청소년 성착취물 대응, 불법스포츠도박 등 복수에 걸칩니다.

## 산출물 한눈에

| # | 내용 | 핵심 결과 | 폴더 |
|---|---|---|---|
| 1 | 학습 데이터 산출물(확장성) | 규약만 맞추면 학습→산출물→재사용이 반복되는 환경 | `rnd-dataset-artifacts` |
| 2 | 분류 잘하는 법 | Rule vs BERT vs 하이브리드 — 단일 승자 없음, 업무별 선택지 | `rnd-rule-vs-bert` |
| 3 | 통합 UXUI | 모듈 5개를 탭 4개 데모로 통합 (신규 학습 0) | `rnd-uxui-demo` |
| 4 | CLIP 상황 판단 | KoCLIP+한국어 프롬프트 — 재학습 없이 분류 체계 교체 | `rnd-clip` |
| 5 | YOLO 직접 라벨링 | 라벨링→학습→탐지 실증 (mAP50 0.995) | `rnd-dataset-artifacts` |

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
- 최소 Python 3.10 (ko-pii 요건). 첫 실행 시 모델 자동 다운로드(klue/bert-base·KoCLIP 등 약 1.5GB).
- 장치는 코드가 자동 감지 — 맥은 MPS(Apple GPU) 사용.

## 2. 가중치 (기본 포함 — 별도 작업 불필요)

데모에 필요한 가중치(스팸 분류 3종 세트·NER·YOLO 기본)는 **패키지에 포함**되어 바로 실행됩니다.
- 가벼운 배포본(가중치 미포함)을 받았다면 스팸·NER 탭은 "가중치 없음" 안내가 뜨고 **PII·이미지 탭은 그대로 동작**.
  직접 만들려면 각 폴더의 `train_text.py`·`finetune_ner.py`(3_사용법 참조).
- KoCLIP·YOLO 기본 모델은 첫 실행 시 자동 다운로드.

## 3. 통합 데모 실행

```bash
cd rnd-uxui-demo/export
python app.py                            # → 브라우저 http://127.0.0.1:7860
YOLO_PT=<best.pt 경로> python app.py       # 직접 학습한 YOLO 모델로 교체
```
탭 4개: 텍스트 분석 / 검수·라벨링 / 이미지 분석 / 이미지 검색·일괄.
개별 모듈 실행법은 `3_사용법.md`.

## 4. 폴더 지도 & 문서 읽는 순서

```
탐지계층-R&D-보고/
├─ README_mac.md / README_windows.md   ← 시작점 (OS별)
├─ 6_보고서.md / .pdf                    ← 공식 보고서 (수행기간·결과·기대효과)
├─ 5_도식도.md / .pdf                    ← 아키텍처·데이터 순환·처리 흐름 그림
├─ 1_연구문서.md   2_소스코드.md   3_사용법.md   4_가이드.md
├─ requirements.txt                      ← 전체 의존성 (한 파일)
└─ rnd-*/                                ← 모듈 코드 (dataset-artifacts·rule-vs-bert·clip·uxui-demo·detection-models)
```
- **급하면**: `6_보고서` + `5_도식도` 두 개면 충분 (둘 다 PDF 있음 — 뷰어 없이 열람·인쇄 가능).
- **더 깊이**: `1_연구문서` → `4_가이드` → `3_사용법` → `2_소스코드`.

## 5. 문제 해결 (macOS)

| 증상 | 원인·조치 |
|---|---|
| `No module named 'ko_pii'` 등 | venv 활성화 후 `pip install -r requirements.txt` 재확인. Python 3.10 미만이면 ko-pii 설치 실패 |
| KLUE 로드 실패 (`Invalid HF URI 'klue'`) | `datasets`가 5.0.0인지 확인 (`pip show datasets`) — 구버전 미지원 |
| 스팸·NER 탭 "가중치 없음" | 가벼운 배포본 — 가중치 포함본을 쓰거나 §2대로 생성 (PII·이미지 탭은 동작) |
| Apple Silicon 연산 오류 | 코드에 `PYTORCH_ENABLE_MPS_FALLBACK=1` 내장. 결과가 이상하면 소량을 CPU와 비교 (MPS 조용한 오답 사례 있음) |
| 데모 포트 충돌(7860) | `app.py` 마지막 줄 `server_port` 변경 |
| 다운로드 실패 | 네트워크에서 huggingface.co, github.com 허용 확인 |
