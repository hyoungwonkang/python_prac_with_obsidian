# 탐지 계층 R&D — 통합 보고 패키지

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

이 패키지의 **진입점은 통합 데모**(`rnd-uxui-demo`)입니다 — 위 모듈들을 한 화면에서 시연합니다.
`rnd-detection-models`(기존 NER·YOLO·PII R&D)는 데모가 재사용하므로 함께 포함됩니다.

## 빠른 시작

### 공통 0단계 — Python 3.12 가상환경 + 의존성

**macOS / Linux**
```bash
python3.12 -m venv rnd-env
source rnd-env/bin/activate
pip install -U pip && pip install -r requirements.txt
```
**Windows (PowerShell)**
```powershell
py -3.12 -m venv rnd-env
.\rnd-env\Scripts\Activate.ps1        # 막히면: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -U pip; pip install -r requirements.txt
# NVIDIA GPU가 있으면 (없으면 CPU 자동):
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu124
```
- 최소 Python 3.10 (ko-pii 요건). 첫 실행 시 모델 자동 다운로드(klue/bert-base·KoCLIP 등 약 1.5GB).
- 장치는 코드가 자동 감지: 맥=MPS / 윈도우 NVIDIA=CUDA / 그 외=CPU.

### 1단계 — 가중치 (기본 포함 — 별도 작업 불필요)

데모에 필요한 가중치(스팸 분류 3종 세트·NER·YOLO 기본)는 **패키지에 포함**되어 있어 바로 실행됩니다.
- 만약 가벼운 배포본(가중치 미포함)을 받았다면, 스팸·NER 탭은 "가중치 없음" 안내가 뜨고
  **PII·이미지 탭은 그대로 동작**합니다. 두 가중치를 직접 만들려면 각 폴더의 `train_text.py`·`finetune_ner.py` 사용(3_사용법 참조).
- KoCLIP·YOLO 기본 모델은 첫 실행 시 자동 다운로드됩니다.

### 2단계 — 통합 데모 실행
```bash
cd rnd-uxui-demo/export
python app.py            # → 브라우저 http://127.0.0.1:7860
# 직접 학습한 YOLO 모델로 바꾸려면: YOLO_PT=<best.pt 경로> python app.py
```
탭 4개: 텍스트 분석 / 검수·라벨링 / 이미지 분석 / 이미지 검색·일괄.

## 폴더 지도

```
탐지계층-R&D-보고/
├─ README.md               ← 지금 이 문서 (시작점)
├─ 6_보고서.md / .pdf        ← 공식 보고서 (수행기간·결과·기대효과 — 큰 그림)
├─ 5_도식도.md / .pdf        ← 아키텍처·데이터 순환·처리 흐름 그림
├─ 1_연구문서.md            ← 무엇을 왜 했고 무엇을 얻었나 (상세)
├─ 2_소스코드.md            ← 어느 파일이 무슨 역할인가
├─ 3_사용법.md              ← 모듈별·통합 실행법 (맥·윈도우)
├─ 4_가이드.md              ← 실무 판단 기준 (분류·프롬프트·PII·YOLO)
├─ requirements.txt         ← 전체 의존성 (한 파일)
├─ rnd-dataset-artifacts/   ├ 학습 데이터 산출물·직접 라벨링
├─ rnd-rule-vs-bert/        ├ 분류 방법 비교
├─ rnd-clip/                ├ CLIP 상황 판단
├─ rnd-uxui-demo/           ├ 통합 데모 (진입점)
└─ rnd-detection-models/    └ 기존 NER·YOLO·PII (데모가 재사용)
```
※ **5·6은 `.pdf`로도 포함** — 마크다운 뷰어 없이 바로 열람·인쇄 가능 (도식은 이미지로 렌더링됨).

## 읽는 순서 (문서)

- **급하면**: `6_보고서`(공식 보고, 큰 그림) + `5_도식도`(그림) 두 개면 충분 — 둘 다 PDF 있음.
- **더 깊이**: `1_연구문서`(상세 결과) → `4_가이드`(실무 판단 기준) → `3_사용법`(직접 실행) → `2_소스코드`(코드 상세).
