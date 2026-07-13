# 탐지 계층 R&D — 통합 보고 패키지

범용 **1차 탐지 계층**(텍스트 분류·PII·NER·이미지 탐지·상황 판단)을 공개 데이터로 로컬에서
end-to-end 재현한 R&D 묶음입니다. 특정 도메인이 아니라 **도메인 교체 가능한 범용 계층**이 목적이며,
적용처는 아동·청소년 성착취물 대응, 불법스포츠도박 등 복수에 걸칩니다.

## 산출물 한눈에

| 지시 | 내용 | 핵심 결과 | 폴더 |
|---|---|---|---|
| 1 | 학습 데이터 산출물(확장성) | 규약만 맞추면 학습→산출물→재사용이 반복되는 환경 | `rnd-dataset-artifacts` |
| 2 | 분류 잘하는 법 | Rule vs BERT vs 하이브리드 — 단일 승자 없음, 업무별 선택지 | `rnd-rule-vs-bert` |
| 3 | 통합 UXUI | 모듈 5개를 탭 4개 데모로 통합 (신규 학습 0) | `rnd-uxui-demo` |
| 4 | CLIP 상황 판단 | KoCLIP+한국어 프롬프트 — 재학습 없이 분류 체계 교체 | `rnd-clip` |
| 4′ | YOLO 직접 라벨링 | 라벨링→학습→탐지 실증 (mAP50 0.995) | `rnd-dataset-artifacts` |

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

### 1단계 — 재생성이 필요한 가중치 2개 (첨부 아님, 학습으로 생성)

데모의 스팸 분류·NER은 학습된 가중치가 필요합니다. 각 몇 분이면 생성됩니다:
```bash
# 스팸 분류 가중치 → rnd-dataset-artifacts/export/artifacts/ko-spam-full/
cd rnd-dataset-artifacts/export
DATA=../../rnd-detection-models/export/ko NAME=ko-spam-full EPOCHS=2 python train_text.py   # 없으면 아래 주 참고
cd ../..
# NER 가중치 → rnd-detection-models/export/ner_klue.pt
cd rnd-detection-models/export
NER_SUBSET=200 EPOCHS=1 python finetune_ner.py     # 스모크(빠름). 보고 수치는 NER_SUBSET=6000 EPOCHS=3
cd ../..
```
> 주: 스팸 학습 데이터(ko 3분할)는 공개 데이터라 저장소에 없을 수 있습니다 — 3_사용법 참조.
> 가중치를 함께 받은 경우 이 단계는 건너뜁니다.

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
├─ 1_연구문서.md            ← 무엇을 왜 했고 무엇을 얻었나
├─ 2_소스코드.md            ← 어느 파일이 무슨 역할인가
├─ 3_사용법.md              ← 모듈별·통합 실행법 (맥·윈도우)
├─ 4_가이드.md              ← 실무 판단 기준 (분류·프롬프트·PII·YOLO)
├─ requirements.txt         ← 전체 의존성 (한 파일)
├─ rnd-dataset-artifacts/   ├ 지시 1·4′
├─ rnd-rule-vs-bert/        ├ 지시 2
├─ rnd-clip/                ├ 지시 4
├─ rnd-uxui-demo/           ├ 지시 3 (진입점)
└─ rnd-detection-models/    └ 기존 NER·YOLO·PII (데모가 재사용)
```

## 읽는 순서 (문서)

1. **README**(지금) → 2. **1_연구문서**(결과) → 3. **3_사용법**(직접 돌려보기) →
4. **4_가이드**(판단 기준) → 5. **2_소스코드**(코드 상세). 급하면 README + 4_가이드만으로 충분합니다.
