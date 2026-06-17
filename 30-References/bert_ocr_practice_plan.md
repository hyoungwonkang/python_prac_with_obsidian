# BERT · OCR 실습 학습 플랜 (PyTorch 기반)

> PyTorch로 BERT와 OCR을 대표 예제로 학습하기 위한 로드맵

이 문서는 "BERT 개념 + PyTorch 기초 + OCR 개념(PaddlePaddle 존재까지 인지)"을 갖춘 상태에서, **PyTorch만으로** BERT와 OCR을 실습 예제로 완주하기 위한 학습 플랜입니다.

## 0. 진입 시점 — LLM 트랙과의 관계 (정본 결정)

이 플랜은 메인 트랙 [[../10-Projects/llm-from-scratch]](Raschka 교재 1~7장, 밑바닥부터 직접 구현)와 **별도 트랙**이며, **병행하지 않고 순차 진행**한다. 두 트랙은 학습 방식이 정반대다 — 교재는 원리를 손으로 구현, 이 플랜은 고수준 라이브러리(HF Transformers·EasyOCR·docTR)로 결과부터 도출. 동시에 붙들면 맥락 전환 비용·환경 부담만 커진다.

| 항목 | 결정 | 이유 |
|---|---|---|
| **BERT 실습 진입 시점** | 교재 **6장(분류 미세튜닝) 완료 직후** | GPT 분류 head를 직접 짜본 뒤 "같은 분류 FT를 BERT+HF로는 이렇게"를 비교 → 이 플랜이 복습+확장이 됨. 거꾸로(지금 진입)면 토크나이저·`[CLS]`·트랜스퍼러닝을 원리 없이 API로만 흡수해 교재 목표와 상충. |
| **OCR 실습 우선순위** | **최하 — 교재 7장 완주 후 또는 별도 여유 시간** | OCR(CRAFT/CRNN·비전)은 LLM 트랙과 개념 교집합이 거의 없어 지금 끼우면 순수 추가 부담. |
| **지금(Phase 1) 병행 허용 범위** | **문서 읽고 큰 그림만**(코드 실습 X) | Phase 1도 개념 단계라 "나중에 이렇게 연결된다" 인지 정도는 부담 없음. |

> 한 줄: **교재 트랙을 먼저 진행하고, BERT 실습은 6장 직후 접점, OCR은 그 이후.** 이 결정의 근거 분석은 본 문서 7장(주의점)·메인 노트 [[../10-Projects/llm-from-scratch]] Phase 6 참조.

## 목차

1. [한눈에 보는 결론](#1-한눈에-보는-결론)
2. [라이브러리 선택 가이드](#2-라이브러리-선택-가이드)
3. [1단계 — BERT 실습 (PyTorch 안마당)](#3-1단계--bert-실습-pytorch-안마당)
4. [2단계 — OCR 실습 (EasyOCR → docTR)](#4-2단계--ocr-실습-easyocr--doctr)
5. [3단계 — PaddleOCR은 비교용으로만](#5-3단계--paddleocr은-비교용으로만)
6. [개념 ↔ 코드 매핑 (지난 학습과 연결)](#6-개념--코드-매핑-지난-학습과-연결)
7. [주의점 체크리스트](#7-주의점-체크리스트)

---

## 1. 한눈에 보는 결론

**BERT와 OCR 둘 다 PyTorch로 완주 가능합니다.** 단, 성격이 정반대입니다.

| | BERT 실습 | OCR 실습 |
|---|---|---|
| PyTorch 적합도 | ⭐⭐⭐⭐⭐ 홈그라운드 | ⭐⭐⭐ 됨, 라이브러리 골라야 |
| 대표 도구 | Hugging Face Transformers | docTR / EasyOCR |
| 난이도 | 입문에 최적 | 중급 (파이프라인이 2단계) |
| PaddleOCR 필요? | 무관 | **불필요** — PyTorch 길 따로 있음 |

> 핵심: OCR을 PyTorch로 하려고 PaddleOCR이나 PaddlePaddle을 억지로 끌어올 필요가 **전혀 없습니다.** PyTorch 진영에 OCR 전용 라이브러리(docTR, EasyOCR)가 이미 잘 갖춰져 있습니다. PaddleOCR과 PyTorch OCR은 평행한 선택지이지 갈아타는 관계가 아닙니다.

---

## 2. 라이브러리 선택 가이드

| 목적 | 추천 라이브러리 | 한 줄 이유 |
|---|---|---|
| BERT 파인튜닝 | **Hugging Face Transformers** | PyTorch 표준, 개념이 코드에 1:1 매핑 |
| OCR 빠른 맛보기 | **EasyOCR** | 몇 줄로 이미지 → 텍스트, 입문용 |
| OCR 정식 학습 | **docTR** | PyTorch 공식 생태계 편입, 검출/인식 구조가 명확 |
| (나중에) 비교 학습 | PaddleOCR | Paddle 진영은 어떻게 푸는지 대조용 |

---

## 3. 1단계 — BERT 실습 (PyTorch 안마당)

BERT는 PyTorch가 가장 빛나는 영역입니다. **Hugging Face Transformers**가 사실상 표준이고, 앞서 배운 개념들이 코드로 거의 1:1 매핑됩니다.

```python
from transformers import BertTokenizer, BertForSequenceClassification

# ① 토크나이제이션 (그 토크나이저)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ② 사전학습 모델 + 위에 분류 층 얹기
#    (트랜스퍼 러닝의 "머리 갈아끼우기")
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)
```

여기서 `BertForSequenceClassification`이 바로 **패턴 A — `[CLS]` 벡터에 분류 층 하나 얹기**입니다. 개념과 코드가 같은 그림이라 학습 효과가 큽니다.

**대표 실습 예제: 감성 분석 (영화 리뷰 긍정/부정)**

- 입문자 표준 코스
- 데이터셋: IMDb, SST-2, NSMC(한국어 네이버 영화 리뷰) 등
- 흐름: 토크나이즈 → 사전학습 모델 로드 → 분류 층 파인튜닝 → 평가

**목표:** 토크나이제이션 · `[CLS]` · 트랜스퍼 러닝이 코드로 확인되는 순간을 직접 경험하기.

---

## 4. 2단계 — OCR 실습 (EasyOCR → docTR)

OCR은 BERT보다 한 겹 복잡합니다. **검출(글자 위치 찾기) + 인식(그 글자가 뭔지)** 의 2단계 파이프라인이기 때문입니다.

### ① EasyOCR — 가장 쉬운 입문용

```python
import easyocr

reader = easyocr.Reader(['ko', 'en'])   # 한국어 + 영어
result = reader.readtext('문서.jpg')     # → 글자 + 위치 + 신뢰도
```

- PyTorch로 구현돼 있고, CUDA GPU가 있으면 검출·인식 속도가 크게 빨라짐
- 내부 구조: 검출은 **CRAFT**, 인식은 **CRNN**
  - CRNN = 합성곱층(이미지 특징 추출) + 순환층(프레임별 라벨 예측) + 전사층(최종 글자열 변환)
  - 앞서 배운 "특징 추출" 방식이 바로 이 합성곱층

**목표:** "이미지 → 텍스트"를 눈으로 빠르게 확인(추론 감 잡기).

### ② docTR — 정석 학습용 (추천)

**PyTorch 공식 생태계에 편입된** 라이브러리라 학습 자료로 가장 깔끔합니다. PyTorch 블로그가 직접 통합을 발표했습니다.

```python
from doctr.models import ocr_predictor

# 검출 아키텍처 + 인식 아키텍처를 각각 골라 끼움
# → 2단계 구조가 코드에 그대로 드러남
model = ocr_predictor(
    det_arch="db_resnet50",
    reco_arch="crnn_vgg16_bn",
    pretrained=True
)
```

- 사전학습된 검출·인식 모델 제공
- 애플리케이션이나 데이터셋에 맞춰 **커스텀 모델 훈련**도 지원
- 즉 추론부터 직접 훈련까지 PyTorch로 전부 가능 — PaddleOCR로 하려던 "훈련 + 결과 도출"을 PyTorch에서 동일하게 수행

**목표:** 검출/인식 2단계 구조를 뜯어보고, 여유되면 내 데이터로 파인튜닝까지.

---

## 5. 3단계 — PaddleOCR은 비교용으로만

PyTorch로 OCR을 한 바퀴 돈 뒤에, "같은 OCR을 Paddle 진영은 어떻게 하나"를 비교하면 시야가 넓어집니다.

> 처음부터 PyTorch와 PaddlePaddle 두 프레임워크를 동시에 붙들면 환경 설정 지옥(버전 궁합 문제)에 빠지기 쉽습니다. 한 진영으로 완주한 뒤 비교하는 순서를 권장합니다.

---

## 6. 개념 ↔ 코드 매핑 (지난 학습과 연결)

앞서 정리한 개념이 실습에서 어디에 대응되는지 한눈에 정리합니다.

| 배운 개념 | 실습에서 만나는 지점 |
|---|---|
| 토크나이제이션 | `BertTokenizer.from_pretrained(...)` |
| `[CLS]` 벡터 + 분류 층 (패턴 A) | `BertForSequenceClassification` |
| 트랜스퍼 러닝 (머리만 교체) | `from_pretrained` + `num_labels` 지정 후 파인튜닝 |
| OCR 4단계 중 "특징 추출" | CRNN/CRNN 계열의 합성곱층 |
| OCR 검출 + 인식 2단계 | docTR의 `det_arch` + `reco_arch` |
| 추론 vs 훈련 (툴킷의 두 능력) | EasyOCR `readtext()` (추론) ↔ docTR 커스텀 학습 (훈련) |
| OCR → NLP 공급 흐름 | OCR로 뽑은 텍스트를 BERT 입력으로 연결 |

---

## 7. 주의점 체크리스트

- [ ] **순서 지키기** — BERT(안마당) 먼저, OCR(2단계라 약간 어려움) 나중
- [ ] **한 프레임워크로 완주** — PyTorch로 끝까지 간 뒤 PaddleOCR 비교
- [ ] **GPU 환경 확인** — OCR 특히 CUDA가 있으면 학습/추론 속도 크게 향상
- [ ] **한국어 목표라면** — docTR/EasyOCR의 한글 지원 수준과 데이터 준비 방식이 영어와 다름. 데이터셋(NSMC 등 한국어 코퍼스, 한글 OCR 데이터) 사전 확인
- [ ] **데이터 품질이 핵심** — 훈련 결과의 성패는 라이브러리가 아니라 라벨링 데이터의 양과 품질에 달려 있음 (garbage in, garbage out)

---

## 추천 학습 순서 요약

```
1단계: BERT (Hugging Face Transformers)
        └ 감성 분석 파인튜닝으로 개념을 코드로 확인
                    ↓
2단계: OCR
        ├ EasyOCR로 추론 맛보기 (이미지 → 텍스트)
        └ docTR로 검출/인식 2단계 + 커스텀 훈련
                    ↓
3단계: PaddleOCR을 비교용으로 살펴보기 (선택)
```

> 한 줄 정리: **BERT와 OCR 모두 PyTorch로 완주 가능합니다. BERT는 Hugging Face Transformers로(가장 쉬움), OCR은 docTR 또는 EasyOCR로 하면 되고, PaddleOCR/PaddlePaddle 없이 PyTorch만으로 추론부터 훈련까지 전부 됩니다.**