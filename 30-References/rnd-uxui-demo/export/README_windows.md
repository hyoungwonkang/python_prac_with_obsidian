# README (Windows) — 탐지 AI 통합 데모 실행

## 1. 선행 설치

1. **공용 환경 먼저**: `..\..\rnd-dataset-artifacts\export\README_windows.md`의 §0~§1 그대로
   (같은 rnd-env를 공유 — NVIDIA GPU면 CUDA 명령 포함).
2. 데모 추가 의존성:
   ```powershell
   pip install -r requirements.txt      # gradio (버전 고정)
   ```
3. **NER 가중치**: `..\..\rnd-detection-models\export\ner_klue.pt`가 있어야 텍스트 탭의 NER이 동작
   (없으면 그 폴더의 README_windows.md §2로 생성 — finetune_ner.py 완주 필요).

## 2. 실행

```powershell
python app.py
# → 브라우저에서 http://127.0.0.1:7860
```

- 모델은 **탭별 첫 분석 때 로딩** — 첫 응답만 수십 초(CPU 기준), 이후 빠름.
- 탭 4개: 텍스트 분석 / 검수·라벨링 / 이미지 분석 / 이미지 검색·일괄.
- GPU 없으면 CPU로 자동 동작 (KoCLIP·BERT 추론은 CPU로도 문장·이미지 단위 수 초 내).

## 3. 문제 해결

| 증상 | 조치 |
|---|---|
| 7860 포트 사용 중 | app.py 마지막 줄 `server_port` 변경 |
| NER 칸 오류 | ner_klue.pt 존재 확인 (§1-3) |
| 첫 분석이 매우 느림 | 모델 다운로드·로딩 중 — 콘솔 진행률 확인, 1회성 |
