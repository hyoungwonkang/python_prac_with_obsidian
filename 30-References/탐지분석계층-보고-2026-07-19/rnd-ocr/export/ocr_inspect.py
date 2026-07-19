"""
EasyOCR 인스펙터(inspector) — 이미지 → (텍스트, 위치 박스, 신뢰도) 목록.

OCR = 검출(글자가 어디에) + 인식(그 글자가 무엇인지) 2단 파이프라인.
EasyOCR 내부: 검출=CRAFT, 인식=CRNN (둘 다 PyTorch 사전학습 모델 — 우리 쪽 학습 0줄, 추론만).
첫 실행 시 ~/.EasyOCR/ 에 모델 자동 다운로드(검출+한국어 인식, 약 150MB).

파라미터(환경변수): IMAGE(기본 images/clean.png) · GPU(기본 0 — CPU. MPS 조용한 오답 전례로 기본 보수적)
실행:  python ocr_inspect.py            또는   IMAGE=images/noisy.png python ocr_inspect.py
"""
import os
import time
from pathlib import Path

import easyocr

HERE = Path(__file__).resolve().parent
IMAGE = HERE / os.environ.get("IMAGE", "images/clean.png")
GPU = os.environ.get("GPU", "0") == "1"


def main():
    t0 = time.time()
    reader = easyocr.Reader(["ko", "en"], gpu=GPU)      # 언어 목록 = 인식 모델 선택
    t_load = time.time() - t0

    t0 = time.time()
    results = reader.readtext(str(IMAGE))               # → [(박스 4점, 텍스트, 신뢰도), ...]
    t_ocr = time.time() - t0

    print(f"이미지: {IMAGE.name}  (모델 로드 {t_load:.1f}초 / 추론 {t_ocr:.1f}초, {'GPU' if GPU else 'CPU'})")
    print("-" * 60)
    for box, text, conf in results:
        x, y = int(box[0][0]), int(box[0][1])           # 박스 좌상단만 표시
        print(f"  ({x:4d},{y:4d})  {conf:.3f}  {text!r}")
    print("-" * 60)
    print("읽은 전체 문장:", " ".join(t for _, t, _ in results))


if __name__ == "__main__":
    main()
