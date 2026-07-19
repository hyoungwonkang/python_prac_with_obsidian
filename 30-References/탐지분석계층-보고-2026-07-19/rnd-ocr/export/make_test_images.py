"""
OCR 테스트 이미지 생성기 — 정답(ground truth)을 아는 한국어 문장을 이미지로 렌더링.

깨끗한 판 + 열화 4종(회전·노이즈·저대비·축소)을 만들어
"전처리(OpenCV) 유무로 인식률이 얼마나 달라지나" 실험의 재료로 쓴다.
이미지는 재생성 가능 산출물 — git에는 이 스크립트만 추적.

산출: images/ 에 clean.png + rotated/noisy/lowcontrast/small.png + ground_truth.txt
파라미터(환경변수): OUT(기본 images) · TEXT(기본 스팸+PII 성격 2줄) · FONT(맥 기본 AppleGothic)
실행:  python make_test_images.py
"""
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get("OUT", "images")
# 미니 파이프라인(OCR→분류·NER·PII) 검증까지 겨냥한 문장 — 스팸 어휘 + 인명 + 전화번호
TEXT = os.environ.get("TEXT", "무료 상품권 당첨을 축하합니다\n홍길동 010-1234-5678 연락 바랍니다")
FONT = os.environ.get("FONT", "/System/Library/Fonts/Supplemental/AppleGothic.ttf")  # 윈도우: C:/Windows/Fonts/malgun.ttf


def render_clean(text: str, font_path: str) -> np.ndarray:
    """문장을 흰 바탕 검정 글씨 이미지로 렌더링 (PIL) → OpenCV 배열(BGR)."""
    font = ImageFont.truetype(font_path, 44)
    lines = text.split("\n")
    pad = 40
    w = max(int(font.getlength(l)) for l in lines) + pad * 2
    h = (44 + 18) * len(lines) + pad * 2
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * (44 + 18)), line, font=font, fill="black")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def degrade(clean: np.ndarray) -> dict[str, np.ndarray]:
    """실무에서 만나는 저품질 4종 — 각각 OpenCV 전처리로 복구를 노릴 수 있는 유형."""
    h, w = clean.shape[:2]
    out = {}
    # ① 회전 8도 (스캔 비뚤어짐) → 복구: 기울기 보정(deskew)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.0)
    out["rotated"] = cv2.warpAffine(clean, m, (w, h), borderValue=(255, 255, 255))
    # ② 가우시안 노이즈 (저조도 촬영) → 복구: 블러/디노이즈
    noise = np.random.default_rng(123).normal(0, 35, clean.shape)  # seed 고정 = 재현성
    out["noisy"] = np.clip(clean.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    # ③ 저대비 (흐린 인쇄·팩스) → 복구: 이진화/히스토그램 평활화
    out["lowcontrast"] = np.clip(clean.astype(np.float64) * 0.25 + 165, 0, 255).astype(np.uint8)
    # ④ 축소 40% (저해상도 썸네일) → 복구: 확대 리사이즈
    out["small"] = cv2.resize(clean, (int(w * 0.4), int(h * 0.4)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clean = render_clean(TEXT, FONT)
    cv2.imwrite(str(OUT / "clean.png"), clean)
    for name, img in degrade(clean).items():
        cv2.imwrite(str(OUT / f"{name}.png"), img)
    (OUT / "ground_truth.txt").write_text(TEXT + "\n", encoding="utf-8")
    print(f"생성 완료 → {OUT}/  (clean + 열화 4종 + ground_truth.txt)")
    print(f"정답 문장: {TEXT!r}")


if __name__ == "__main__":
    main()
