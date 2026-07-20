"""
OpenCV 전처리 — 열화 이미지를 복구해 images_pre/ 에 저장 (OCR 전 단계, 아키텍처 ②).

열화 4종과 복구 기법이 1:1 짝 (make_test_images.py의 degrade와 대응):
  rotated     → 기울기 보정(deskew): 글자 픽셀의 최소 외접 회전 사각형으로 각도 자동 탐지
  noisy       → 디노이즈: 주변 유사 블록 평균(Non-local Means) — 글자 경계는 살리고 노이즈만
  lowcontrast → 이진화: Otsu 자동 문턱으로 흑/백 이분 — 뿌연 회색을 선명한 대비로
  small       → 확대: INTER_CUBIC 보간 ×2.5 — 인식 모델이 기대하는 글자 크기로
clean은 그대로 복사(대조군 — 전처리 파이프라인이 멀쩡한 이미지를 해치지 않는지 확인).

채점은 잣대 재사용:  IMAGES=images_pre python ocr_eval.py   (before는 IMAGES=images)
실행:  python preprocess.py
"""
import shutil
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
SRC = HERE / "images"
DST = HERE / "images_pre"


def deskew(img):
    """기울기 보정 — 글자 픽셀 전체를 감싸는 최소 회전 사각형의 각도로 역회전."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu 이진화(반전): 글자=흰색 픽셀로 만들어 좌표 수집 대상으로
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(bw)
    angle = cv2.minAreaRect(coords)[-1]          # 실측: rotated.png에서 -7.996° 검출
    if angle < -45:
        angle += 90                              # minAreaRect 각도 규약 보정 (짧은 변 기준일 때)
    elif angle > 45:
        angle -= 90
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderValue=(255, 255, 255))


def denoise(img):
    """Non-local Means — 이미지 전체에서 닮은 조각들을 찾아 평균 (점 노이즈 상쇄, 경계 보존)."""
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def binarize(img):
    """Otsu 이진화 — 밝기 분포가 두 무리로 갈리는 지점을 자동으로 찾아 흑/백 이분."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)  # 채점 파이프라인 호환 위해 3채널 복원


def upscale(img):
    """확대 ×2.5 (축소 0.4의 역수) — 3차 보간(INTER_CUBIC)으로 계단 현상 최소화."""
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * 2.5), int(h * 2.5)), interpolation=cv2.INTER_CUBIC)


FIX = {"rotated": deskew, "noisy": denoise, "lowcontrast": binarize, "small": upscale}


def main():
    DST.mkdir(exist_ok=True)
    shutil.copy2(SRC / "ground_truth.txt", DST / "ground_truth.txt")  # 정답은 동일
    for src in sorted(SRC.glob("*.png")):
        img = cv2.imread(str(src))
        assert img is not None, f"로드 실패: {src}"   # imread는 실패해도 예외 없이 None
        fix = FIX.get(src.stem)
        out = fix(img) if fix else img               # clean = 무처리 대조군
        cv2.imwrite(str(DST / src.name), out)
        print(f"{src.stem:<12} → {fix.__name__ if fix else '복사(대조군)'}")
    print(f"\n완료 → {DST}/  이제:  IMAGES=images_pre python ocr_eval.py")


if __name__ == "__main__":
    main()
