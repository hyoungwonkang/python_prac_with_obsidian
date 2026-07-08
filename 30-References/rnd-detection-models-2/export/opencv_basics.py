"""
OpenCV 기본 영상 처리 — 로드·색공간·리사이즈·블러·엣지.

플랫폼 아키텍처 ②추출·전처리 계층의 최소 유틸리티. 파인튜닝(학습)이 아니라
라이브러리 실습 — 수집된 이미지를 1차 탐지(YOLO/이미지 분석)에 넣기 전
전처리하는 기본기를 코드로 검증한다.

입력 이미지:
  SOURCE 환경변수로 지정. 없으면 ultralytics 번들 샘플(bus.jpg)을 자동 사용하고,
  그것도 없으면 합성 이미지를 생성 — 어떤 환경에서도 무조건 실행되도록.

실행:
  ~/rnd-env/bin/python opencv_basics.py
  SOURCE=/path/to/img.jpg ~/rnd-env/bin/python opencv_basics.py
산출: opencv_out/basics/ 에 gray/resized/blur/edges PNG 저장.
"""
import os
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "opencv_out" / "basics"


def load_image():
    """SOURCE → ultralytics 샘플 → 합성 순으로 이미지를 확보한다."""
    src = os.environ.get("SOURCE")
    if src and Path(src).exists():
        img = cv2.imread(src)
        if img is not None:
            return img, f"SOURCE={src}"

    # ultralytics 번들 샘플 (사람·버스 — COCO 클래스 포함)
    try:
        import ultralytics
        sample = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
        if sample.exists():
            img = cv2.imread(str(sample))
            if img is not None:
                return img, f"ultralytics 샘플({sample.name})"
    except Exception:
        pass

    # 최후 폴백: 합성 이미지 (그라디언트 + 도형) — 외부 의존 0
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    xs = np.linspace(0, 255, 640, dtype=np.uint8)
    img[:, :, 0] = xs                      # 파랑 그라디언트
    img[:, :, 2] = xs[::-1]                # 빨강 역그라디언트
    cv2.rectangle(img, (200, 150), (440, 330), (0, 255, 0), -1)
    cv2.circle(img, (320, 240), 60, (255, 255, 255), -1)
    return img, "합성 이미지(폴백)"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img, origin = load_image()
    h, w = img.shape[:2]
    print(f"입력: {origin} | 해상도 {w}x{h} | 채널 {img.shape[2]}")

    # ① 그레이스케일 (연산량↓, 엣지·특징 추출 전처리)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ② 리사이즈 (YOLO 입력은 보통 640 기준 — 가로 640 맞추고 비율 유지)
    scale = 640 / w
    resized = cv2.resize(img, (640, int(h * scale)), interpolation=cv2.INTER_AREA)

    # ③ 가우시안 블러 (노이즈 완화 — 비식별화의 기본 연산)
    blur = cv2.GaussianBlur(img, (25, 25), 0)

    # ④ 캐니 엣지 (윤곽 — 화면 유형/템플릿 분석의 기초 신호)
    edges = cv2.Canny(gray, 100, 200)

    saved = {
        "gray.png": gray,
        "resized.png": resized,
        "blur.png": blur,
        "edges.png": edges,
    }
    for name, arr in saved.items():
        cv2.imwrite(str(OUT / name), arr)

    print(f"리사이즈: {w}x{h} → 640x{int(h*scale)}")
    print(f"엣지 픽셀 수: {int((edges > 0).sum())}")
    print(f"저장 완료 → {OUT} ({len(saved)}개)")


if __name__ == "__main__":
    main()
