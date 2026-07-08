"""
OpenCV 영상 프레임 추출 — 동영상 증적을 이미지 분석 입력으로 분해.

플랫폼 아키텍처에서 수집된 동영상(채증 영상)을 프레임 단위 이미지로 쪼개
YOLO/이미지 분석에 넣는 전처리. 파인튜닝 아님.

입력 영상:
  VIDEO 환경변수로 지정. 없으면 합성 영상(움직이는 도형)을 생성해서 그걸 분해 —
  외부 파일 없이도 프레임 추출 로직을 end-to-end 검증.

파라미터(환경변수):
  VIDEO   : 입력 mp4/avi 경로 (없으면 합성)
  EVERY   : N프레임마다 1장 저장 (기본 5)

실행:
  ~/rnd-env/bin/python opencv_frames.py
  VIDEO=/path/clip.mp4 EVERY=10 ~/rnd-env/bin/python opencv_frames.py
산출: opencv_out/frames/ 에 frame_XXXX.png
"""
import os
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "opencv_out" / "frames"
SYNTH = HERE / "opencv_out" / "_synth.mp4"
EVERY = int(os.environ.get("EVERY", "5"))


def make_synth_video(path, n_frames=60, size=(640, 360), fps=30):
    """움직이는 흰 원 — 프레임마다 위치가 바뀌는 합성 영상."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    w, h = size
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)
        x = int(w * (i + 1) / (n_frames + 1))
        cv2.circle(frame, (x, h // 2), 30, (255, 255, 255), -1)
        cv2.putText(frame, f"frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        writer.write(frame)
    writer.release()
    return n_frames


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    video = os.environ.get("VIDEO")
    if video and Path(video).exists():
        origin = f"VIDEO={video}"
    else:
        n = make_synth_video(SYNTH)
        video = str(SYNTH)
        origin = f"합성 영상({n}프레임)"

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"영상 열기 실패: {video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"입력: {origin} | 총 {total}프레임 | {fps:.1f}fps | {EVERY}프레임마다 저장")

    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % EVERY == 0:
            cv2.imwrite(str(OUT / f"frame_{idx:04d}.png"), frame)
            saved += 1
        idx += 1
    cap.release()

    print(f"읽은 프레임: {idx} | 저장: {saved}장 → {OUT}")


if __name__ == "__main__":
    main()
