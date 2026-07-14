"""
OCR 채점기 — 이미지들을 OCR로 읽고 정답(ground_truth.txt)과 대조해 CER을 계산.

CER(Character Error Rate, 문자 오류율) = 편집거리(틀린 글자 수) ÷ 정답 글자 수.
낮을수록 좋다 (0 = 완벽). "몇 곳이 어떻게 틀렸나"는 difflib로 구간 표시.
눈 대조가 아니라 코드로 채점 — 엔진·이미지가 늘어도 같은 잣대 유지 (동일 잣대 비교 원칙).

파라미터(환경변수): ENGINE(easy|paddle, 기본 easy) · IMAGES(기본 images — 그 안의 *.png 전부)
실행:  python ocr_eval.py            또는   ENGINE=paddle python ocr_eval.py
"""
import difflib
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = os.environ.get("ENGINE", "easy")
IMAGES = HERE / os.environ.get("IMAGES", "images")


def edit_distance(a: str, b: str) -> int:
    """레벤슈타인 편집거리 — a를 b로 만드는 최소 수정(치환·삽입·삭제) 횟수."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,                      # 삭제
                           cur[j - 1] + 1,                   # 삽입
                           prev[j - 1] + (ca != cb)))        # 치환(같으면 0)
        prev = cur
    return prev[-1]


def show_diff(gold: str, pred: str) -> list[str]:
    """정답↔읽음이 다른 구간만 뽑아 '정답→읽음' 형태로 나열 (오독 위치 확인)."""
    out = []
    sm = difflib.SequenceMatcher(a=gold, b=pred)
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op != "equal":
            out.append(f"{gold[a1:a2] or '∅'}→{pred[b1:b2] or '∅'}")
    return out


def make_reader():
    """엔진별 (로드함수, 읽기함수) — 같은 인터페이스로 감싸 채점부는 엔진을 모름."""
    if ENGINE == "easy":
        import easyocr
        reader = easyocr.Reader(["ko", "en"], gpu=False)
        return lambda p: " ".join(t for _, t, _ in reader.readtext(str(p)))
    if ENGINE == "paddle":
        from paddleocr import PaddleOCR                       # 미설치면 여기서 ImportError
        reader = PaddleOCR(lang="korean")
        def read(p):
            res = reader.predict(str(p))
            return " ".join(t for page in res for t in page["rec_texts"])
        return read
    raise SystemExit(f"모르는 ENGINE: {ENGINE} (easy|paddle)")


def main():
    gold = " ".join((IMAGES / "ground_truth.txt").read_text(encoding="utf-8").split())
    read = make_reader()

    print(f"엔진: {ENGINE} | 정답 {len(gold)}자: {gold!r}")
    print(f"{'이미지':<14} {'CER':>6}  {'오독(정답→읽음)'}")
    print("-" * 72)
    for img in sorted(IMAGES.glob("*.png")):
        t0 = time.time()
        pred = " ".join(read(img).split())
        cer = edit_distance(gold, pred) / len(gold)
        diffs = show_diff(gold, pred)
        print(f"{img.stem:<14} {cer:6.1%}  {len(diffs)}곳: {', '.join(diffs) or '—'}"
              f"  ({time.time()-t0:.1f}초)")


if __name__ == "__main__":
    main()
