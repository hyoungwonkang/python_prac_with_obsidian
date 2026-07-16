"""
OCR 앙상블 실험 — 두 엔진 교차 검증: "합의는 자동 채택, 불일치만 검수".

오답 프로필이 상이(Easy=모양 혼동 / Paddle=끝 글자 절단)하다는 관찰의 활용 실험.
같은 이미지를 두 엔진으로 읽어 단어 단위로 대조:
  합의(같은 단어)   → 자동 채택 (사람 안 봄)
  불일치(다른 단어) → 검수 큐 (사람이 정답으로 교정한다고 가정)
정답 대조로 세 가지를 채점:
  검수 부담 = 불일치 단어 비율 (사람 작업량)
  맹점     = 합의했는데 둘 다 틀린 단어 (이 정책이 놓치는 것 — 조용한 실패 후보)
  정책 정확도 = 합의 채택 + 불일치 인간 교정 후의 단어 정확도

실행:  ~/rnd-env/bin/python ocr_ensemble.py       (IMAGES=images 기본, images_pre 가능)
"""
import difflib
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = HERE / os.environ.get("IMAGES", "images")
OUT = HERE / "out"
PY = {"easy": Path.home() / "rnd-env/bin/python",
      "paddle": Path.home() / "ocr-env/bin/python"}


def ocr_all(engine: str) -> dict[str, list[str]]:
    """엔진별 전용 환경 서브프로세스 1회로 폴더 전체 읽기 → {이미지: 단어 목록}."""
    code = (f"import pathlib, ocr_eval\n"
            f"read = ocr_eval.make_reader()\n"
            f"for p in sorted(pathlib.Path(r'{IMAGES}').glob('*.png')):\n"
            f"    t = ' '.join(read(p).split())\n"
            f"    (pathlib.Path(r'{OUT}') / (p.stem + '.{engine}.txt')).write_text(t, encoding='utf-8')\n")
    r = subprocess.run([str(PY[engine]), "-c", code], cwd=HERE,
                       capture_output=True, text=True,
                       env={**os.environ, "ENGINE": engine})
    if r.returncode != 0:
        raise SystemExit(f"{engine} OCR 실패:\n{r.stderr[-800:]}")
    return {p.stem: (OUT / f"{p.stem}.{engine}.txt").read_text(encoding="utf-8").split()
            for p in sorted(IMAGES.glob("*.png"))}


def gold_align(words: list[str], gold: list[str]) -> set[int]:
    """words에서 gold와 정렬상 일치하는 인덱스 집합 (그 단어가 '정답'인지 판정용)."""
    ok = set()
    for op, a1, a2, b1, b2 in difflib.SequenceMatcher(a=words, b=gold).get_opcodes():
        if op == "equal":
            ok.update(range(a1, a2))
    return ok


def main():
    OUT.mkdir(exist_ok=True)
    gold = (IMAGES / "ground_truth.txt").read_text(encoding="utf-8").split()

    print(f"세트: {IMAGES.name} | 정답 {len(gold)}단어 | 정책: 합의=자동 채택, 불일치=검수(교정)")
    easy_all, paddle_all = ocr_all("easy"), ocr_all("paddle")

    print(f"\n{'이미지':<13} {'합의':>4} {'검수부담':>8} {'맹점':>4}  {'정책 결과'}")
    print("-" * 64)
    tot_agree = tot_review = tot_blind = tot_words = 0
    for stem in easy_all:
        e, p = easy_all[stem], paddle_all[stem]
        e_ok = gold_align(e, gold)                    # easy 단어 중 정답인 인덱스

        agree_idx, agree_correct = [], 0
        for op, a1, a2, b1, b2 in difflib.SequenceMatcher(a=e, b=p).get_opcodes():
            if op == "equal":
                agree_idx.extend(range(a1, a2))
        blind = [e[i] for i in agree_idx if i not in e_ok]   # 합의했는데 정답 아님
        n_agree, n_blind = len(agree_idx), len(blind)
        n_review = max(len(e), len(p)) - n_agree             # 불일치(검수 대상) 근사
        # 정책 결과: 합의 채택(맹점만 오답) + 불일치는 사람이 정답으로 교정
        acc = (len(gold) - n_blind) / len(gold)
        print(f"{stem:<13} {n_agree:>4} {n_review:>6}개 {n_blind:>4}  "
              f"단어 정확도 {acc:.0%}" + (f"  ⚠️ 맹점: {blind}" if blind else ""))
        tot_agree += n_agree; tot_review += n_review
        tot_blind += n_blind; tot_words += max(len(e), len(p))

    print("-" * 64)
    print(f"합계: 합의 {tot_agree}/{tot_words} ({tot_agree/tot_words:.0%} 자동 통과) | "
          f"검수 부담 {tot_review}개 ({tot_review/tot_words:.0%}) | 맹점 {tot_blind}개")


if __name__ == "__main__":
    main()
