"""ch01_basics — PyTorch 환경 검증과 텐서 기초.

로컬(인텔 맥, PyTorch 2.2.2/MPS)과 Colab(2.6.0/T4 CUDA) 양쪽에서
동일하게 동작하도록 작성. 디바이스는 자동 선택.
"""

import torch


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    device = select_device()

    print(f"PyTorch 버전 : {torch.__version__}")
    print(f"선택된 디바이스: {device}")

    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)
    print(f"x + y       : {x + y}")


if __name__ == "__main__":
    main()
