"""logistic — 로지스틱 회귀의 forward 한 스텝.

입력 x1 → 선형결합(z = w1·x1 + b) → 시그모이드(a) → BCE 손실(loss).
gradient.py에서 이 그래프를 이어받아 autograd로 ∂loss/∂w1, ∂loss/∂b를 구한다.
"""
import torch
import torch.nn.functional as F

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2])
b = torch.tensor([0.0])

z = w1 * x1 + b
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)
