"""gradient — autograd로 손실에 대한 파라미터 기울기 구하기.

logistic.py와 같은 forward 그래프지만, 학습 대상(w1, b)에 requires_grad=True를 줘
PyTorch가 backward 그래프를 자동 구축하게 한다. 그 뒤 grad(loss, param)으로
∂loss/∂param을 직접 꺼낸다 — loss.backward() 한 번에 .grad를 채우는 방식과 등가.

retain_graph=True인 이유: 같은 그래프에서 grad()를 두 번 호출하기 때문.
기본은 한 번 backward하면 그래프 메모리가 해제되어 두 번째 호출이 실패한다.
"""
import torch
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = w1 * x1 + b
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print(grad_L_w1)
print(grad_L_b)
