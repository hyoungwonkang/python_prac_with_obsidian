import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(

            # 첫 번째 은닉층
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 두 번째 은닉층
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # 출력층
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

torch.manual_seed(123)

model = NeuralNetwork(50, 3)
print(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("훈련 가능한 모델의 총 파라미터 개수:", num_params)

print(model.layers[0].weight)
print(model.layers[0].bias)

X = torch.randn((1, 50))
out = model(X)
print(out)

with torch.no_grad():
    out2 = model(Y := torch.randn((1, 50)))
print(out2)