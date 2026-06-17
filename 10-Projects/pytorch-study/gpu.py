import torch
import torch.nn.functional as F

print(torch.cuda.is_available())

tensor_1 = torch.tensor([1.,2.,3.])
tensor_2 = torch.tensor([4.,5.,6.])
print(tensor_1 + tensor_2)

tensor_1 = tensor_1.to("cpu")
tensor_2 = tensor_2.to("cuda" if torch.cuda.is_available() else "cpu")
print(tensor_1 + tensor_2)

from neural import NeuralNetwork
from dataloader import train_loader, X_train, y_train, test_loader

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.5
)

num_epochs = 3
for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## 로깅
        print(f"에포크: {epoch+1:03d}/{num_epochs:03d}"
              f" | 배치 {batch_idx:03d}/{len(train_loader):03d}"
              f" | 훈련 손실: {loss:.2f}")
        
        model.eval()