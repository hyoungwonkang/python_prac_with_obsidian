import torch
import torch.nn.functional as F
from neural import NeuralNetwork
from dataloader import train_loader, X_train, y_train, test_loader

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.5
)

num_epochs = 3
for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
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
        with torch.no_grad():
            outputs = model(X_train)
        print(outputs)

        torch.set_printoptions(sci_mode=False)
        probas = torch.softmax(outputs, dim=1)
        print(probas)

predictions = torch.argmax(probas, dim=1)
print(predictions == y_train)
print(torch.sum(predictions == y_train))

def compute_accuracy(model, dataloader):

    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()

print(compute_accuracy(model, train_loader))
print(compute_accuracy(model, test_loader))