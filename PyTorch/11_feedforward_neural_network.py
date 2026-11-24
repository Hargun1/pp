import torch
import torch.nn as nn
import torch.optim as optim

# 100 random samples, each with 4 input features
X = torch.randn(100, 4)

# Random class labels: 0 or 1
y = torch.randint(0, 2, (100,))

# Feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleNN()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    # Forward pass
    output = model(X)
    loss = loss_function(output, y)

    # Backpropagation and parameter update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

print("Training completed.")
print("Output shape:", model(X).shape)
