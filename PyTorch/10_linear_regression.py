import torch

# Training data: y = 2x + 1
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# Weight and bias are learnable parameters
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

learning_rate = 0.01

for epoch in range(1000):
    # Forward pass
    prediction = w * x + b

    # Calculate loss
    loss = torch.mean((prediction - y) ** 2)

    # Calculate gradients
    loss.backward()

    # Update parameters without tracking these updates in autograd
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Clear old gradients
    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

print("Learned weight:", w.item())
print("Learned bias:", b.item())
