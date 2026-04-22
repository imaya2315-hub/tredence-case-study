import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.gate_scores, mean=0.0, std=0.1)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_sparsity_loss(self):
        l1_loss = 0.0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                l1_loss += torch.sum(gates)
        return l1_loss

    def get_sparsity_level(self, threshold=1e-2):
        total_weights = 0
        pruned_weights = 0
        
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, PrunableLinear):
                    gates = torch.sigmoid(module.gate_scores)
                    total_weights += gates.numel()
                    pruned_weights += torch.sum(gates < threshold).item()
                    
        return (pruned_weights / total_weights) * 100 if total_weights > 0 else 0

    def get_all_gate_values(self):
        all_gates = []
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, PrunableLinear):
                    gates = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()
                    all_gates.append(gates)
        return np.concatenate(all_gates)


def train_and_evaluate(lambda_val, train_loader, test_loader, device, epochs=10):
    print(f"\n--- Starting Experiment with Lambda = {lambda_val} ---")
    model = SelfPruningNet().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            classification_loss = criterion(outputs, labels)
            sparsity_loss = model.get_sparsity_loss()
            total_loss = classification_loss + (lambda_val * sparsity_loss)
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | "
              f"Current Sparsity: {model.get_sparsity_level():.2f}%")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    final_sparsity = model.get_sparsity_level()
    
    print(f"Result for Lambda {lambda_val}: Test Acc = {test_accuracy:.2f}%, Sparsity = {final_sparsity:.2f}%")
    return model, test_accuracy, final_sparsity

def plot_gate_distribution(model, lambda_val):
    gates = model.get_all_gate_values()
    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Gate Value Distribution ($\lambda$ = {lambda_val})')
    plt.xlabel('Gate Value (0 = Pruned, 1 = Active)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'gate_distribution_lambda_{lambda_val}.png')
    plt.close()
    print(f"Saved distribution plot to 'gate_distribution_lambda_{lambda_val}.png'")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

    lambda_values = [0.00001, 0.00005, 0.0001,0.0005]
    results = []
    best_model = None
    best_lambda = None

    for l_val in lambda_values:
        trained_model, acc, sparsity = train_and_evaluate(l_val, train_loader, test_loader, device, epochs=10)
        results.append((l_val, acc, sparsity))
        
        if best_model is None or l_val == 0.001: 
            best_model = trained_model
            best_lambda = l_val

    plot_gate_distribution(best_model, best_lambda)

    print("\n" + "="*40)
    print("FINAL SUMMARY TABLE")
    print("="*40)
    print(f"{'Lambda':<10} | {'Test Accuracy (%)':<20} | {'Sparsity Level (%)':<20}")
    print("-" * 55)
    for l_val, acc, sparsity in results:
        print(f"{l_val:<10} | {acc:<20.2f} | {sparsity:<20.2f}")
    print("="*40)