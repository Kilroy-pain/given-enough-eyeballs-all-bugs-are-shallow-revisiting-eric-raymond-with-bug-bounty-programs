import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class BugBountyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BugBountyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def simulate_bug_discovery(num_bugs, decay_rate, num_rounds):
    """
    Simulates the discovery of bugs over time using a decay model.
    :param num_bugs: Total number of bugs in the system.
    :param decay_rate: Rate at which the probability of finding bugs decreases.
    :param num_rounds: Number of rounds of bug discovery.
    :return: A list of discovered bugs per round.
    """
    discovered_bugs = []
    remaining_bugs = num_bugs

    for _ in range(num_rounds):
        # Probability of finding a bug decreases exponentially
        prob_find_bug = np.exp(-decay_rate * (num_bugs - remaining_bugs))
        found_bugs = np.random.binomial(remaining_bugs, prob_find_bug)
        discovered_bugs.append(found_bugs)
        remaining_bugs -= found_bugs

    return discovered_bugs

def train_bug_bounty_model(model, data, labels, epochs=100, learning_rate=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

if __name__ == '__main__':
    # Simulate bug discovery dynamics
    num_bugs = 100
    decay_rate = 0.1
    num_rounds = 20
    discovered_bugs = simulate_bug_discovery(num_bugs, decay_rate, num_rounds)
    print("Discovered bugs per round:", discovered_bugs)

    # Prepare dummy data for training the bug bounty model
    input_dim = 10
    hidden_dim = 20
    output_dim = 1
    num_samples = 100

    np.random.seed(42)
    torch.manual_seed(42)

    # Generate random data and labels
    data = torch.tensor(np.random.rand(num_samples, input_dim), dtype=torch.float32)
    labels = torch.tensor(np.random.randint(0, 2, size=(num_samples, output_dim)), dtype=torch.float32)

    # Initialize and train the model
    model = BugBountyModel(input_dim, hidden_dim, output_dim)
    train_bug_bounty_model(model, data, labels, epochs=50, learning_rate=0.01)

    # Test the model with a sample input
    test_input = torch.tensor(np.random.rand(1, input_dim), dtype=torch.float32)
    model.eval()
    prediction = model(test_input).item()
    print("Test input prediction:", prediction)