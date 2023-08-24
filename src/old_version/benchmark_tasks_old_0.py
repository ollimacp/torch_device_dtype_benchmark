import torch
import torch.nn as nn
from sklearn.datasets import make_classification
import time
import matplotlib.pyplot as plt
import os

class BenchmarkTasks:
    def __init__(self, device, dtype, scoreboard = []):
        self.device = device
        self.dtype = dtype
        self.scoreboard = scoreboard


    def binary_classification(self):
        # Generate data
        X, y = make_classification(n_features=20, n_redundant=0, n_classes=2)
        print("DEBUG: dtype:", self.dtype) # Add this line

        X = torch.tensor(X, dtype=self.dtype).to(self.device)
        y = torch.tensor(y, dtype=self.dtype).view(-1, 1).to(self.device)

        class BinaryClassifier(nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.hidden1 = nn.Linear(20, 10).to(dtype=dtype)
                self.hidden2 = nn.Linear(10, 5).to(dtype=dtype)
                self.output = nn.Linear(5, 1).to(dtype=dtype)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = torch.relu(self.hidden1(x))
                x = torch.relu(self.hidden2(x))
                x = self.output(x)
                return self.sigmoid(x)

        model = BinaryClassifier(self.dtype).to(self.device)
        loss_function = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        start_time = time.time()
        
        for epoch in range(1000):
            predictions = model(X)
            loss = loss_function(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time() - start_time

        result = {
            'Task': 'Binary Classification',
            'Device': str(self.device),
            'Number of Parameters': sum(p.numel() for p in model.parameters()),
            'Final Loss': loss.item(),
            'Time': end_time
        }
        self.scoreboard.append(result)
        return self

    # ... other tasks can be added here ...

    def display_scoreboard(self):
        print("\n--- Scoreboard ---")
        for result in self.scoreboard:
            print(result)
        print("------------------\n")


    def visualize_scoreboard(self):
        tasks = [result['Task'] for result in self.scoreboard]
        times = [result['Time'] for result in self.scoreboard]

        plt.bar(tasks, times)
        plt.ylabel('Time (seconds)')
        plt.title('Benchmark Performance')

        # Get the current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Navigate to the parent directory and then to the 'visualizations' folder
        output_dir = os.path.join(os.path.dirname(current_dir), 'visualizations')

        # Create the directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Construct the filename
        filename = os.path.join(output_dir, 'benchmark_performance.png')

        # Save the plot
        plt.savefig(filename)

        # Display the plot
        plt.show(block=False)