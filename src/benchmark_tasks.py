import torch
import torch.nn as nn
from torchvision import datasets, transforms
import brevitas.nn as qnn
from tqdm import tqdm
from sklearn.datasets import make_classification
import numpy as np
import time
from pprint import pprint
import matplotlib.pyplot as plt
import os 

class DynamicNN(nn.Module):
    def __init__(self, layers):
        super(DynamicNN, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.layers(x)

class BenchmarkTasks:
    def __init__(self, device, dtype, scoreboard=[]):
        self.device = device
        self.dtype = dtype
        self.scoreboard = scoreboard

        # Load MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.mnist_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        self.data_loader = torch.utils.data.DataLoader(dataset=self.mnist_data, batch_size=1000, shuffle=True)
        self.data_iter = iter(self.data_loader)
        self.X, self.y = next(self.data_iter)
        self.X = self.X.to(self.device)
        self.y = self.y.view(-1, 1).to(self.device).float()


    ##############  HELPER FUNCTIONS


    def generate_dynamic_nn(self, input_size, output_size, max_params, dtype, no_of_hidden_layers=2, activation_function=nn.ReLU):
        """Generate a dynamic neural network based on max parameter count."""
        
        # Calculate parameters required for input and output layers
        params_for_io = input_size * output_size + output_size
        remaining_params = max_params - params_for_io
        
        # Check if remaining_params is sufficient for even a single neuron in hidden layers
        if remaining_params <= no_of_hidden_layers * (input_size + 1):
            raise ValueError("max_params is too low to create the desired network architecture")
        
        # Calculate neurons per layer based on remaining_params
        neurons_per_layer = int((remaining_params / (no_of_hidden_layers * (input_size + 1))) ** 0.5)
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, neurons_per_layer).to(dtype=dtype))
        layers.append(activation_function())
        
        # Hidden layers
        for _ in range(no_of_hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer).to(dtype=dtype))
            layers.append(activation_function())
        
        # Output layer
        layers.append(nn.Linear(neurons_per_layer, output_size).to(dtype=dtype))
        
        return DynamicNN(layers)


    def compute_hidden_neurons(self, max_params, input_neurons, output_neurons):
        coeff = [1, 11, -(max_params - output_neurons)]
        solutions = np.roots(coeff)
        return int(np.max(solutions))

    def benchmark(self, model, loss_function, optimizer, epochs=1000):
        for epoch in tqdm(range(epochs)):
            predictions = model(self.X)
            loss = loss_function(predictions, self.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate performance
        start_time = time.time()
        for _ in range(1000):
            predictions = model(self.X)
        end_time = time.time() - start_time

        return {
            'Number of Parameters': sum(p.numel() for p in model.parameters()),
            'Final Loss': loss.item(),
            'Time': end_time
        }


    ##############  SCOREBOARD FUNCTIONS

    def efficiency_score(self, loss, time, num_params, max_params=100000, max_time=60, target_loss=0.1):
        # This is a simple scoring mechanism. Lower scores are better.
        # We penalize for number of parameters, time taken and deviation from target loss.
        return (num_params / max_params) + (time / max_time) + (loss / target_loss)


    def binary_classification(self, max_params=10000):
        input_neurons = 28 * 28  # MNIST image size
        output_neurons = 1
        hidden_neurons = self.compute_hidden_neurons(max_params, input_neurons, output_neurons)

        class BinaryClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden1 = nn.Linear(input_neurons, hidden_neurons)
                self.hidden2 = nn.Linear(hidden_neurons, hidden_neurons)
                self.output = nn.Linear(hidden_neurons, output_neurons)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = x.view(x.size(0), -1)  # Flatten the image
                x = torch.relu(self.hidden1(x))
                x = torch.relu(self.hidden2(x))
                x = self.output(x)
                return self.sigmoid(x)

        model = BinaryClassifier().to(self.device)
        result = self.benchmark(model, nn.BCELoss(), torch.optim.SGD(model.parameters(), lr=0.01))

        result['Task'] = 'Binary Classification'
        result['Device'] = str(self.device)
        result['Efficiency Score'] = self.efficiency_score(result['Final Loss'], result['Time'], result['Number of Parameters'])

        self.scoreboard.append(result)
        return self

    def display_scoreboard(self):
        print("\n--- Scoreboard ---")
        pprint(self.scoreboard)


    def visualize_scoreboard(self):
        tasks = [result['Task'] for result in self.scoreboard]
        efficiencies = [self.efficiency_score(result['Final Loss'], result['Time'], result['Number of Parameters']) for result in self.scoreboard]
        losses = [-result['Final Loss'] for result in self.scoreboard]  # Using negative since lower loss is better
        times = [result['Time'] for result in self.scoreboard]

        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plotting efficiency score in the first subplot
        axs[0].bar(tasks, efficiencies, label='Efficiency Score', alpha=0.6, color='blue')
        axs[0].set_ylabel('Efficiency Score')
        axs[0].legend()
        axs[0].set_title('Benchmark Performance')

        # Plotting negative loss and execution time in the second subplot
        axs[1].bar(tasks, losses, label='Negative Loss', alpha=0.6, color='red')
        axs[1].bar(tasks, times, label='Execution Time', alpha=0.6, color='green')
        axs[1].set_yscale('log')  # Set y-axis to logarithmic scale for second subplot
        axs[1].set_ylabel('Metrics (Log Scale)')
        axs[1].set_xticks(tasks)
        axs[1].set_xticklabels(tasks, rotation=45, ha='right')
        axs[1].legend()

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
        plt.tight_layout()
        plt.savefig(filename)

        # Display the plot
        plt.show(block=False)


    ##############  BENCHMARK FUNCTIONS


    def int_float_cpu_gpu_neural_network_benchmark(self, max_params=10000):
        # Use the pre-loaded MNIST data
        X, y = self.X, self.y.argmax(dim=1)  # Convert one-hot encoded labels to class indices

        # Compute neurons for the hidden layer based on max_params
        input_neurons = 28 * 28  # MNIST image size
        output_neurons = 10  # MNIST classes (digits 0-9)

        # Use the formula to calculate the number of neurons in the hidden layer to approximate max_params
        coeff = [1, input_neurons + output_neurons, -max_params]
        solutions = np.roots(coeff)
        hidden_neurons = int(np.max(solutions))

        # Define Brevitas neural network with dynamic hidden layer size
        class BrevitasIntNN(nn.Module):
            def __init__(self):
                super(BrevitasIntNN, self).__init__()
                self.quant_inp = qnn.QuantIdentity(bit_width=4)
                self.fc1 = qnn.QuantLinear(input_neurons, hidden_neurons, bias=True, weight_bit_width=4)
                self.relu1 = qnn.QuantReLU(bit_width=4)
                self.fc2 = qnn.QuantLinear(hidden_neurons, output_neurons, bias=True, weight_bit_width=4)

            def forward(self, x):
                x = x.view(x.size(0), -1)  # Flatten the image
                x = self.quant_inp(x)
                x = self.relu1(self.fc1(x))
                x = self.fc2(x)
                return x  # No sigmoid here, since we're doing multi-class classification

        model = BrevitasIntNN().to(self.device)
        loss_function = nn.CrossEntropyLoss()  # Replace BCE with categorical cross-entropy
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Train the Brevitas model
        print(f"Training Int Neural Network with {sum(p.numel() for p in model.parameters())} parameters...")
        for epoch in tqdm(range(1000)):  # Use tqdm for progress bar
            predictions = model(X)
            loss = loss_function(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model's performance
        start_time = time.time()
        for _ in tqdm(range(1000)):  # Using the same number of evaluations for consistency
            predictions = model(X)
        end_time = time.time() - start_time

        # Calculate final loss
        loss = loss_function(predictions, y)

        result = {
            'Task': 'Int Neural Network (Brevitas)',
            'Device': str(self.device),
            'Number of Parameters': sum(p.numel() for p in model.parameters()),
            'Final Loss': loss.item(),
            'Time': end_time
        }
        self.scoreboard.append(result)
        return self

        
    def quantization_neural_network(self, max_params=10000):  # default max_params can be set as needed
        # Use the pre-loaded MNIST data
        X, y = self.X, self.y.argmax(dim=1)  # Convert one-hot encoded labels to class indices

        # Use dynamic NN generation
        model = self.generate_dynamic_nn(784, 10, max_params, self.dtype).to(self.device) #mnist pixels and 10 classes

        loss_function = nn.CrossEntropyLoss()  # Use categorical cross-entropy
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 1. Train the model in floating-point
        print(f"Training Int Neural Network with {sum(p.numel() for p in model.parameters())} parameters...")
        for epoch in tqdm(range(1000)):  # Use tqdm for progress bar
            predictions = model(X)
            loss = loss_function(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 2. Calibrate the model and convert to quantized version
        model.eval()

        # Set the qconfig for per_tensor_affine quantization
        qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model.qconfig = qconfig

        torch.quantization.prepare(model, inplace=True)
        with torch.no_grad():
            model(X)
        torch.quantization.convert(model, inplace=True)

        # 3. Evaluate the quantized model's performance
        start_time = time.time()
        for _ in tqdm(range(1000)):  # Using the same number of evaluations for consistency
            predictions = model(X)
        end_time = time.time() - start_time

        # Calculate final loss with quantized model
        loss = loss_function(predictions, y)

        result = {
            'Task': 'Int Neural Network (Quantized)',
            'Device': str(self.device),
            'Number of Parameters': sum(p.numel() for p in model.parameters()),
            'Final Loss': loss.item(),
            'Time': end_time
        }
        self.scoreboard.append(result)
        return self


    def float_neural_network(self, max_params=10000):
        # Use the pre-loaded MNIST data
        X, y = self.X, self.y.argmax(dim=1)  # Convert one-hot encoded labels to class indices
        

        # Use dynamic NN generation
        model = self.generate_dynamic_nn(784, 10, max_params, self.dtype).to(self.device)  # Set input_size=784, output_size=10

        loss_function = nn.CrossEntropyLoss()  # Use categorical cross-entropy
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        start_time = time.time()

        print(f"Training Float Neural Network with {sum(p.numel() for p in model.parameters())} parameters...")

        for epoch in tqdm(range(1000)):
            predictions = model(X)
            loss = loss_function(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time() - start_time

        result = {
            'Task': 'Float Neural Network',
            'Device': str(self.device),
            'Number of Parameters': sum(p.numel() for p in model.parameters()),
            'Final Loss': loss.item(),
            'Time': end_time
        }
        self.scoreboard.append(result)
        return self


    def simple_cnn(self):
        # Use the pre-loaded MNIST data
        X, y = self.X, self.y.argmax(dim=1)  # Convert one-hot encoded labels to class indices

        class SimpleCNN(nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, 3, 1).to(dtype=dtype)
                self.fc1 = nn.Linear(26*26*16, 64).to(dtype=dtype)
                self.fc2 = nn.Linear(64, 10).to(dtype=dtype)  # Predicting 10 classes

            def forward(self, x):
                x = self.conv1(x)
                x = torch.relu(x)
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return x  # Removed sigmoid for multi-class prediction

        model = SimpleCNN(self.dtype).to(self.device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        start_time = time.time()

        for epoch in tqdm(range(500)):  # Reduced number of epochs for quick benchmark
            predictions = model(X)
            loss = loss_function(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time() - start_time

        result = {
            'Task': 'Simple CNN',
            'Device': str(self.device),
            'Number of Parameters': sum(p.numel() for p in model.parameters()),
            'Final Loss': loss.item(),
            'Time': end_time
        }
        self.scoreboard.append(result)
        return self

        
    def simple_gan(self, max_params=10000000):
        # Use the pre-loaded MNIST data
        X = self.X

        # Generator using dynamic NN generation
        generator = self.generate_dynamic_nn(100, 784, max_params=10000, dtype=self.dtype, no_of_hidden_layers=2, activation_function=nn.ReLU()).to(self.device)
        
        # Discriminator using dynamic NN generation
        discriminator = self.generate_dynamic_nn(784, 1, max_params=10000, dtype=self.dtype, no_of_hidden_layers=2, activation_function=nn.ReLU()).to(self.device)

        optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.01)
        loss_function = nn.BCELoss()

        start_time = time.time()
        
        for epoch in tqdm(range(500)):  # Reduced number of epochs for quick benchmark
            # Train Discriminator
            optimizer_d.zero_grad()
            
            # Real data
            real_data = X
            real_labels = torch.ones((X.size(0), 1), dtype=self.dtype).to(self.device)
            outputs = discriminator(real_data)
            loss_real = loss_function(outputs, real_labels)
            
            # Fake data
            z = torch.randn((X.size(0), 100), dtype=self.dtype).to(self.device)
            fake_data = generator(z)
            fake_labels = torch.zeros((X.size(0), 1), dtype=self.dtype).to(self.device)
            outputs = discriminator(fake_data.detach())
            loss_fake = loss_function(outputs, fake_labels)
            
            # Combine losses
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()
            
            # Train Generator
            optimizer_g.zero_grad()
            z = torch.randn((X.size(0), 100), dtype=self.dtype).to(self.device)
            fake_data = generator(z)
            outputs = discriminator(fake_data)
            loss_g = loss_function(outputs, real_labels)
            loss_g.backward()
            optimizer_g.step()

        end_time = time.time() - start_time

        result = {
            'Task': 'Simple GAN',
            'Device': str(self.device),
            'Number of Parameters': sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in discriminator.parameters()),
            'Final Loss': (loss_g.item() + loss_d.item()) / 2,
            'Time': end_time
        }
        self.scoreboard.append(result)
        return self


        # Now, the ExtendedBenchmarkTasks class includes all three new benchmark tasks.
