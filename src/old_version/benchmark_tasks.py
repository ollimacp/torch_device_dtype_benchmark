import torch
import torch.nn as nn
import torch.quantization


import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant


from sklearn.datasets import make_classification
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


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


    def int_float_cpu_gpu_neural_network_benchmark(self, max_params=10000):
        # Generate simple integer dataset
        X = np.random.randint(0, 10, size=(1000, 10))
        y = (X.sum(axis=1) > 45).astype(int)
        
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)



        # Compute neurons for the hidden layer based on max_params
        input_neurons = 10
        output_neurons = 1

        # Use the formula to calculate the number of neurons in the hidden layer to approximate max_params
        # Simplified equation: hidden_neurons^2 + 11*hidden_neurons - max_params = 0
        # We'll solve this quadratic equation for hidden_neurons
        coeff = [1, 11, -(max_params - output_neurons)]
        solutions = np.roots(coeff)
        hidden_neurons = int(np.max(solutions))*3  # Choose the maximum root

        # Define Brevitas neural network with dynamic hidden layer size
        class BrevitasIntNN(nn.Module):
            def __init__(self):
                super(BrevitasIntNN, self).__init__()
                self.quant_inp = qnn.QuantIdentity(bit_width=4)
                self.fc1 = qnn.QuantLinear(input_neurons, hidden_neurons, bias=True, weight_bit_width=4)
                self.relu1 = qnn.QuantReLU(bit_width=4)
                self.fc2 = qnn.QuantLinear(hidden_neurons, output_neurons, bias=True, weight_bit_width=4)

            def forward(self, x):
                x = self.quant_inp(x)
                x = self.relu1(self.fc1(x))
                x = self.fc2(x)
                return torch.sigmoid(x)

        model = BrevitasIntNN().to(self.device)
        loss_function = nn.BCELoss()
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



    # Dummy function for the dynamic neural network generation
    # This is a basic demonstration. In practice, this can be more sophisticated.
    def generate_dynamic_nn(self, input_size, max_params, dtype):
        """Generate a dynamic neural network based on max parameter count."""
        # For simplicity, let's assume each layer has equal neurons
        # and the network has 2 hidden layers.
        # Calculate neurons per layer based on max_params.
        # This is a simplistic calculation and can be made more sophisticated.
        neurons_per_layer = int((max_params / (input_size + 1)) ** 0.5)
        hidden_layer1 = nn.Linear(input_size, neurons_per_layer).to(dtype=dtype)
        hidden_layer2 = nn.Linear(neurons_per_layer, neurons_per_layer).to(dtype=dtype)
        output_layer = nn.Linear(neurons_per_layer, 1).to(dtype=dtype)
        
        return nn.Sequential(hidden_layer1, nn.ReLU(), hidden_layer2, nn.ReLU(), output_layer, nn.Sigmoid())



    def quantization_neural_network(self, max_params=1000):  # default max_params can be set as needed

        # Generate simple integer dataset
        X = np.random.randint(0, 10, size=(1000, 10))
        y = (X.sum(axis=1) > 45).astype(int)
        
        X = torch.tensor(X, dtype=self.dtype).to(self.device)  # Convert to float for training
        y = torch.tensor(y, dtype=self.dtype).view(-1, 1).to(self.device)

        # Use dynamic NN generation
        model = self.generate_dynamic_nn(X.size(1), max_params, self.dtype).to(self.device)
        
        loss_function = nn.BCELoss()
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
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
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







    # Sample updated int_neural_network function
    def float_neural_network(self, max_params=1000):  # default max_params can be set as needed
        # Generate simple integer dataset
        X = np.random.randint(0, 10, size=(1000, 10))
        y = (X.sum(axis=1) > 45).astype(int)
        
        X = torch.tensor(X, dtype=self.dtype).to(self.device)
        y = torch.tensor(y, dtype=self.dtype).view(-1, 1).to(self.device)

        # Use dynamic NN generation
        model = self.generate_dynamic_nn(X.size(1), max_params, self.dtype).to(self.device)
        
        loss_function = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        start_time = time.time()
        
        print(f"Training Int Neural Network with {sum(p.numel() for p in model.parameters())} parameters...")
        
        for epoch in tqdm(range(1000)):  # Use tqdm for progress bar
            predictions = model(X)
            loss = loss_function(predictions, y.float())  # Ensure y is float for BCELoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time() - start_time

        result = {
            'Task': 'Int Neural Network',
            'Device': str(self.device),
            'Number of Parameters': sum(p.numel() for p in model.parameters()),
            'Final Loss': loss.item(),
            'Time': end_time
        }
        self.scoreboard.append(result)
        return self

        '''

        # Neural Network for CPU Cache with Int Types
        def int_neural_network(self):
            # Generate simple integer dataset
            X = np.random.randint(0, 10, size=(1000, 10))
            y = (X.sum(axis=1) > 45).astype(int)  # Label 1 if sum > 45 else 0
            
            X = torch.tensor(X, dtype=self.dtype).to(self.device)
            y = torch.tensor(y, dtype=self.dtype).view(-1, 1).to(self.device)

            class IntNN(nn.Module):
                def __init__(self, dtype):
                    super().__init__()
                    self.hidden1 = nn.Linear(10, 5).to(dtype=dtype)
                    self.output = nn.Linear(5, 1).to(dtype=dtype)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x):
                    x = torch.relu(self.hidden1(x))
                    x = self.output(x)
                    return self.sigmoid(x)

            model = IntNN(self.dtype).to(self.device)
            loss_function = nn.BCELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            start_time = time.time()
            
            for epoch in range(100000):  # Reduced number of epochs for quick benchmark
                predictions = model(X)
                loss = loss_function(predictions, y.float())  # Ensure y is float for BCELoss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            end_time = time.time() - start_time

            result = {
                'Task': 'Int Neural Network',
                'Device': str(self.device),
                'Number of Parameters': sum(p.numel() for p in model.parameters()),
                'Final Loss': loss.item(),
                'Time': end_time
            }
            self.scoreboard.append(result)
            return self

        '''
    
    # Simple CNN Benchmark
    def simple_cnn(self):
        # Generate small image dataset
        X = np.random.rand(1000, 1, 28, 28)
        y = np.random.randint(0, 2, 1000)
        
        X = torch.tensor(X, dtype=self.dtype).to(self.device)
        y = torch.tensor(y, dtype=self.dtype).view(-1, 1).to(self.device)

        class SimpleCNN(nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, 3, 1).to(dtype=dtype)
                self.fc1 = nn.Linear(26*26*16, 64).to(dtype=dtype)
                self.fc2 = nn.Linear(64, 1).to(dtype=dtype)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.conv1(x)
                x = torch.relu(x)
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return self.sigmoid(x)

        model = SimpleCNN(self.dtype).to(self.device)
        loss_function = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        start_time = time.time()
        
        for epoch in range(500):  # Reduced number of epochs for quick benchmark
            predictions = model(X)
            loss = loss_function(predictions, y.float())  # Ensure y is float for BCELoss

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
    
    
    # Simple GAN Benchmark
    def simple_gan(self):
        # Generate small image dataset
        X = np.random.rand(1000, 1, 28, 28)
        
        X = torch.tensor(X, dtype=self.dtype).to(self.device)

        class Generator(nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.fc1 = nn.Linear(100, 256).to(dtype=dtype)
                self.fc2 = nn.Linear(256, 1*28*28).to(dtype=dtype)

            def forward(self, x):
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return torch.tanh(x).view(x.size(0), 1, 28, 28)

        class Discriminator(nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.fc1 = nn.Linear(1*28*28, 256).to(dtype=dtype)
                self.fc2 = nn.Linear(256, 1).to(dtype=dtype)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return torch.sigmoid(x)

        generator = Generator(self.dtype).to(self.device)
        discriminator = Discriminator(self.dtype).to(self.device)
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.01)
        loss_function = nn.BCELoss()

        start_time = time.time()
        
        for epoch in range(100):  # Reduced number of epochs for quick benchmark
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
