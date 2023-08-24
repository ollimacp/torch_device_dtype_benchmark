'''
To run the script, use the following command:
python main.py --device cuda:0 --dtype float16
'''

import argparse
import torch
import json
import os
import pprint
from benchmark_tasks import BenchmarkTasks
#from benchmark_tasks import ExtendedBenchmarkTasks  # Import the extended class


dtype_mapping = {
    'float32': torch.float32, 'float': torch.float, 'float64': torch.float64, 'double': torch.double,
    'float16': torch.float16, 'half': torch.half, 'bfloat16': torch.bfloat16,
    'int8': torch.int8, 'int16': torch.int16, 'short': torch.short,
    'int32': torch.int32, 'int': torch.int, 'int64': torch.int64, 'long': torch.long,
    'bool': torch.bool, 'uint8': torch.uint8
}

def prompt_for_parameters(device, dtype):
    new_device = input(f"Please specify device (cpu/cuda:0/cuda:1/...)\nPress Enter to continue without changing")
    if new_device == "":
        pass 
    else:
        device = torch.device(new_device)
        new_device = ""
    
    new_dtype = input(f"Please specify new dtype of these types:\ņ{dtype_mapping.keys()}\nPress Enter to continue without changing")
    if new_dtype == "":
        pass 
    else:
        dtype = dtype_mapping[new_dtype]

    return device, dtype

def load_scoreboard(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []

def save_scoreboard(scoreboard, file_path):
    with open(file_path, 'w') as file:
        json.dump(scoreboard, file, indent=4)
    print(f"Scoreboard saved to {file_path}")





def get_benchmark_methods(obj):
    """Get all callable methods of an object excluding unwanted methods."""
    # List of methods that aren't benchmarks but are callable
    excluded_methods = ['display_scoreboard', 'Getstate', '__get_state__', '__init_subclass__','Init Subclass', 'compute_hidden_neurons', 'benchmark', 'efficiency_score', 'generate_dynamic_nn', 'DynamicNN', '', '']
    
    methods = [method for method in dir(obj) 
               if callable(getattr(obj, method)) and 
               not method.startswith("__") and 
               method not in excluded_methods]
    return methods







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark a Neural Network')
    parser.add_argument('--device', type=str, default=None, help='Device to run on (e.g., "cpu", "cuda:0")')
    parser.add_argument('--dtype', type=str, default=None, help='Data type to use (from torch datatypes)')
    args = parser.parse_args()

    if args.device is None or args.dtype is None:
        device, dtype = prompt_for_parameters(args.device, args.dtype)
    else:
        # Convert argparse values to torch.device and torch.dtype
        device = torch.device(args.device)
        dtype = dtype_mapping[args.dtype]
        print(f"This is the dtype: {dtype}")


    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the parent directory and then to the 'tests' folder
    log_file_path = os.path.join(os.path.dirname(current_dir), 'tests/scoreboard_log.json')

    # Load previous scoreboard
    scoreboard_log = load_scoreboard(log_file_path)
    
    print(f"Tasks.scoreboard is type: {type(scoreboard_log)} and contents {scoreboard_log}")


    tasks = BenchmarkTasks(device, dtype, scoreboard_log)
    #tasks = ExtendedBenchmarkTasks(device, dtype, scoreboard_log)  # Use the extended class

    tasks.scoreboard = scoreboard_log

    MAX_ITERATIONS = 10  # Set a maximum number of iterations for the loop

    for iteration in range(1, MAX_ITERATIONS + 1):
        
        print(f"\n=== Iteration {iteration}/{MAX_ITERATIONS} ===\n")
        
        methods = get_benchmark_methods(tasks)
        
        print("Choose the task to execute:")
        for idx, method in enumerate(methods, 1):
            print(f"{idx}. {method.replace('_', ' ').title()}")  # Convert method name to a readable format
        print(f"{len(methods) + 1}. All Tasks")
        
        choice = int(input("Enter the task number: "))

        # Update the device and dtype in tasks
        device = torch.device(args.device)
        dtype = dtype_mapping[args.dtype]

        if 1 <= choice <= len(methods):
            func = getattr(tasks, methods[choice - 1])
            func()
        elif choice == len(methods) + 1:  # All tasks
            for method in methods:
                func = getattr(tasks, method)
                func()
        else:
            print("Invalid choice! Please choose a valid task number.")
            continue  # Skip the rest of this iteration and prompt again
        
        # Display the scoreboard
        pretty_scoreboard = [dict(item) for item in tasks.scoreboard]  # Deep copy
        for item in pretty_scoreboard:
            item['Final Loss'] = round(item['Final Loss'], 3)
            item['Time'] = f"{round(float(item['Time']), 3)} seconds"  # Ensure 'Time' is treated as a float
        pprint.pprint(pretty_scoreboard)


        # Save the updated scoreboard_log at the end of each iteration
        save_scoreboard(tasks.scoreboard, log_file_path)

        # Prompt to rerun or change settings
        rerun = input("Do you want to rerun the benchmark? (YES/no/change/delete): ")
        
        if rerun.lower() == 'yes':
            continue
        elif rerun.lower() == 'no':
            break
        elif rerun.lower() == "change":
            device, dtype = prompt_for_parameters(device, dtype) # Update device and dtype
            args.device = str(device)  # Update the args values too
            args.dtype = str(dtype)
        elif rerun.lower() == "delete":
            tasks.scoreboard = []
            save_scoreboard(tasks.scoreboard, log_file_path)
        else:
            print(f"Please provide valid input!")
            continue  # Skip the rest of this iteration and prompt again



        # Reset tasks for next run
        tasks = BenchmarkTasks(device, dtype, tasks.scoreboard)