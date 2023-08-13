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

    tasks.scoreboard = scoreboard_log


    while True:

        print("Choose the task to execute:")
        print("1. Binary Classification")
        print("2. All Tasks")
        choice = int(input("Enter the task number: "))

        #args.device , args.dtype = prompt_for_parameters()

        # Update the device and dtype in tasks if changed
        device = torch.device(args.device)
        dtype = dtype_mapping[args.dtype]


        if choice == 1:
            tasks = tasks.binary_classification()
        elif choice == 2:
            for func_name in dir(tasks):
                if not func_name.startswith("__"):
                    func = getattr(tasks, func_name)
                    if callable(func) and func_name not in ['display_scoreboard', 'visualize_scoreboard']:
                        func()


   

        # Create a copy of the scoreboard for pretty printing
        pretty_scoreboard = [dict(item) for item in tasks.scoreboard]  # Deep copy
        for item in pretty_scoreboard:
            item['Final Loss'] = round(item['Final Loss'], 3)
            item['Time'] = f"{round(float(item['Time']), 3)} seconds"  # Ensure 'Time' is treated as a float
        pprint.pprint(pretty_scoreboard)

        # Prompt to rerun
        rerun = input("Do you want to rerun the benchmark? (yes/no/delete): ")
        if rerun.lower() == 'yes':
            device, dtype = prompt_for_parameters(device, dtype) # Notice that I'm passing device and dtype, not args.device and args.dtype
            tasks.device = device
            tasks.dtype = dtype
            pass
        elif rerun.lower() == 'no':
            break
        elif rerun.lower() == "delete":
            tasks.scoreboard =[]
            break
        else:
            print(f"Please provide valid input!")
            continue

        # Save the updated scoreboard_log
        save_scoreboard(tasks.scoreboard, log_file_path)

        # Reset tasks for new run
        tasks = BenchmarkTasks(device, dtype, tasks.scoreboard)