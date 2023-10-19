import gc
import os

import torch
import numpy as np
import torch.nn as nn

# from datasets import load_dataset
import json
import pandas as pd

# Set the random seed for NumPy
np.random.seed(0)

# Set the random seed for PyTorch
torch.manual_seed(0)

# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(0)

harmless_instructions_path = "../datasets/custom_harm_dataset/harmless_instructions.json"
harmful_instructions_path = "../datasets/custom_harm_dataset/harmful_instructions.json"

def create_harmless_instructions_json():
    # first download the alpaca data from https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
    file_name = "../datasets/custom_harm_dataset/alpaca_data.json"
    alpaca_data = json.load(open(file_name, "r"))
    instructions = [datapoint["instruction"].strip(".") for datapoint in alpaca_data]
    print(instructions[:10])
    # sort all instructions by length in decreasing order
    instructions.sort(key=lambda x: len(x), reverse=True)
    # we want to take instructions of length of rougly 75 characters
    # to match the average length of the harmful prompts in advbench
    instructions = instructions[9000:9000+388]
    print(instructions[:10])
    print(sum([len(x) for x in instructions]) / len(instructions))
    with open(harmless_instructions_path, "w") as file:
        json.dump(instructions, file)

def create_harmful_instructions_json():
    original_harmful_instructions_path = "../datasets/advbench/harmful_instructions.csv"
    with open(original_harmful_instructions_path, "r") as file:
        instructions = file.readlines()
    instructions = [instruction.strip(".") for instruction in instructions]
    print(sum([len(x) for x in instructions]) / len(instructions))
    with open("../datasets/custom_harm_dataset/harmful_instructions.json", "w") as file:
        json.dump(instructions, file)

def extract_harmful_LAT_reading_vector():
    # create_harmless_instructions_json()
    # create_harmful_instructions_json()

    harmless_instructions = json.load(harmless_instructions_path)
    harmful_instructions = json.load(harmful_instructions_path)

    # shuffle(harmless_instructions)

    # we set aside some of the instructions to test how good our reading vector is
    train_harmless_instructions = harmless_instructions[:int(len(harmless_instructions) * 0.8)]
    test_harmless_instructions = harmless_instructions[int(len(harmless_instructions) * 0.8):]

    train_harmful_instructions = harmful_instructions[:int(len(harmful_instructions) * 0.8)]
    test_harmful_instructions = harmful_instructions[int(len(harmful_instructions) * 0.8):]
    
    
    
    

if __name__ == "__main__":
    extract_harmful_LAT_reading_vector()