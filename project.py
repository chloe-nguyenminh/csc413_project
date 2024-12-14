#%pip install numpy
#%pip install pandas
#%pip install matplotlib
#%pip install datasets
#%pip install kagglehub
#%pip install transformers
#%pip install accelerate
#%pip install latex2sympy
#%pip install --upgrade torch torchvision torchaudio
#%pip install --upgrade torchtext

def print_entries(dataset, start=0, end=10, split=""):
  if split == "":
    for split in dataset:
      print(f"Entries {start+1} - {end} of the {split} data:")
      for i in range(start, end):
        print(dataset[split][i])
      print("-" * 20)
  else:
    print(f"Entries {start+1} - {end} of the {split} data:")
    for i in range(start, end):
      print(dataset[split][i])
    print("-" * 20)

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from datasets import load_dataset

cot_ds = load_dataset("AI-MO/NuminaMath-CoT")

print("Before preprocessing")
print(cot_ds)
print()

# Preprocess COT dataset
cot_ds['train'] = cot_ds['train'].remove_columns(['messages'])
cot_ds['test'] = cot_ds['test'].remove_columns(['messages'])
cot_ds['train'] = cot_ds['train'].remove_columns(['source'])
cot_ds['test'] = cot_ds['test'].remove_columns(['source'])
print("After preprocessing")
print(cot_ds)

# Remove chinese characters from COT dataset
import re

def contains_chinese(text):
    # match Chinese characters
    pattern = re.compile(r'[\u4e00-\u9fff\u2e80-\u2eff\u31c0-\u31ef\uff00-\uffef]')
    return bool(pattern.search(text))

def filter_entries(dataset, fields):
    # Filter out entries that contain Chinese characters
    filtered_dataset = dataset.filter(lambda example: not any(contains_chinese(example[field]) for field in fields))
    return filtered_dataset

# remove entries with Chinese characters
fields_to_check = ['problem', 'solution']
cot_ds['train'] = filter_entries(cot_ds['train'], fields_to_check)
cot_ds['test'] = filter_entries(cot_ds['test'], fields_to_check)
print(cot_ds)

# print first 10 entries for COT dataset

print(cot_ds)
print_entries(cot_ds)

import kagglehub

# Download latest version
path = kagglehub.dataset_download("mathurinache/math-dataset")

print("Path to dataset files:", path)

# Preprocess MATH dataset (load all json files into into Dataset object)

import os
from datasets import Dataset, DatasetDict

def load_json_files(data_dir):
    """Loads JSON files from a directory into a Dataset."""
    all_data = []
    problems = 0
    for subdir in os.listdir(data_dir):
      subdir_path = os.path.join(data_dir, subdir)
      for filename in os.listdir(subdir_path):
        if filename.endswith(".json"):
          problems += 1
          filepath = os.path.join(subdir_path, filename)
          with open(filepath, "r") as f:
            all_data.append(json.load(f))
    # Create a Pandas DataFrame to easily convert into a Dataset\
    print(f"Loaded {problems} problems.")
    return all_data

# Assuming 'path' is from kagglehub.dataset_download
math_dir = os.path.join(path, "MATH")
train_dir = os.path.join(math_dir, "train")
test_dir = os.path.join(math_dir, "test")

train_data = load_json_files(train_dir)
test_data = load_json_files(test_dir)

# Convert the train and test data into Dataset objects
train_dataset = Dataset.from_dict({
    "problem": [item["problem"] for item in train_data],
    # "level": [item["level"] for item in train_data],
    # "type": [item["type"] for item in train_data],
    "solution": [item["solution"] for item in train_data]
})

test_dataset = Dataset.from_dict({
    "problem": [item["problem"] for item in test_data],
    # "level": [item["level"] for item in test_data],
    # "type": [item["type"] for item in test_data],
    "solution": [item["solution"] for item in test_data]
})

math_ds = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})



# print first 10 entries for COT dataset

print(math_ds)
print_entries(math_ds)

from datasets import concatenate_datasets

# Make validation dataset
train_valid_split = cot_ds['train'].train_test_split(test_size=0.1)
cot_ds['train'] = train_valid_split['train']
cot_ds['test'] = train_valid_split['test']

train_valid_split = cot_ds['train'].train_test_split(test_size=0.12)
cot_ds['train'] = train_valid_split['train']
cot_ds['validation'] = train_valid_split['test']

# Add MATH dataset as test dataset
merged_math = concatenate_datasets([math_ds['train'], math_ds['test']])
cot_ds['test'] = concatenate_datasets([cot_ds['test'], merged_math])

ds = cot_ds
print(ds)

print()
print("Split")
print("train:", len(ds['train']) / ( len(ds['train']) + len(ds['validation']) + len(ds['test']) ))
print("test:", len(ds['test']) / ( len(ds['train']) + len(ds['validation']) + len(ds['test']) ))
print("validation:", len(ds['validation']) / ( len(ds['train']) + len(ds['validation']) + len(ds['test']) ))

print(ds)
print("train:", len(ds['train']) / ( len(ds['train']) + len(ds['validation']) + len(ds['test']) ))
print("test:", len(ds['test']) / ( len(ds['train']) + len(ds['validation']) + len(ds['test']) ))
print("validation:", len(ds['validation']) / ( len(ds['train']) + len(ds['validation']) + len(ds['test']) ))

# # Tokenize Data

# from transformers import AutoTokenizer
# model_name = "tbs17/MathBERT"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# ds = ds.map(lambda entries: tokenizer(entries['problem'], entries['solution']), batched=True)

# print(ds)

print(ds)
print_entries(ds)

# Generate embeddings
import torch
from transformers import AutoModel, AutoTokenizer
model_name = "tbs17/MathBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Tokenizer
model = AutoModel.from_pretrained(model_name)  # Model for embeddings


def generate_embeddings(text):
  inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
  with torch.no_grad():  # Disable gradient calcuim tokenizing first, then lation during inference
    outputs = model(**inputs)
  embeddings = outputs.last_hidden_state[:, 0, :]  # Get the [CLS] token embedding
  return embeddings

ds = ds.map(lambda entries: {
    'problem_embeddings': generate_embeddings(entries['problem']),
    'solution_embeddings': generate_embeddings(entries['solution'])
}, batched=True)

torch.save(model.state_dict(), 'mathbert_weights.pth')
# model.load_state_dict(torch.load('mathbert_weights.pth'))

print(ds)

print_entries(ds)

from latex2sympy import latex2sympy
from sympy import symbols, Eq, solve
import re

problem = cot_ds['train'][0]['problem']
print(problem)

# Convert LaTeX to SymPy expression
parts = re.split(r'(?<!\\)\$(.*?)(?<!\\)\$', problem)
sympy_parts = []

for part in parts:
    if re.match(r'(?<!\\)\$(.*?)(?<!\\)\$', part):  # Check if LaTeX
        try:
            sympy_expr = latex2sympy(part[1:-1])  # Remove $ signs
            sympy_parts.append(sympy_expr)
        except Exception as e:
            print(f"Error converting LaTeX to SymPy: {e}")
            sympy_parts.append(part)  # Keep original if conversion fails
    else:
        sympy_parts.append(part)


# Assuming the equation is the second SymPy expression in sympy_parts
equation = sympy_parts[1]

# Define the variable 'y'
y = symbols('y')

# # Solve the equation for 'y'
# solutions = solve(equation, y)

# # Print the solutions
# print("Solutions for y:", solutions)

# import torch
# from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pad_sequence
# import torch.optim as optim
# import torch.nn as nn
# import torchtext
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import Vocab, build_vocab_from_iterator
