# CSC413 Final Project
# Math-BERT-T5 Hybrid Model for Mathematical Reasoning

This repository contains the final project for CSC413, where we develop a hybrid model combining Math-BERT and T5 for advanced mathematical reasoning tasks. The project focuses on leveraging the strengths of both Math-BERT (optimized for mathematical text understanding) and T5 (a powerful text-to-text transformer model) to solve challenging mathematical problems.

## Project Overview

Mathematical reasoning is a complex task that requires both comprehension of mathematical language and problem-solving skills. By combining Math-BERT's ability to understand mathematical expressions and T5's general-purpose reasoning capabilities, this project aims to create a hybrid model capable of tackling:

- Mathematical equation solving
- Proof generation
- Word problem analysis

The repository includes code, experimental results, and analysis.

### Key Files
- **`notebook/project.ipynb`**: The primary Jupyter Notebook where all the code implementation, model training, and evaluation results are documented. 
   Navigate to the `notebook/project.ipynb` notebook to view the implementation details, training process, and evaluation results. 
 - **project.pdf**: The final written report on the project.

**Run the Notebook**
   Follow the instructions in the notebook to reproduce the experiments. Note that our current results are limited to training on a relatively small subset of the available MATH dataset (10,000 out of approximately 670,000 samples) on the Google Colab T4 GPU. Future directions include more extensive training, as well as implementation of parallelization and distributed training on a stronger GPU with greater RAM limit.

## Contributions

This project was developed in a collaborative effort as part of our CSC413 coursework. 

