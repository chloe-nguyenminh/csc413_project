# MathBERT-T5 Hybrid Model for Mathematical Reasoning

This repository contains the final project for CSC413H5 - Deep Learning and Neural Networks in Fall 2024 at the University of Toronto Mississauga. The project is the development of a hybrid MathBERT-T5 model capable of advanced mathematical reasoning tasks. We aim to leverage the strengths of both MathBERT (optimized for mathematical text understanding) and T5 (text-to-text transformer model) to be able to solve a wide range of mathematical problems.

## Project Overview

Mathematical reasoning is a complex task that requires both comprehension of mathematical language and problem-solving skills. By combining MathBERT's ability to understand mathematical expressions and T5's general-purpose reasoning capabilities, this project aims to create a hybrid model capable of tackling:

- Mathematical equation solving
- Math word problem analysis

The repository includes code, experimental results, and analysis.

### Key Files 
- `notebook/project.ipynb`: This notebook currently contains all of the implementation details, training process, and evaluation results. 
- **project.pdf**: The final written report on the project.

### Reproducing Results
   Follow the instructions in the notebook to reproduce the experiments. Note that our current results are limited to training on a relatively small subset of the available MATH dataset (10,000 out of approximately 670,000 samples) on the Google Colab T4 GPU. 

### Future Directions
   Below are some TODOs that will be implemented in the future to further improve the project:
   - Update the final report according to feedbacks from TAs:
        - Abstract: Include summary of results and limitations of the model.
        - Data: Include methods to address data imbalance. (e.g., weighted loss functions, augmentation, separate evaluations for under-represented classes, etc.)
        - Discussion: More in-depth investigation of overfitting in perplexity score.
        - Conclusion: Add section
   - Further implementation:
        - Complete implementation of datastream training. Conduct training on the entire dataset on a stronger GPU with higher RAM limit.
        - Add implementation of parallelized and distributed training. 

## Contributions

This project was completed in a collaborative effort as part of our CSC413 coursework, with equal contribution from all members.

