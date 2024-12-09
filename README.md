
# BALANCER: Data Augmentation Method Selection for Imbalanced Time-Series Classification

This repository contains the implementation of **BALANCER** (imBALanced AugmeNtation reCommendER), a machine learning-based decision support system that recommends the most effective data augmentation (DA) technique for imbalanced time-series classification (ITSC) problems. It uses dataset-specific features and classifier information to predict the performance improvements after applying a DA method.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Running BALANCER](#running-balancer)
  - [Understanding the Example](#notebook-example)

## Overview

BALANCER supports the following data augmentation techniques:
- SMOTE
- ADASYN
- Jittering
- Time Warping (TW)
- DTW-SMOTE
- TimeGAN
- Random Oversampling (ROS)

By evaluating various time-series datasets and classifiers, BALANCER predicts which augmentation technique will yield the best performance improvements for your specific dataset.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/DorianJoubaud/balancer.git
   cd balancer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running BALANCER

You can run BALANCER with your own time-series dataset by following these steps:

1. Prepare your dataset in the required format.
2. Run the BALANCER script:
   ```bash
   python balancer.py
   ```

### Notebook Example

The repository includes an example in the form of a Jupyter Notebook (`example_run.ipynb`). It demonstrates how to use the BALANCER model with a simple dataset.

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook example_run.ipynb
   ```
2. Follow the instructions in the notebook to run the model on a sample dataset and visualize the recommended DA techniques and their performance.

