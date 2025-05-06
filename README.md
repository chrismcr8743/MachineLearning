## Overview
This repository implements a full end-to-end pipeline for predicting residual bullet velocity after penetration through different laminates, using both neural networks (with automated hyperparameter tuning) and Random Forests.  It includes data preprocessing, Hyperband‐driven tuning, model fine-tuning, per-laminate specialization, evaluation scripts, and a suite of customizable plotting utilities.

## Features
- **Data preprocessing**: one-hot encoding of categorical features and MinMax scaling  
- **Hyperparameter search**: Keras-Tuner Hyperband with configurable search space and budgets  
- **Neural network training**: build, tune, and fine-tune a global model on train+val data  
- **Random Forest baseline**: train & evaluate alongside the NN  
- **Per-laminate specialization**: clone the global NN, freeze early layers, and fine-tune on each laminate subset  
- **Evaluation**: compute MRE, R², MSE for both NN and RF  
- **Shot selection**: pick representative or all simulation “shots” (Projectile, Angle, Laminate, Velocity)  
- **Plotting utilities**: scatter plots, residual histograms, error vs. velocity, time-series (single plots, subplots by projectile/angle, with configurable scaling and formatting)

## Prerequisites
- **Python 3.8+**  
- **Required packages**:
  - numpy  
  - pandas  
  - scikit-learn  
  - matplotlib  
  - tensorflow  
  - keras-tuner  
  - joblib  

Install with:
```bash
pip install numpy pandas scikit-learn matplotlib tensorflow keras-tuner joblib
