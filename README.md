## Overview  
This repository implements an end-to-end pipeline for **real-time prediction of a projectile’s entire velocity profile**—both before and after penetration—through steel/polyurea laminates.  A Keras-based neural network (with automated Hyperband tuning) and a Random Forest baseline both learn from finite-element simulation data **provided by the author of “Real-time prediction of projectile penetration to laminates…”** (Pushkar Wadagbalkar and G.R. Liu, *Defence Technology* 2021).  The goal is to outperform the original models in that paper.

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
