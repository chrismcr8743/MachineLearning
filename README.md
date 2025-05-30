# Real-Time Projectile Penetration Prediction

This project implements and evaluates machine-learning surrogates for predicting the full velocity-vs-time history of a projectile penetrating multi-layer steel–polyurea laminates. The dataset (3 042 runs × 20 time samples) was generated via high-fidelity finite-element simulations, covering:

- **Projectile shapes:** Flat, Jacket, Pointed, Round  
- **Impact angles:** 0°, 30°, 45°  
- **Initial speeds:** 50–750 m/s (13 discrete values)  
- **Laminate stacks:** Steel-only, Polyurea–Steel, Steel–Polyurea  

## Methods

1. **Feed-Forward Neural Network**  
   - 5 dense layers (tuned via Hyperband + Keras Tuner)  
   - Input: one-hot encoded categorical features + min–max-scaled continuous variables  
   - Achieved ~9.95 % mean relative error (R² ≈ 0.975)

2. **Random Forest Regressor**  
   - Max depth = 10, no additional tuning  
   - Achieved ~8.9 % mean relative error (R² ≈ 0.996)

## Key Results

- Random Forest outperforms the neural network by ≈1 pp in error and is instantaneous to train and predict.  
- Both models exhibit the highest errors in the 200–300 m/s “near-punch-through” regime.  
- Per-stack retraining of the neural network yields a modest error reduction (~0.5 pp).

![velvtime_jacket_pb_45deg_v0=200](https://github.com/user-attachments/assets/25491127-7a46-4307-926b-76a629b7d9af)
![rf_actual_vs_pred](https://github.com/user-attachments/assets/92e47100-365d-445c-b6b8-e1e5c7278b9b)
![nn_actual_vs_pred](https://github.com/user-attachments/assets/2740a51f-eec1-415d-8f9d-ba3e48aa929b)

## Future Directions

- **LSTM-based modeling** of the time-series data to capture temporal dependencies more explicitly.  
- Augment the dataset with additional angles and materials; benchmark surrogate speed and accuracy against the FE solver.  
- Package the best-performing surrogate as an API for integration into real-time design workflows.


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
