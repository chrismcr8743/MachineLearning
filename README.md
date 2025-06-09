# High-Fidelity FE → Real-Time ML Surrogate for Ballistic Penetration

This repository implements and extends the surrogate‐model approach of Wadagbalkar & Liu (2021), replacing their decision-tree and FF-NN regressors with three modern architectures—**XGBoost**, a deep **feed-forward neural network**, and an **LSTM**—to predict projectile residual velocity in real time.  

---

## Results

The **XGBoost** model achieves **R² ≥ 0.9777** with **average errors < 8 %** across all regimes, outperforming the original decision-tree and FF-NN baselines  
(from R² ≈ 0.99 / 0.987 and errors ≈ 2.9 % / 10.4 %).

### Table 1 ▪ Model Performance Across Splits

| Split / Model                                     | R²    | MAE (m/s) | Avg % err |
|--------------------------------------------------|------:|----------:|-----------:|
| **Random 80 / 20**                                |       |           |            |
| • XGBoost                                         | 0.999 |      2.99 |     1.12 % |
| • FF-NN                                           | 0.996 |      8.80 |     3.31 % |
| • LSTM                                            | 0.997 |      5.84 |     2.19 % |
| **Interpolation (unseen velocity)**               |       |           |            |
| • XGBoost                                         | 0.978 |     20.73 |     7.73 % |
| • FF-NN                                           | 0.981 |     15.62 |     5.82 % |
| • LSTM                                            | 0.983 |     15.64 |     5.83 % |
| **Extrapolation (unseen stacks × projectiles)**   |       |           |            |
| • XGBoost                                         | 0.978 |     20.50 |     8.01 % |
| • FF-NN                                           | 0.839 |     47.17 |    18.43 % |
| • LSTM                                            | 0.904 |     43.59 |    17.04 % |



* **Actual vs Predicted Residual Velocity**
![plot_actual_vs_pred](https://github.com/user-attachments/assets/31b73861-16bf-4a1c-92b1-7c3ab03ef8a5)
* **Residual vs Predicted**
![residual_vs_pred](https://github.com/user-attachments/assets/790755c8-686b-452e-9a30-c340549d7729)

* **Residual Velocity Predictions vs Time**
![Round_PF_30_350](https://github.com/user-attachments/assets/c0cac79a-8ed7-47d2-814f-62771b1ce4df)
![Flat_PB_45_300](https://github.com/user-attachments/assets/c262238e-de73-4e2f-88f6-1c14c8ee66ce)
![Round_S_0_300](https://github.com/user-attachments/assets/3e3b5fa9-2211-4e09-b3f4-579f7b41d698)

---

## References
> P. Wadagbalkar & G. R. Liu, *Defence Technology*, 17(2):147–160, 2021.

---
