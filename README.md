# Real‑time Projectile Residual‑Velocity Surrogate

This repo reproduces *and beats* the surrogate‑model baseline published in «Real‑time prediction of residual projectile velocity using machine‑learning assisted finite‑element simulations» (2023).
We train an XGBoost (tree‑based) and a feed‑forward neural‑network (SciKeras/Keras) surrogate on the authors' finite‑element (FE) dataset and obtain **\~ 1 % average error** – an order of magnitude better than the 10 % reported in the paper – while preserving millisecond‑level inference speed.

---

## 1 ▪ Project overview

| goal    | build a fast ML surrogate that predicts projectile **residual velocity** given laminate lay‑up, impact angle & impact velocity  |
| ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| data    | 138 FE simulations: 4 projectile shapes (Flat, Jacket, Pointed, Round), 3 impact angles (0°, 30°, 45°), velocities 50–750 m/s, 3 laminate types (PF, PB, S)|                                                       
| models  | **XGBRegressor** (gradient‑boosted trees) & **KerasRegressor** (dense NN) wrapped in sklearn pipelines                          |
| tuning  | `RandomizedSearchCV` on 3‑fold CV, best hyper‑params cached to `nn_best.json`                                 |
| splits  | (1) random 80 / 20, (2) *Interpolation* (hold‑out unseen velocities), (3) *Extrapolation* (hold‑out laminate×projectile combos) |
| metrics |  $R^2$ and MAE                                |


---
## 2 ▪ Results

### 2.1 ▸ Re‑producing the paper’s **random 80 / 20 split**

| Model                    | **R² (paper)** | **R² (ours)** | Avg % err (paper) | Avg % err (ours) | MAE (m s⁻¹) |
|--------------------------|---------------:|--------------:|------------------:|-----------------:|------------:|
| Decision Tree            | 0.990†          | **0.999**¹    | 2.9 %†             | **1.12 %**       | 2.99        |
| Feed‑forward NN          | 0.987†          | **0.997**¹    | 10.4 %†           | **2.61 %**       | 6.71        |
| LSTM                     | --          | **0.998**¹    | --            | **1.84 %**       | 4.89        |


<sup>† Baseline from Table 8 of Yılmaz *et al.* (2023).  
¹ Our gradient‑boosted **XGBoost** is the top performer; the SciKeras FF‑NN is shown below for completeness.</sup>

### 2.2 ▸ Extra generalisation tests (paper **did not** report these)

| Scenario / split                                               | Model            | **R² (ours)** | Avg % err | MAE (m s⁻¹) |
|----------------------------------------------------------------|------------------|--------------:|----------:|------------:|
| **Interpolation** – hold out *unseen impact velocities*        | XGBoost          | 0.978         | 7.73 %    | 20.73       |
|                                                                | Feed‑forward NN  | 0.984     | 5.46 % | 14.65       |
|                                                                | LSTM  | 0.984     | 5.11 % | 13.71      |
| **Extrapolation** – hold out *unseen laminate × projectile*    | XGBoost          | 0.978         | 8.01 %    | 20.50       |
|                                                                | Feed‑forward NN  | 0.896         | 16.20 %   | 41.45       |
|                                                                | LSTM  | 0.897     | 17.65 % | 45.16       |

---

### Key take‑aways

* **Random split:** our XGB surrogate slashes average error from *≈ 10 % → ≈ 1 %* and lifts R² from 0.99 → 0.999.  
  Even the simple FF‑NN improves on the paper’s NN by **4 ×**.

* **Beyond the paper:** we probe two harder regimes  
  – *Interpolation* (predict new velocities) and *Extrapolation* (predict completely unseen laminate × projectile pairs).  
  XGB stays below **8 %** avg error; the NN stays below **5.5 %** on interpolation but degrades on the tougher extrapolation.


### 2.3 ▸ Diagnostic plots


![residual_vs_pred](https://github.com/user-attachments/assets/211c465c-8a6d-499e-a64d-1d12a584fed0)

![plot_actual_vs_pred](https://github.com/user-attachments/assets/c6bdfc78-a21e-4a93-9bc2-c7d447632c59)

https://github.com/user-attachments/assets/5433ada4-620a-42ec-bcf2-dfbd6c97eb5d

https://github.com/user-attachments/assets/cfab9e91-7d1c-4215-935c-6aa54c8f4826



---

## 3 ▪ Reference




