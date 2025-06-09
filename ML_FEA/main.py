import sys
sys.path.append('.')  
import copy, pathlib, numpy as np, pandas as pd
import tensorflow as tf
tf.get_logger().setLevel("ERROR")          
from figs.plotters import plot_actual_vs_pred, plot_actual_vs_pred, plot_residual_vs_pred, save_all_shot_time_series
from models.model_utils import engineer, build_xgb, get_nn_pipeline, evaluate, get_lstm_pipeline
from collections import defaultdict
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from config import DATA_PATH, TARGET
from tabulate import tabulate 

# Load data and perform feature engineering
df               = engineer(pd.read_csv(DATA_PATH))                      
df["combo_key"]  = df["Projectile"] + "_" + df["Laminate"]            # group id for extrap split
df["config_key"] = df["combo_key"]  + "_" + df["Angle"].astype(str)   # id incl. angle for interp
X_all, y_all     = df.drop(columns=[TARGET]), df[TARGET]              # full feature / target sets
results          = defaultdict(dict)                                  # dict to collect metrics

############################################################################################################
# Perform 80/20 random split - baseline
############################################################################################################
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

xgb_base                   = build_xgb().fit(X_tr, y_tr)# train XGB with CV search
results["RandomSplit‑XGB"] = evaluate(xgb_base.best_estimator_, X_te, y_te)

nn_base                   = get_nn_pipeline(X_tr, y_tr)
results["RandomSplit‑NN"] = evaluate(nn_base, X_te, y_te)

lstm_base                   = get_lstm_pipeline(X_tr, y_tr)       
results["RandomSplit‑LSTM"] = evaluate(lstm_base, X_te, y_te)


############################################################################################################
# Interpolation - unseen velocities
############################################################################################################
mask = pd.Series(False, index=df.index)
rng  = np.random.default_rng(42)# reproducible RNG
for _, g in df.groupby("config_key"):# iterate each stack angle cfg
    mask.loc[g.index[g["Velocity"] == rng.choice(g["Velocity"].unique())]] = True

itr_tr, itr_te = df[~mask], df[mask]# split by mask

xgb_i                 = build_xgb().fit(itr_tr.drop(columns=[TARGET]), itr_tr[TARGET])
results["Interp‑XGB"] = evaluate(xgb_i.best_estimator_, itr_te.drop(columns=[TARGET]), itr_te[TARGET])

nn_i                 = get_nn_pipeline(itr_tr.drop(columns=[TARGET]), itr_tr[TARGET])
results["Interp‑NN"] = evaluate(nn_i, itr_te.drop(columns=[TARGET]), itr_te[TARGET])

lstm_i                 = get_lstm_pipeline(itr_tr.drop(columns=[TARGET]), itr_tr[TARGET])
results["Interp‑LSTM"] = evaluate(lstm_i, itr_te.drop(columns=[TARGET]), itr_te[TARGET])


############################################################################################################
# Extrapolation - unseen laminate × projectile
############################################################################################################
gss                = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # hold out full combos
tr_idx, te_idx     = next(gss.split(df, groups=df["combo_key"]))                   # split indices
extra_tr, extra_te = df.iloc[tr_idx], df.iloc[te_idx]                              # build sets

xgb_e                 = build_xgb().fit(extra_tr.drop(columns=[TARGET]), extra_tr[TARGET])
results["Extrap‑XGB"] = evaluate(xgb_e.best_estimator_, extra_te.drop(columns=[TARGET]), extra_te[TARGET])

nn_e                 = get_nn_pipeline(extra_tr.drop(columns=[TARGET]), extra_tr[TARGET])
results["Extrap‑NN"] = evaluate(nn_e, extra_te.drop(columns=[TARGET]), extra_te[TARGET])

lstm_e                 = get_lstm_pipeline(extra_tr.drop(columns=[TARGET]), extra_tr[TARGET])
results["Extrap‑LSTM"] = evaluate(lstm_e, extra_te.drop(columns=[TARGET]), extra_te[TARGET])


############################################################################################################
# Add predictions to the main dataframe for plotting
############################################################################################################
df["Pred_XGB"]  = xgb_base.best_estimator_.predict(X_all) 
df["Pred_NN"]   = nn_base.predict(X_all).ravel()
df["Pred_LSTM"] = lstm_base.predict(X_all).ravel()
models_dict     = {"XGB": "Pred_XGB", 
                   "NN": "Pred_NN", 
                   "LSTM": "Pred_LSTM"}  

#----------------------------------------------------------------------------------------------------------
# Plot the results 
#----------------------------------------------------------------------------------------------------------
plot_actual_vs_pred(models_dict, df)  
plot_residual_vs_pred(models_dict, df)
save_all_shot_time_series(df, models_dict)


############################################################################################################
# Print table with the performance metrics 
############################################################################################################
rows = []
for split, mets in results.items():
    rows.append([
        split,
        f"{mets['R2']:.4f}",
        f"{mets['MAE']:.2f}",
        f"{mets['AvgPct']:.2f} %"])
print(tabulate(rows,headers=["Split", "R²", "MAE", "Avg % err"],tablefmt="github"))