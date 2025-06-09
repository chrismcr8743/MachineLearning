from sklearn.compose import ColumnTransformer
from sklearn.pipeline  import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import json, pathlib, config, numpy as np, pandas as pd, xgboost as xgb
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
from tensorflow.keras import regularizers, activations, optimizers
from scipy.stats import loguniform, randint
import tensorflow as tf
from tensorflow.keras.layers import LSTM
import numpy as np
tf.get_logger().setLevel("ERROR") 

NUM_COLS     = config.NUM_COLS     
CAT_COLS     = config.CAT_COLS
SEARCH_ITERS = config.SEARCH_ITERS
CACHE_PATH   = config.CACHE_PATH


def engineer(df: pd.DataFrame) -> pd.DataFrame:   # create derived features
    df = df.copy()                                              
    df["vel_sq"]               = df["Velocity"] ** 2            
    df["time_sq"]              = df["Time"] ** 2               
    df["vel_time_interaction"] = df["Velocity"] * df["Time"]    
    return df


def make_pre() -> ColumnTransformer:# fresh preprocessor factory
    return ColumnTransformer(
        [("num", StandardScaler(), NUM_COLS),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)]
    )


############################################################################################################
# xgboost
############################################################################################################
def build_xgb() -> RandomizedSearchCV:    # xgboost & hyperparam search
    # preprocessing stage
    pipe = Pipeline([("prep", make_pre()),("model", xgb.XGBRegressor(objective="reg:squarederror", random_state=42))])
    # search space
    param = {  
        "model__n_estimators":      [ 150,  300, 500],
        "model__max_depth":         [   3,    5,   7],
        "model__learning_rate":     [0.01, 0.05, 0.1],
        "model__subsample":         [ 0.6,  0.8, 1.0],
        "model__colsample_bytree":  [ 0.6,  0.8, 1.0]}
    return RandomizedSearchCV(pipe, param, n_iter=20,   
                              scoring="r2", cv=3, random_state=42, n_jobs=-1)


############################################################################################################
# neural network
############################################################################################################
def build_nn(n_feats: int) -> Sequential:    # feed forward NN
    model = Sequential([
        Input(shape=(n_feats,)), 
        Dense(64, activation="relu"), BatchNormalization(), Dropout(0.1),
        Dense(64, activation="relu"), BatchNormalization(), Dropout(0.1),
        Dense(32, activation="relu"), BatchNormalization(), Dropout(0.1),
        Dense(1)])
    model.compile(optimizer="adam", loss="mse") 
    return model


def build_nn_model(meta, n_units=64, n_layers=2, dropout=0.1,
                   activation="relu", l2=0.0, lr=1e-3, **_):
    act  = activations.get(activation)
    reg  = regularizers.l2(l2) if l2 > 0 else None
    model = Sequential()
    model.add(Input(shape=(meta["n_features_in_"],)))
    for _ in range(n_layers):
        model.add(Dense(n_units, activation=act,
                        kernel_regularizer=reg))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=optimizers.get({"class_name":"Adam","config":{"learning_rate":lr}}),
                  loss="mse")
    return model


def build_nn_search():
    pipe = Pipeline([
        ("prep", make_pre()),
        ("nn", KerasRegressor(        
            model=build_nn_model,
            verbose=0,
            epochs=150,
            batch_size=32,
            validation_split=0.1,                
            callbacks=[
                EarlyStopping(
                    monitor="val_loss", patience=15,
                    restore_best_weights=True, verbose=0
                )
            ],
        )),
    ])

    param = {
        "nn__model__n_layers":   randint(1, 5),                 
        "nn__model__n_units":    randint(32, 257),              
        "nn__model__activation": ["relu", "gelu", "elu"],
        "nn__model__dropout":    [0.0, 0.1, 0.25, 0.4, 0.5],
        "nn__model__l2":         loguniform(1e-6, 1e-3),        
        "nn__model__lr":         loguniform(1e-4, 5e-3),        
        "nn__batch_size":        [16, 32, 64, 128],             
        "nn__optimizer":         ["adam", "nadam", "adamw"]}

    return RandomizedSearchCV(
        pipe, param,
        n_iter=10, scoring="r2",
        cv=3, random_state=42, n_jobs=-1,
        error_score="raise")


def get_nn_pipeline(X, y):
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r") as f:
            best_params = json.load(f)

 
        pipe = Pipeline([
            ("prep", make_pre()),
            ("nn", KerasRegressor(
                model       = build_nn_model,
                verbose     = 0,
                epochs      = 150,
                validation_split = 0.1,
                callbacks   = [EarlyStopping(monitor="val_loss",
                                             patience=15,
                                             restore_best_weights=True)],
            ))
        ])

        pipe.set_params(**best_params)
        pipe.fit(X, y)
        return pipe

    search = RandomizedSearchCV(
        build_nn_search().estimator,   
        build_nn_search().param_distributions,
        n_iter        = SEARCH_ITERS,
        scoring       = "r2",
        cv            = 3,
        n_jobs        = -1,
        random_state  = 42,
        error_score   = "raise",
    ).fit(X, y)
 
    with open(CACHE_PATH, "w") as f:
        json.dump(search.best_params_, f, indent=2)

    return search.best_estimator_


def fit_nn(prep, X_tr, y_tr, X_val, y_val): # train NN with early stop
    X_tr = prep.transform(X_tr).astype("float32") # apply encoder → train set
    X_val = prep.transform(X_val).astype("float32") # apply encoder → val set
    es = EarlyStopping(monitor="val_loss", patience=10,
                       restore_best_weights=True, verbose=0) # stop when val loss plateaus
    model = build_nn(X_tr.shape[1])
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=150, batch_size=32, callbacks=[es], verbose=0)
    return model


############################################################################################################
# lstm
############################################################################################################
def build_lstm_model(meta,
                     lstm_units=64,
                     dense_units=32,
                     dropout=0.1,
                     lr=1e-3,
                     **_):
    """SciKeras build_fn producing a simple 1‑layer LSTM → Dense."""
    T, n_feats = meta["X_shape_"][1:]           
    model = Sequential([
        Input(shape=(T, n_feats)),
        LSTM(lstm_units, dropout=dropout, return_sequences=False),
        Dense(dense_units, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizers.Adam(lr), loss="mse")
    return model


def build_lstm_search():
    pipe = Pipeline([
        ("prep", make_pre()),              
        ("reshape", FunctionTransformer(   
            lambda X: X.reshape(X.shape[0], 1, X.shape[1]),
            validate=False, feature_names_out="one-to-one")),
        ("lstm", KerasRegressor(
            model=build_lstm_model,
            epochs=500,
            batch_size=32,
            verbose=0,
            validation_split=0.1,
            callbacks=[EarlyStopping(
                monitor="val_loss", patience=30,
                restore_best_weights=True)]
        ))
    ])

    param = {
        "lstm__model__lstm_units":  [8, 16, 32],
        "lstm__model__dense_units": [16, 32, 64],
        "lstm__model__dropout":     [0.0, 0.1, 0.25],
        "lstm__model__lr":          loguniform(1e-4, 5e-3),
        "lstm__batch_size":         [8, 16],
        "lstm__optimizer":          ["adam", "nadam"],
    }
    return RandomizedSearchCV(pipe, param,
                              n_iter=10, cv=3,
                              scoring="r2", n_jobs=-1,
                              random_state=42)


def get_lstm_pipeline(X, y):
    cache_p = pathlib.Path("models/lstm_best.json")
    if cache_p.exists():
        best_params = json.load(open(cache_p))
        pipe = build_lstm_search().estimator    
        pipe.set_params(**best_params).fit(X, y)
        return pipe

    search = build_lstm_search().fit(X, y)
    json.dump(search.best_params_, open(cache_p, "w"), indent=2)
    return search.best_estimator_



#===========================================================================================================
# Evaluation metrics 
#===========================================================================================================
def evaluate(model, X_te, y_te, *, prep=None):
    if prep is not None:
        X_te = prep.transform(X_te).astype("float32")

    try:
        preds = model.predict(X_te, verbose=0)
    except TypeError:
        preds = model.predict(X_te)

    preds   = np.asarray(preds).ravel()
    r2      = r2_score(y_te, preds)
    mae     = mean_absolute_error(y_te, preds)
    avg_pct = 100.0 * mae / y_te.mean()   

    return {"R2": r2, "MAE": mae, "AvgPct": avg_pct}
