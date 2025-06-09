from pathlib import Path
DATA_PATH    = Path("data/Finite_element_entire_data_set.csv")
TARGET       = "Residual Velcity"
CACHE_PATH   = Path("models/nn_best.json")
SEARCH_ITERS = 10
NUM_COLS     = ["Time", "Velocity", "vel_sq", "time_sq", "vel_time_interaction"]
CAT_COLS     = ["Projectile", "Laminate", "Angle"]