# Settings shared by cross_validation.py and run.py

FOLDS = 29 # Extracted by checking for valid divisor
MAX_ITERS = 300
GAMMA = 0.5
SEED = 12345
SUBMISSION_PATH = "submission.csv"
THRESHOLD = 0.2

# Keyword arguments for preprocessing.preprocess
PREPROCESSING = {
    # One of "mean", "median", "zero", "mode"
    "strategy": "zero",
    # Drop columns with more missing values than this, 1.0 keeps all
    "max_nan_fraction": 0.9, 
    # Clip values to these percentiles, or None for no clipping
    "clip_percentiles": (1, 99), 
    # Add indicator columns on whether the data was actually present or NaN
    "add_missing_indicators": True,
    # One hot encode columns marked as categorial
    "one_hot": True,
}
