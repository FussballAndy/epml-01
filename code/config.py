# Settings shared by cross_validation.py and run.py

FOLDS = 29 # Extracted by checking for valid divisor
MAX_ITERS = 300
GAMMA = 0.5
SEED = 12345
SUBMISSION_PATH = "submission.csv"

# Keyword arguments for preprocessing.preprocess
PREPROCESSING = {
    "strategy": "zero", # One of "mean", "median", "zero", "mode"
    "max_nan_fraction": 0.9, # Drop columns with more missing values than this, 1.0 keeps all
    "clip_percentiles": (1, 99), # None disables clipping
    "add_missing_indicators": True,
    "one_hot": True,
}
