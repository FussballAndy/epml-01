import numpy as np

from config import MAX_ITERS, GAMMA, SUBMISSION_PATH, PREPROCESSING, THRESHOLD
from helpers import load_csv_data, create_csv_submission
from implementations import logistic_regression, sigmoid
from labels import x_labels, column_specs
from preprocessing import recode, preprocess

x_train, x_test, y_train, train_ids, test_ids = load_csv_data("./")

x_train, names, categorical = recode(x_train, x_labels, column_specs)
x_test, _, _ = recode(x_test, x_labels, column_specs)

# Convert {-1,1} labels to {0,1}
y = (y_train == 1).astype(float)

tx_train, tx_test = preprocess(x_train, x_test, categorical, **PREPROCESSING)
w, loss = logistic_regression(y, tx_train, np.zeros(tx_train.shape[1]), MAX_ITERS, GAMMA)

# Convert labels back to {-1, 1}
y_test = np.where(sigmoid(tx_test @ w) >= THRESHOLD, 1, -1)
create_csv_submission(test_ids, y_test, SUBMISSION_PATH)
print(f"Final model: train loss={loss:.4f}, {np.mean(y_test == 1):.2%} of test samples predicted positive")
print(f"Predictions written to {SUBMISSION_PATH}")
