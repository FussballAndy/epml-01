import numpy as np

from config import MAX_ITERS, GAMMA, SUBMISSION_PATH, PREPROCESSING
from helpers import load_csv_data, create_csv_submission
from implementations import logistic_regression, sigmoid
from labels import x_labels, column_specs
from preprocessing import recode, preprocess

x_train, x_test, y_train, train_ids, test_ids = load_csv_data("./")

x_train, names, categorical = recode(x_train, x_labels, column_specs)
x_test, _, _ = recode(x_test, x_labels, column_specs)

# Logistic regression expects labels in {0, 1}
y = (y_train == 1).astype(float)

# Train on the whole training set, preprocess the test set with its statistics
tx_train, tx_test = preprocess(x_train, x_test, categorical, **PREPROCESSING)
w, loss = logistic_regression(y, tx_train, np.zeros(tx_train.shape[1]), MAX_ITERS, GAMMA)

# Submission expects labels in {-1, 1}
y_test = np.where(sigmoid(tx_test @ w) >= 0.5, 1, -1)
create_csv_submission(test_ids, y_test, SUBMISSION_PATH)
print(f"Final model: train loss={loss:.4f}, {np.mean(y_test == 1):.2%} of test samples predicted positive")
print(f"Predictions written to {SUBMISSION_PATH}")
