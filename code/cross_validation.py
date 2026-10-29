import numpy as np

from config import FOLDS, MAX_ITERS, GAMMA, SEED, PREPROCESSING, THRESHOLD
from helpers import load_csv_data
from implementations import logistic_regression, log_loss, sigmoid
from evaluation_metrics import binary_evaluation
from labels import x_labels, column_specs
from preprocessing import recode, preprocess


def cross_validation(y, x, categorical, k_indices, k):
    """Train logistic regression on all folds except fold k and evaluate on fold k.

    Args:
        y: numpy array of shape (N,) with labels in {0, 1}
        x: numpy array of shape (N, D), recoded features
        categorical: boolean array of shape (D,) marking the categorical columns
        k_indices: list of index arrays, one per fold
        k: index of the validation fold

    Returns:
        A tuple of the training loss, the validation loss and a dictionary of validation metrics.
    """
    te_idx = k_indices[k]
    tr_idx = np.concatenate([k_indices[i] for i in range(len(k_indices)) if i != k])

    tx_tr, tx_te = preprocess(x[tr_idx], x[te_idx], categorical, **PREPROCESSING)
    y_tr, y_te = y[tr_idx], y[te_idx]

    initial_w = np.zeros(tx_tr.shape[1])
    w, loss_tr = logistic_regression(y_tr, tx_tr, initial_w, MAX_ITERS, GAMMA)
    _, loss_te = log_loss(y_te, tx_te, w)

    y_pred = (sigmoid(tx_te @ w) >= THRESHOLD).astype(int)
    return loss_tr, loss_te, binary_evaluation(y_te, y_pred)


if __name__ == "__main__":
    x_train, _, y_train, _, _ = load_csv_data("./")

    # Codebook based recoding does not depend on the data, so it is done once for all folds
    x_train, names, categorical = recode(x_train, x_labels, column_specs)

    # Logistic regression expects labels in {0, 1}
    y = (y_train == 1).astype(float)

    # Shuffle before splitting so each fold is a random sample
    rng = np.random.default_rng(SEED)
    k_indices = np.split(rng.permutation(len(y)), FOLDS)

    results = []
    for k in range(FOLDS):
        loss_tr, loss_te, metrics = cross_validation(y, x_train, categorical, k_indices, k)
        results.append((loss_tr, loss_te, metrics))
        print(
            f"Fold {k + 1:2d}/{FOLDS}: train loss={loss_tr:.4f}, val loss={loss_te:.4f}, "
            + ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
        )

    print(f"\nMean over {FOLDS} folds (NaN strategy: {PREPROCESSING['strategy']}):")
    print(f"  train loss: {np.mean([r[0] for r in results]):.4f}")
    print(f"  val loss:   {np.mean([r[1] for r in results]):.4f}")
    for name in results[0][2]:
        values = [r[2][name] for r in results]
        # nanmean: precision/f1 are undefined for folds without any positive prediction
        print(f"  {name}: {np.nanmean(values):.4f} ± {np.nanstd(values):.4f}")
