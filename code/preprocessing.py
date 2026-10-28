import numpy as np

from implementations import standardize


def recode(x, labels, specs):
    """Apply the codebook: replace special codes, decode rate encoded columns and drop unused columns.

    This only depends on the codebook and not on the data, so it can be applied once to the whole
    dataset before splitting into folds.

    Args:
        x: numpy array of shape (N, D) with the raw features
        labels: list of D column names
        specs: dict mapping a column name to its spec, see labels.column_specs

    Returns:
        A tuple (x, names, categorical) where x has shape (N, D'), names are the D' remaining column
        names and categorical is a boolean array of shape (D',) marking columns to one-hot encode.
    """
    columns, names, categorical = [], [], []
    for d, name in enumerate(labels):
        spec = specs[name]
        if spec["type"] == "drop":
            continue

        raw = x[:, d]
        col = raw.copy()
        if "units" in spec:
            factor = np.array([spec["units"].get(int(v), np.nan) for v in np.nan_to_num(raw // 100)])
            col = (raw % 100) * factor
        if "missing" in spec:
            col[np.isin(raw, spec["missing"])] = np.nan
        if "zero" in spec:
            col[np.isin(raw, spec["zero"])] = 0

        columns.append(col)
        names.append(name)
        categorical.append(spec["type"] == "categorical")

    return np.column_stack(columns), names, np.array(categorical)


def drop_sparse_columns(x_tr, x_te, mask, max_nan_fraction):
    """Drop columns whose fraction of NaNs in the training data exceeds max_nan_fraction.

    Args:
        x_tr: numpy array of shape (N_tr, D)
        x_te: numpy array of shape (N_te, D)
        mask: boolean array of shape (D,) describing the columns (e.g. categorical), filtered alongside
        max_nan_fraction: scalar in [0, 1]

    Returns:
        A tuple (x_tr, x_te, mask) with the sparse columns removed.
    """
    keep = np.mean(np.isnan(x_tr), axis=0) <= max_nan_fraction
    return x_tr[:, keep], x_te[:, keep], mask[keep]


def clip_outliers(x_tr, x_te, columns, low, high, min_unique=10):
    """Clip columns to the [low, high] percentiles of the training data.

    Only columns with at least min_unique distinct values are clipped, so that rare answers of
    binary or ordinal columns are not merged into the common ones.

    Args:
        x_tr: numpy array of shape (N_tr, D)
        x_te: numpy array of shape (N_te, D)
        columns: boolean array of shape (D,) marking the columns that may be clipped
        low, high: percentiles in [0, 100]
        min_unique: minimum number of distinct values for a column to be clipped

    Returns:
        A tuple (x_tr, x_te) of clipped copies.
    """
    x_tr, x_te = x_tr.copy(), x_te.copy()
    for d in np.flatnonzero(columns):
        col = x_tr[:, d]
        if np.unique(col[~np.isnan(col)]).size < min_unique:
            continue
        lo, hi = np.nanpercentile(col, [low, high])
        x_tr[:, d] = np.clip(col, lo, hi)
        x_te[:, d] = np.clip(x_te[:, d], lo, hi)
    return x_tr, x_te


def missing_indicators(x_tr, x_te, columns):
    """Build a 0/1 column per feature telling whether the value was missing.

    Only columns that contain NaNs in the training data get an indicator.

    Args:
        x_tr: numpy array of shape (N_tr, D)
        x_te: numpy array of shape (N_te, D)
        columns: boolean array of shape (D,) marking the columns to consider

    Returns:
        A tuple (ind_tr, ind_te) of shapes (N_tr, K) and (N_te, K).
    """
    has_nan = columns & np.any(np.isnan(x_tr), axis=0)
    return np.isnan(x_tr[:, has_nan]).astype(float), np.isnan(x_te[:, has_nan]).astype(float)


def one_hot_encode(x_tr, x_te, categorical):
    """Replace categorical columns by one 0/1 column per category seen in the training data.

    NaNs and categories not seen in training are encoded as all zeros.

    Args:
        x_tr: numpy array of shape (N_tr, D)
        x_te: numpy array of shape (N_te, D)
        categorical: boolean array of shape (D,) marking the columns to encode

    Returns:
        A tuple (x_tr, x_te) where the numeric columns come first, followed by the one-hot columns.
    """
    enc_tr, enc_te = [x_tr[:, ~categorical]], [x_te[:, ~categorical]]
    for d in np.flatnonzero(categorical):
        col = x_tr[:, d]
        values = np.unique(col[~np.isnan(col)])
        enc_tr.append((x_tr[:, d, None] == values).astype(float))
        enc_te.append((x_te[:, d, None] == values).astype(float))
    return np.hstack(enc_tr), np.hstack(enc_te)


def nan_fill_values(x, strategy):
    """Compute the per-column value used to replace NaNs.

    Args:
        x: numpy array of shape (N, D), may contain NaNs
        strategy: one of "mean", "median", "zero", "mode"

    Returns:
        numpy array of shape (D,). Columns that are entirely NaN get NaN.
    """
    if strategy == "mean":
        return np.nanmean(x, axis=0)
    if strategy == "median":
        return np.nanmedian(x, axis=0)
    if strategy == "zero":
        fill = np.zeros(x.shape[1])
        fill[np.all(np.isnan(x), axis=0)] = np.nan
        return fill
    if strategy == "mode":
        fill = np.full(x.shape[1], np.nan)
        for d in range(x.shape[1]):
            col = x[~np.isnan(x[:, d]), d]
            if col.size > 0:
                values, counts = np.unique(col, return_counts=True)
                fill[d] = values[np.argmax(counts)]
        return fill
    raise ValueError(f"Unknown NaN strategy: {strategy}")


def fill_nans(x_tr, x_te, strategy):
    """Replace NaNs with values computed on the training data.

    Columns without any value in the training data cannot be filled and are dropped.

    Args:
        x_tr: numpy array of shape (N_tr, D)
        x_te: numpy array of shape (N_te, D)
        strategy: NaN fill strategy, see nan_fill_values

    Returns:
        A tuple (x_tr, x_te) without NaNs.
    """
    fill = nan_fill_values(x_tr, strategy)
    keep = ~np.isnan(fill)
    x_tr, x_te, fill = x_tr[:, keep], x_te[:, keep], fill[keep]

    x_tr = np.where(np.isnan(x_tr), fill, x_tr)
    x_te = np.where(np.isnan(x_te), fill, x_te)
    return x_tr, x_te


def preprocess(x_tr, x_te, categorical, strategy, max_nan_fraction=1.0, clip_percentiles=None,
               add_missing_indicators=False, one_hot=False):
    """Drop sparse columns, clip outliers, encode categories, fill NaNs, standardize and add a bias column.

    All statistics are computed on x_tr only, so nothing leaks into x_te.

    Args:
        x_tr: numpy array of shape (N_tr, D), recoded training features
        x_te: numpy array of shape (N_te, D), features transformed with the training statistics
        categorical: boolean array of shape (D,) marking the categorical columns
        strategy: NaN fill strategy, see nan_fill_values
        max_nan_fraction: drop columns with more missing values than this, 1.0 keeps all
        clip_percentiles: tuple (low, high) of percentiles to clip to, None disables clipping
        add_missing_indicators: whether to add a 0/1 column per feature marking missing values
        one_hot: whether to one-hot encode the categorical columns

    Returns:
        A tuple (tx_tr, tx_te).
    """
    x_tr, x_te, categorical = drop_sparse_columns(x_tr, x_te, categorical, max_nan_fraction)

    if clip_percentiles is not None:
        x_tr, x_te = clip_outliers(x_tr, x_te, ~categorical, *clip_percentiles)

    # Missing categorical values are already encoded by the one-hot columns being all zero
    if add_missing_indicators:
        ind_tr, ind_te = missing_indicators(x_tr, x_te, ~categorical if one_hot else np.ones_like(categorical))

    if one_hot:
        x_tr, x_te = one_hot_encode(x_tr, x_te, categorical)

    x_tr, x_te = fill_nans(x_tr, x_te, strategy)

    if add_missing_indicators:
        x_tr, x_te = np.hstack([x_tr, ind_tr]), np.hstack([x_te, ind_te])

    x_tr, x_te = standardize(x_tr, x_te)

    tx_tr = np.c_[np.ones(x_tr.shape[0]), x_tr]
    tx_te = np.c_[np.ones(x_te.shape[0]), x_te]
    return tx_tr, tx_te
