import numpy as np

def comp_mse(y, tx, w):
    return 1/2 * np.mean(np.sum((y - tx @ w) ** 2))

def comp_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -(tx @ err) / len(err)
    return grad, err

def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    loss = 0.0
    w = initial_w
    for _ in range(max_iters):
        grad, err = comp_gradient(y, tx, w)
        loss = comp_mse(err)
        w = w - gamma * grad
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    # currently does not pick feature to traverse
    w = initial_w
    loss = 0.0

    for n_iter in range(max_iters):
        grad, _ = comp_gradient(y, tx, w)
        w = w - gamma * grad
        loss = comp_mse(y, tx, w)

    return w, loss

def least_squares(y, tx):
    a = tx @ tx
    b = tx @ y
    w = np.linalg.solve(a, b)
    mse = comp_mse(y, tx, w)
    return w, mse

def ridge_regression(y, tx, lambda_):
    N = tx.shape[0]
    D = tx.shape[1]
    aI = 2 * N * lambda_ * np.identity(D)
    a = tx @ tx + aI
    b = tx @ y
    w = np.linalg.solve(a, b)
    mse = comp_mse(y, tx, w)
    return w, mse

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    threshold = 1e-8

    w = initial_w
    losses = []
    for iter in range(max_iters):
        pred = sigmoid(tx.dot(w))
        loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
        loss = np.squeeze(-loss).item() * (1 / y.shape[0])
        grad = tx.T.dot(pred - y) * (1 / y.shape[0])
        w -= gamma * grad

        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return w, losses[-1]

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    threshold = 1e-8
    
    w = initial_w
    losses = []
    for iter in range(max_iters):
        pred = sigmoid(tx.dot(w))
        loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
        loss = np.squeeze(-loss).item() * (1 / y.shape[0])
        grad = tx.T.dot(pred - y) * (1 / y.shape[0]) + 2 * lambda_ * w
        w -= gamma * grad

        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return w, losses[-1]