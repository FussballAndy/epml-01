import numpy as np

def comp_mse(y, tx, w):
    """Calculate the mse.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: numpy array of shape(D,)

    Returns:
        Scalar
    """
    return 1/2 * np.mean((y - tx @ w) ** 2)

def comp_gradient(y, tx, w):
    """Compute the gradient.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: numpy array of shape(D,)
    
    Returns:
        grad: shape (D, )
        err: shape (N,)
    """

    err = y - tx.dot(w)
    grad = -(tx.T @ err) / len(err)
    return grad, err

def sigmoid(t):
    """Compute the sigmoid.
    
        Args:
            t: numpy array of shape (N,), N is the number of samples.
    
        Returns:
            A numpy array of shape (N, ) containing the calculated sigmoid for each value of t.
    """
    return 1.0 / (1 + np.exp(-t))

def log_loss(y, tx, w):
    """Compute the log loss for logistic regression.
    
        Args:
            y: numpy array of shape (N,), N is the number of samples.
            tx: numpy array of shape (N,D), D is the number of features.
            w: numpy array of shape(D,)
    
        Returns:
            A tuple containing the predicted values and the loss at w.
    """
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    loss = np.squeeze(-loss).item() * (1 / y.shape[0])

    return pred, loss


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Perform gradient descent using the mean squared error.
    
        Args:
            y: numpy array of shape (N,), N is the number of samples.
            tx: numpy array of shape (N,D), D is the number of features.
            initial_w: numpy array of shape(D,) denoting the initialization of the model parameters
            max_iters: scalar indicating the maximum number of iterations
            gamma: scalar denoting the step size
    
        Returns:
            A tuple containing the last weight vector w and its loss.
    """
    loss = 0.0
    w = initial_w
    for _ in range(max_iters):
        grad, _ = comp_gradient(y, tx, w)
        w = w - gamma * grad
    
    loss = comp_mse(y, tx, w)
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Perform stochastic gradient descent using the mean squared error.
    
        Args:
            y: numpy array of shape (N,), N is the number of samples.
            tx: numpy array of shape (N,D), D is the number of features.
            initial_w: numpy array of shape(D,) denoting the initialization of the model parameters
            max_iters: scalar indicating the maximum number of iterations
            gamma: scalar denoting the step size
        
        Returns:
            A tuple containing the last weight vector w and its loss.
    """
    N = tx.shape[0]

    # Sampling with replacement
    rng = np.random.default_rng(seed=12345)

    w = initial_w
    loss = 0.0

    batch_size = 1

    for n_iter in range(max_iters):
        index = int(rng.integers(low=0, high=N))
        tx_batch = tx[index:index+batch_size]
        y_batch = y[index:index+batch_size]

        grad, _ = comp_gradient(y_batch, tx_batch, w)
        w = w - gamma * grad

    loss = comp_mse(y, tx, w)

    return w, loss

def least_squares(y, tx):
    """Perform least squares regression using normal equations.
        
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        A tuple containing the weight vector w and its mse.
    """
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    mse = comp_mse(y, tx, w)
    return w, mse

def ridge_regression(y, tx, lambda_):
    """Perform ridge regression.

        Args:
            y: numpy array of shape (N,), N is the number of samples.
            tx: numpy array of shape (N,D), D is the number of features.
            lambda_: regularization parameter
        
        Returns:
            A tuple containing the weight vector w and its mse.
    """
    N = tx.shape[0]
    D = tx.shape[1]
    aI = 2 * N * lambda_ * np.identity(D)
    a = tx.T @ tx + aI
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    mse = comp_mse(y, tx, w)
    return w, mse

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Perform logistic regression.
                
        Args:
            y: numpy array of shape (N,), N is the number of samples.
            tx: numpy array of shape (N,D), D is the number of features.
            initial_w: numpy array of shape(D,) denoting the initialization of the model parameters
            max_iters: scalar indicating the maximum number of iterations
            gamma: scalar denoting the step size
        
        Returns:
            A tuple containing the last weight vector w and its log loss.
    """
    threshold = 1e-8

    w = initial_w
    losses = []
    for iter in range(max_iters):
        pred, loss = log_loss(y, tx, w)
        grad = tx.T.dot(pred - y) * (1 / y.shape[0])
        w = w - gamma * grad

        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    _, loss = log_loss(y, tx, w)
    
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Perform regularized logistic regression.
            
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: regularization parameter
        initial_w: numpy array of shape(D,) denoting the initialization of the model parameters
        max_iters: scalar indicating the maximum number of iterations
        gamma: scalar denoting the step size
    
    Returns:
        A tuple containing the last weight vector w and its log loss.
    """
    threshold = 1e-8
    
    w = initial_w
    losses = []
    for iter in range(max_iters):
        pred, loss = log_loss(y, tx, w)
        grad = tx.T.dot(pred - y) * (1 / y.shape[0]) + 2 * lambda_ * w
        w = w - gamma * grad

        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    _, loss = log_loss(y, tx, w)
    
    return w, loss