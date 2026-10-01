import numpy as np


def compute_mse_loss(e):
    """
    Calculate the MSE loss

    Args:
        e: np.ndarray
            Error vector
    Returns:
        loss: float
            MSE loss
    """

    n = len(e)
    loss = (1 / (2 * n)) * (e.T @ e)
    return loss


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.

    Args:
        y: np.ndarray
            (nx1) Targets of the model
        tx: np.ndarray
            (nxd) Entries of the model
        initial_w: np.ndarray
            Initial weigths
        max_iters: int
            Max number of iterations for the algorithm
        gamma: float
            Learning rate

    Returns:
        w: np.ndarray
            (dx1) Optimal weights
        loss: float
            Empirical loss for the final model
    """
    w = initial_w
    n = len(y)

    for _ in range(max_iters):
        e = y - tx @ w
        grad = -(1 / n) * tx.T @ e
        w = w - gamma * grad

    e_final = y - tx @ w
    loss = compute_mse_loss(e_final)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.

    Args:
        y: np.ndarray
            (nx1) Targets of the model
        tx: np.ndarray
            (nxd) Entries of the model
        initial_w: np.ndarray
            Initial weigths
        max_iters: int
            Max number of iterations for the algorithm
        gamma: float
            Learning rate

    Returns:
        w: np.ndarray
            (dx1) Optimal weights
        loss: float
            Empirical loss for the final model
    """
    w = initial_w
    n = len(y)

    for _ in range(max_iters):
        # Mini-batch size 1
        i = np.random.randint(0, n)
        x_i = tx[i]
        y_i = y[i]
        e = y_i - x_i @ w
        grad = -x_i * e
        w = w - gamma * grad

    e_final = y - tx @ w
    loss = compute_mse_loss(e_final)
    return w, loss


def least_squares(y, tx):
    """
    Least squares regression using normal equations.
    Args:
        y: np.ndarray
            (nx1) Targets of the model
        tx: np.ndarray
            (nxd) Entries of the model

    Returns:
        w: np.ndarray
            (dx1) Optimal weights
        loss: float
            Empirical loss for the final model
    """
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    e = y - tx @ w
    loss = compute_mse_loss(e)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.
    Args:
        y: np.ndarray
            (nx1) Targets of the model
        tx: np.ndarray
            (nxd) Entries of the model
        lambda_: float
            Regularizer weight

    Returns:
        w: np.ndarray
            (dx1) Optimal weights
        loss: float
            Empirical loss for the final model
    """
    n = len(y)
    d = tx.shape[1]
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    e = y - tx @ w
    loss = compute_mse_loss(e)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.
    Args:
        y: np.ndarray
            (nx1) Targets of the model
        tx: np.ndarray
            (nxd) Entries of the model
        initial_w: np.ndarray
            Initial weigths
        max_iters: int
            Max number of iterations for the algorithm
        gamma: float
            Learning rate

    Returns:
        w: np.ndarray
            (dx1) Optimal weights
        loss: float
            Empirical loss for the final model
    """
    pass


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent.
    Args:
        y: np.ndarray
            (nx1) Targets of the model
        tx: np.ndarray
            (nxd) Entries of the model
        initial_w: np.ndarray
            Initial weigths
        max_iters: int
            Max number of iterations for the algorithm
        gamma: float
            Learning rate

    Returns:
        w: np.ndarray
            (dx1) Optimal weights
        loss: float
            Empirical loss for the final model
    """
    pass
