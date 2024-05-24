import numpy as np


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates the Pearson correlation coefficient between two arrays x and y.

    Parameters:
    x : numpy.ndarray
        x coordinates array
    y : numpy.ndarray
        y coordinates array

    Returns:
    float
        The Pearson correlation coefficient between x and y.
    """

    if len(x) != len(y):
        raise ValueError("Arrays x and y must have the same length")

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    covariance = np.sum((x - mean_x) * (y - mean_y))

    std_dev_x = np.sqrt(np.sum((x - mean_x) ** 2))
    std_dev_y = np.sqrt(np.sum((y - mean_y) ** 2))

    correlation = covariance / (std_dev_x * std_dev_y)

    return correlation


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Squared Error (MSE) between two arrays y_true and y_pred.

    Parameters:
    y_true : numpy.ndarray
        The array of true values.
    y_pred : numpy.ndarray
        The array of predicted values.

    Returns:
    float
        The Mean Squared Error between y_true and y_pred.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays y_true and y_pred must have the same length")

    squared_errors = (y_true - y_pred) ** 2

    mse = np.mean(squared_errors)

    return mse


def r_squared(actual, predicted):
    """
    Function to calculate the coefficient of determination (R^2).

    :param actual: Actual values.
    :param predicted: Predicted values.
    :return: Coefficient of determination R^2.
    """
    mean_actual = sum(actual) / len(actual)
    ss_total = sum((actual_i - mean_actual) ** 2 for actual_i in actual)
    ss_residual = sum((actual_i - predicted_i) ** 2 for actual_i, predicted_i in zip(actual, predicted))

    r_squared = 1 - (ss_residual / ss_total)

    return r_squared
