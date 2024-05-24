import numpy as np
from compmath.linalg import gaussian_elimination


def linear_least_squares(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Compute the coefficients (a, b) of the linear equation ax + b that best fits the given points (x, y)
    using the least squares method.

    Args:
    x (np.ndarray): List of x-coordinates of the points.
    y (np.ndarray): List of y-coordinates of the points.

    Returns:
    tuple: Coefficients (a, b) of the linear equation and approximate y values.
    """

    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_squared = np.sum(x ** 2)
    sum_xy = np.sum(x * y)

    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    b = (sum_y - a * sum_x) / n

    y_approx = a * x + b

    return a, b, y_approx


def quadratic_least_squares(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Compute the coefficients (a, b, c) of the quadratic equation ax^2 + bx + c that best fits
    the given points (x, y) using the least squares method.

    Args:
    x (list or np.ndarray): List or array of x-coordinates of the points.
    y (list or np.ndarray): List or array of y-coordinates of the points.

    Returns:
    tuple: Coefficients (a, b, c) of the quadratic equation and approximate y values.
    """

    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_squared = sum(x_i ** 2 for x_i in x)
    sum_x_cubed = sum(x_i ** 3 for x_i in x)
    sum_x_fourth = sum(x_i ** 4 for x_i in x)
    sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
    sum_x_squared_y = sum(x_i ** 2 * y_i for x_i, y_i in zip(x, y))

    A = [[sum_x_fourth, sum_x_cubed, sum_x_squared],
         [sum_x_cubed, sum_x_squared, sum_x],
         [sum_x_squared, sum_x, n]]
    B = [sum_x_squared_y, sum_xy, sum_y]

    coefficients = gaussian_elimination(A, B)

    y_approx = coefficients[0] * x ** 2 + coefficients[1] * x + coefficients[2]

    return tuple(coefficients + [y_approx])


def cubic_least_squares(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Compute the coefficients (a, b, c, d) of the cubic equation ax^3 + bx^2 + cx + d that best fits
    the given points (x, y) using the least squares method.

    Args:
    x (list or np.ndarray): List or array of x-coordinates of the points.
    y (list or np.ndarray): List or array of y-coordinates of the points.

    Returns:
    tuple: Coefficients (a, b, c, d) of the cubic equation and approximate y values.
    """

    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_squared = sum(x_i ** 2 for x_i in x)
    sum_x_cubed = sum(x_i ** 3 for x_i in x)
    sum_x_fourth = sum(x_i ** 4 for x_i in x)
    sum_x_fifth = sum(x_i ** 5 for x_i in x)
    sum_x_sixth = sum(x_i ** 6 for x_i in x)
    sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
    sum_x_squared_y = sum(x_i ** 2 * y_i for x_i, y_i in zip(x, y))
    sum_x_cubed_y = sum(x_i ** 3 * y_i for x_i, y_i in zip(x, y))

    A = [
        [sum_x_sixth, sum_x_fifth, sum_x_fourth, sum_x_cubed],
        [sum_x_fifth, sum_x_fourth, sum_x_cubed, sum_x_squared],
        [sum_x_fourth, sum_x_cubed, sum_x_squared, sum_x],
        [sum_x_cubed, sum_x_squared, sum_x, n]
    ]
    B = [sum_x_cubed_y, sum_x_squared_y, sum_xy, sum_y]

    coefficients = gaussian_elimination(A, B)

    y_approx = (coefficients[0] * x ** 3 +
                coefficients[1] * x ** 2 +
                coefficients[2] * x +
                coefficients[3])

    return tuple(coefficients + [y_approx])


def logarithmic_least_squares(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Compute the coefficients (a, b) of the logarithmic equation y = a * ln(x) + b that best fits the given points (x, y)
    using the least squares method.

    Args:
    x (np.ndarray): List of x-coordinates of the points.
    y (np.ndarray): List of y-coordinates of the points.

    Returns:
    tuple: Coefficients (a, b) of the logarithmic equation and approximate y values.
    """
    log_x = np.log(x)

    a, b, _ = linear_least_squares(log_x, y)

    y_approx = a * np.log(x) + b

    return a, b, y_approx


def exponential_least_squares(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Compute the coefficients (a, b) of the exponential equation y = a * e^(bx) that best fits the given points (x, y)
    using the least squares method.

    Args:
    x (np.ndarray): List of x-coordinates of the points.
    y (np.ndarray): List of y-coordinates of the points.

    Returns:
    tuple: Coefficients (a, b) of the exponential equation and approximate y values.
    """
    log_y = np.log(y)

    a, b, _ = linear_least_squares(x, log_y)

    b = np.exp(b)

    y_approx = a * np.exp(b * x)

    return a, b, y_approx


def power_least_squares(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Approximate the given points (x, y) with a power function using linear approximation.

    Args:
    x (np.ndarray): List of x-coordinates of the points.
    y (np.ndarray): List of y-coordinates of the points.

    Returns:
    tuple: Coefficients (a, b) of the power equation and approximate y values.
    """
    x_log = np.log(x)
    y_log = np.log(y)

    a, b, _ = linear_least_squares(x_log, y_log)

    a_power = np.exp(b)
    b_power = a

    y_approx = a_power * x ** b_power

    return a_power, b_power, y_approx
