from compmath.calc import derivative_at_point, second_derivative_at_point
import numpy as np


def simple_iteration(phi, f, a, b, eps=1e-6, max_iter=100):
    """
    Solve equation f(x) = 0 using simple iteration method.

    Args:
    phi: The function representing the iteration function.
    f: The function for which the root is to be found.
    a, b: The interval in which to search for the root.
    eps: Tolerance for stopping criterion (default: 1e-6).
    max_iter: Maximum number of iterations (default: 100).

    Returns:
    List of tuples containing iteration log information.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("Method is not applicable on this interval")

    x_prev = a
    log = [('x_k', 'x_k+1', 'phi(x_k+1)', 'f(x_k+1)', '|x_k+1 - x_k|')]

    for _ in range(max_iter):
        x_new = phi(x_prev)
        log.append((x_prev, x_new, phi(x_new), f(x_new), abs(x_new - x_prev)))

        if abs(x_new - x_prev) < eps:
            return log

        x_prev = x_new

    raise RuntimeError("Method did not converge within the maximum number of iterations")


def chord_method(f, a, b, eps=1e-6, max_iter=100):
    """
    Solve equation f(x) = 0 using chord method.

    Args:
    f: The function for which the root is to be found.
    a, b: The interval in which to search for the root.
    eps: Tolerance for stopping criterion (default: 1e-6).
    max_iter: Maximum number of iterations (default: 100).

    Returns:
    List of tuples containing iteration log information.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("Method is not applicable on this interval")

    x0 = a
    x1 = b
    log = [('x0', 'x1', 'x2', 'f(x0)', 'f(x1)', 'f(x2)', 'x1 - x0')]

    for _ in range(max_iter):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0 = x1
        x1 = x2
        log.append((x0, x1, x2, f(x0), f(x1), f(x2), x1 - x0))

        if abs(f(x1)) < eps:
            return log

    raise RuntimeError("Method did not converge within the maximum number of iterations")


def bin_search(f, a, b, eps=1e-6, max_iter=100):
    """
    Solve equation f(x) = 0 using binary search method.

    Args:
    f: The function for which the root is to be found.
    a, b: The interval in which to search for the root.
    eps: Tolerance for stopping criterion (default: 1e-6).
    max_iter: Maximum number of iterations (default: 100).

    Returns:
    List of tuples containing iteration log information.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("Method is not applicable on this interval")

    log = [('a', 'b', 'm', 'f(a)', 'f(b)', 'f(m)', 'b - a')]
    start = f(a)

    for _ in range(max_iter):
        m = (a + b) / 2
        log.append((a, b, m, f(a), f(b), f(m), b - a))

        if f(m) * start < 0:
            b = m
        else:
            a = m

        if abs(f(m)) < eps:
            return log

    raise RuntimeError("Method did not converge within the maximum number of iterations")


def _is_sign_constant(f, a, b):
    grid = np.linspace(a, b, 100)
    sign = f(a) > 0

    for x in grid:
        if (f(x) > 0) != sign:
            return False

    return True


def newton_method(f, a, b, eps=1e-6, max_iter=100):
    """
    Find a root of a function within the given interval using Newton's method.

    Args:
    f: The function for which the root is to be found.
    a, b: The interval in which to search for the root.
    eps: Tolerance for stopping criterion (default: 1e-6).
    max_iter: Maximum number of iterations (default: 100).

    Returns:
    The approximate root of the function within the given interval.
    """
    if f(a) * f(b) > 0:
        raise ValueError("No root found in the given interval")

    if not _is_sign_constant(lambda x: derivative_at_point(f, x), a, b):
        raise ValueError("First derivative of f is not sign constant")

    if not _is_sign_constant(lambda x: second_derivative_at_point(f, x), a, b):
        raise ValueError("Second derivative of f is not sign constant")

    x = (a + b) / 2

    log = [('x_k', 'f(x_k)', "f'(x_k)", 'x_k+1', '|x_new - x|')]

    for _ in range(max_iter):
        df_dx = derivative_at_point(f, x)

        if abs(df_dx) < 1e-10:
            raise ValueError("Derivative is close to zero. Newton's method may not converge")

        x_new = x - f(x) / df_dx

        log.append((x, f(x), df_dx, x_new, abs(x_new - x)))

        if abs(f(x)) < eps:
            return log

        x = x_new

    raise RuntimeError("Method did not converge within the maximum number of iterations")

