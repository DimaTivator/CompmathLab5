from compmath.calc import *


def check_gradient_condition(phi1, phi2, x, y, step=0.01):
    for i in range(-50, 50):
        for j in range(-50, 50):
            x_test = x + i * step
            y_test = y + j * step
            try:
                gradient_phi1 = grad(phi1, (x_test, y_test))
                gradient_phi2 = grad(phi2, (x_test, y_test))
            except ValueError:
                continue
            if sum(gradient_phi1) >= 1 or sum(gradient_phi2) >= 1:
                return False
    return True


def simple_iteration_2d(f1, f2, phi1, phi2, initial_guess, max_iter=100, eps=1e-6):
    x, y = initial_guess
    if not check_gradient_condition(phi1, phi2, x, y):
        raise ValueError("Gradient condition not satisfied")

    log = [('x_k', 'y_k', 'x_k+1', 'y_k+1', 'F1(x_k+1, y_k+1)', 'F2(x_k+1, y_k+1)', '|x_k+1 - x_k|', '|y_k+1 - y_k|')]

    for _ in range(max_iter):
        x_next = phi1(x, y)
        y_next = phi2(x, y)

        print(x, y, x_next, y_next, f1(x_next, y_next), f2(x_next, y_next), abs(x_next - x), abs(y_next - y))

        log.append((x, y, x_next, y_next, f1(x_next, y_next), f2(x_next, y_next), abs(x_next - x), abs(y_next - y)))

        if abs(x_next - x) < eps and abs(y_next - y) < eps:
            return log

        x, y = x_next, y_next

    raise RuntimeError("Failed to converge in 100 iterations")
