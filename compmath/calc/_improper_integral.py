def _rectangle_method_integration(f, a, b, n=100):
    a += 1e-8
    b -= 1e-8

    dx = (b - a) / n
    integral = 0

    for i in range(n):
        x = a + i * dx
        integral += f(x) * dx

    return integral


def check_convergence(f, a, b, tol=1e2):
    integral_approx = _rectangle_method_integration(f, a, b)
    print(integral_approx)
    print(_rectangle_method_integration(f, a, b, n=200))
    return abs(integral_approx - _rectangle_method_integration(f, a, b, n=200)) < tol



