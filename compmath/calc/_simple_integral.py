from compmath.calc._improper_integral import check_convergence


def left_rectangles(f, a, b, eps=1e-2):
    if not check_convergence(f, a, b):
        raise Exception('Integral diverges')

    a += 1e-6
    b -= 1e-6

    n = 4
    integral_prev = None

    log = [('n partition', 'segment length', 'integral', 'inaccuracy')]

    while True:
        h = (b - a) / n
        integral = 0

        for i in range(n):
            integral += f(a + i * h)

        integral *= h

        log.append((n, h, integral, '-' if integral_prev is None else abs(integral - integral_prev)))

        if integral_prev is not None:
            if abs(integral - integral_prev) < eps:
                break

        integral_prev = integral
        n *= 2

        if n >= 4194304:
            raise Exception('Integral diverges or required precision is too low')

    return log


def right_rectangles(f, a, b, eps=1e-2):
    if not check_convergence(f, a, b):
        raise Exception('Integral diverges')

    a += 1e-6
    b -= 1e-6

    n = 4
    integral_prev = None

    log = [('n partition', 'segment length', 'integral', 'inaccuracy')]

    while True:
        h = (b - a) / n
        integral = 0

        for i in range(n):
            integral += f(a + (i + 1) * h)

        integral *= h

        log.append((n, h, integral, '-' if integral_prev is None else abs(integral - integral_prev)))

        if integral_prev is not None:
            if abs(integral - integral_prev) < eps:
                break

        integral_prev = integral
        n *= 2

        if n >= 4194304:
            raise Exception('Integral diverges or required precision is too low')

    return log


def midpoint_rectangles(f, a, b, eps=1e-2):
    if not check_convergence(f, a, b):
        raise Exception('Integral diverges')

    a += 1e-6
    b -= 1e-6

    n = 4
    integral_prev = None

    log = [('n partition', 'segment length', 'integral', 'inaccuracy')]

    while True:
        h = (b - a) / n
        integral = 0

        for i in range(n):
            midpoint = a + (i + 0.5) * h
            integral += f(midpoint)

        integral *= h

        log.append((n, h, integral, '-' if integral_prev is None else abs(integral - integral_prev)))

        if integral_prev is not None:
            if abs(integral - integral_prev) < eps:
                break

        integral_prev = integral
        n *= 2

        if n >= 4194304:
            raise Exception('Integral diverges or required precision is too low')

    return log


def trapezoidal(f, a, b, eps=1e-2):
    if not check_convergence(f, a, b):
        raise Exception('Integral diverges')

    a += 1e-6
    b -= 1e-6

    n = 4
    integral_prev = None

    log = [('n partition', 'segment length', 'integral', 'inaccuracy')]

    while True:
        h = (b - a) / n
        integral = 0

        for i in range(n):
            x0 = a + i * h
            x1 = a + (i + 1) * h
            integral += (f(x0) + f(x1)) / 2

        integral *= h

        log.append((n, h, integral, '-' if integral_prev is None else abs(integral - integral_prev)))

        if integral_prev is not None:
            if abs(integral - integral_prev) < eps:
                break

        integral_prev = integral
        n *= 2

        if n >= 4194304:
            raise Exception('Integral diverges or required precision is too low')

    return log


def simpson(f, a, b, eps=1e-2):
    if not check_convergence(f, a, b):
        raise Exception('Integral diverges')

    a += 1e-6
    b -= 1e-6

    n = 2
    integral_prev = None

    log = [('n partition', 'segment length', 'integral', 'inaccuracy')]

    while True:
        h = (b - a) / n
        integral = 0

        for i in range(n):
            x0 = a + i * h
            x1 = a + (i + 1) * h
            x_mid = (x0 + x1) / 2
            integral += f(x0) + 4 * f(x_mid) + f(x1)

        integral *= h / 6

        log.append((n, h, integral, '-' if integral_prev is None else abs(integral - integral_prev)))

        if integral_prev is not None:
            if abs(integral - integral_prev) < eps:
                break

        integral_prev = integral
        n *= 2

        if n >= 4194304:
            raise Exception('Integral diverges or required precision is too low')

    return log

