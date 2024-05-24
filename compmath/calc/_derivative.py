def derivative_at_point(func, x, h=1e-6):
    return (func(x + h) - func(x - h)) / (2 * h)


def second_derivative_at_point(func, x, h=1e-6):
    return (func(x + h) - 2 * func(x) + func(x - h)) / (h ** 2)


def grad(f, point, h=1e-6):
    gradient_x = (f(point[0] + h, point[1]) - f(point[0] - h, point[1])) / (2 * h)
    gradient_y = (f(point[0], point[1] + h) - f(point[0], point[1] - h)) / (2 * h)
    return gradient_x, gradient_y
