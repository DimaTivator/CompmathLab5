from compmath.linalg import Matrix


def abs_deviation(prev: list, cur: list):
    return max([abs(prev[i] - cur[i]) for i in range(len(prev))])


def relative_diff(prev: list, cur: list):
    return max([abs((prev[i] - cur[i]) / cur[i]) if cur[i] != 0 else float('inf') for i in range(len(prev))])


# def discrepancy_diff(A: Matrix, b: Matrix, x: Matrix):
#     r = A * x - b
#     return max([r[i] for i in range(len(r))])

