import itertools
from compmath.linalg import Matrix


def _greedy_get_diagonally_dominant(A: Matrix):
    a = A.copy()
    num_rows, num_cols = a.shape
    for i in range(a.num_rows):
        # the sum of the row without the diagonal one
        row_sum = sum(abs(a[i][j]) for j in range(num_cols)) - abs(a[i][i])
        if abs(a[i][i]) <= row_sum:
            # use custom key to get argmax
            max_col_index = max(range(a.num_cols), key=lambda j: abs(a[i][j]))
            # swap the rows
            for k in range(num_rows):
                a[k][i], a[k][max_col_index] = a[k][max_col_index], a[k][i]

    return a


def get_diagonally_dominant(A: Matrix):
    # try greedy approach
    greedy_a = _greedy_get_diagonally_dominant(A)
    if is_diagonally_dominant(greedy_a):
        return greedy_a

    # bruteforce doesn't make sense here
    if A.num_cols > 10:
        return None

    # bruteforce approach
    a = A.copy()
    for permutation in itertools.permutations(range(a.num_cols)):
        # rearrange the columns
        new_a = Matrix([[row[i] for i in permutation] for row in a])
        if is_diagonally_dominant(new_a):
            return new_a

    # matrix is not diagonally dominant
    return None


def is_diagonally_dominant(A: Matrix):
    if A.num_rows != A.num_cols:
        raise ValueError("Matrix must be square to be diagonally dominant")

    f = True
    for i in range(A.num_rows):
        f &= A[i][i] >= sum(A[i][:]) - A[i][i]
    return f


def gaussian_elimination(A, B):
    n = len(B)

    # Forward elimination
    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            B[j] -= factor * B[i]

    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = B[i] / A[i][i]
        for j in range(i - 1, -1, -1):
            B[j] -= A[j][i] * x[i]

    return x


