class Matrix:
    def __init__(self, rows):
        self.rows = rows
        self.num_rows = len(rows)
        self.num_cols = len(rows[0])

        self.shape = (self.num_rows, self.num_cols)

    def __add__(self, other):
        if self.num_rows != other.num_rows or self.num_cols != other.num_cols:
            raise ValueError("Matrices must have the same dimensions for addition")

        result = [[self[i][j] + other[i][j] for j in range(self.num_cols)] for i in range(self.num_rows)]
        return Matrix(result)

    def __sub__(self, other):
        if self.num_rows != other.num_rows or self.num_cols != other.num_cols:
            raise ValueError("Matrices must have the same dimensions for subtraction")

        result = [[self[i][j] - other[i][j] for j in range(self.num_cols)] for i in range(self.num_rows)]
        return Matrix(result)

    def __mul__(self, other):
        if self.num_cols != other.num_rows:
            raise ValueError(
                "Number of columns in first matrix must be equal to the number of rows in the second matrix")

        result = [[0] * other.num_cols for _ in range(self.num_rows)]

        for i in range(self.num_rows):
            for j in range(other.num_cols):
                for k in range(self.num_cols):
                    result[i][j] += self.rows[i][k] * other[k][j]

        return Matrix(result)

    def __pow__(self, p):
        if self.num_rows != self.num_cols:
            raise ValueError("Matrix must be square for exponentiation")

        if p == 0:
            return Matrix([[1 if i == j else 0 for j in range(self.num_rows)] for i in range(self.num_rows)])

        result = self
        for _ in range(p - 1):
            result = result * self

        return result

    def upper_triangular(self):
        matrix = self.copy()
        row_swaps = 0

        for i in range(self.num_rows):
            # Find the maximum element in the current column below the diagonal
            max_row = i
            for j in range(i + 1, self.num_rows):
                if abs(matrix[j][i]) > abs(matrix[max_row][i]):
                    max_row = j

            # Swap the rows to ensure the maximum element is on the diagonal
            if max_row != i:
                matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
                row_swaps += 1

            # Make the elements below the diagonal zero
            for j in range(i + 1, self.num_rows):
                try:
                    factor = -matrix[j][i] / matrix[i][i]
                except ZeroDivisionError:
                    factor = 0
                for k in range(i, self.num_rows):
                    matrix[j][k] += factor * matrix[i][k]

        return matrix, row_swaps

    def det(self):
        if self.num_rows != self.num_cols:
            raise ValueError("Matrix must be square to compute determinant")

        # Make the matrix upper triangular
        upper_triangular_matrix, row_swaps = self.upper_triangular()

        # Compute the determinant as the product of diagonal elements
        d_sign = (-1) ** row_swaps
        d = 1
        for i in range(self.num_rows):
            d *= upper_triangular_matrix[i][i]

        return d * d_sign

    def det_slow(self):
        if self.num_rows != self.num_cols:
            raise ValueError("Matrix must be square to compute determinant")

        if self.num_rows == 1:
            return self[0][0]

        d = 0
        for j in range(self.num_cols):
            sign = (-1) ** j
            sub_matrix = [row[:j] + row[j + 1:] for row in self.rows[1:]]
            d += sign * self[0][j] * Matrix(sub_matrix).det_slow()

        return d

    def __getitem__(self, index):
        return self.rows[index]

    def __setitem__(self, index, value):
        self.rows[index] = value

    def __eq__(self, other):
        return self.rows == other.rows

    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.rows])

    def __len__(self):
        return self.num_rows

    def __iter__(self):
        return iter(self.rows)

    def copy(self):
        return Matrix([row[:] for row in self.rows])
