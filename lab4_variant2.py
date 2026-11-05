from fractions import Fraction
from math import gcd, isqrt

import numpy as np


# Задание 1. Все действия выполняются с целыми остатками по модулю p.
class Drob:
    def __init__(self, numerator, denominator=1, p=101):
        self.p = p  # В этой работе выбираем простой модуль.
        self.N = isqrt((p - 1) // 2)  # Граница: 2*N*N < p.
        self.value = numerator * pow(denominator, -1, p) % p
        self.farey = self.restore()

    def restore(self):
        # Обратное преобразование: расширенный алгоритм Евклида.
        r0, r1 = self.p, self.value
        t0, t1 = 0, 1
        while r1 > self.N:
            q = r0 // r1
            r0, r1 = r1, r0 - q * r1
            t0, t1 = t1, t0 - q * t1
        if t1 < 0:
            r1, t1 = -r1, -t1
        if not (0 < t1 <= self.N) or gcd(r1, t1) != 1:
            raise ValueError("Остаток не имеет дроби Фарея в выбранных границах")
        return Fraction(r1, t1)

    def check(self, other):
        if self.p != other.p:
            raise ValueError("У дробей должны быть одинаковые модули")

    def __add__(self, other):
        self.check(other)
        return Drob(self.value + other.value, p=self.p)

    def __sub__(self, other):
        self.check(other)
        return Drob(self.value - other.value, p=self.p)

    def __mul__(self, other):
        self.check(other)
        return Drob(self.value * other.value, p=self.p)

    def __truediv__(self, other):
        self.check(other)
        if other.value == 0:
            raise ZeroDivisionError("Деление на нулевую дробь")
        return Drob(self.value * pow(other.value, -1, self.p), p=self.p)

    def __str__(self):
        return str(self.farey)


# Задание 2. Индексы i и j начинаются с нуля, поэтому H[i,j] = 1/(i+j+1).
def hilbert(n, dtype):
    indices = np.arange(n, dtype=dtype)
    return dtype(1) / (indices[:, None] + indices[None, :] + dtype(1))


def gauss_jordan(matrix):
    n = len(matrix)
    a = np.hstack((matrix.copy(), np.eye(n, dtype=matrix.dtype)))
    for k in range(n):
        # Выбираем наибольший по модулю элемент в текущем столбце.
        pivot = k + np.argmax(np.abs(a[k:, k]))
        if a[pivot, k] == 0:
            raise ValueError("Нулевой ведущий элемент: обращение не удалось")
        a[[k, pivot]] = a[[pivot, k]]
        a[k] /= a[k, k]
        for i in range(n):
            if i != k:
                a[i] -= a[i, k] * a[k]
    return a[:, n:]


def main():
    print("Задание 1. Модуль p = 101, порядок Фарея N = 7")
    a, b = Drob(1, 2), Drob(1, 3)
    print(f"a = {a}, остаток = {a.value}; b = {b}, остаток = {b.value}")
    print("a + b =", a + b)
    print("a - b =", a - b)
    print("a * b =", a * b)
    print("a / b =", a / b)
    print("Пример псевдопереполнения: 7 * 7 =", Drob(7) * Drob(7))
    print("Точный ответ 49 вышел за границу N=7, поэтому восстановлена другая дробь.")

    print("\nЗадание 2. C = H * H_obr. Порог потери точности: max|C-I| > 0.01")
    print("err = max|C-I|; rel_diag = max|C[i,i]-1| / 1")
    print(f"{'N':>3} {'err32':>12} {'rel_diag32':>12} {'err64':>12} {'rel_diag64':>12}")
    first_bad = {np.float32: None, np.float64: None}
    for n in range(2, 16):
        errors = []
        for dtype in (np.float32, np.float64):
            h = hilbert(n, dtype)
            try:
                inverse = gauss_jordan(h)
                product = h @ inverse
                error = np.max(np.abs(product - np.eye(n, dtype=dtype)))
                relative = np.max(np.abs(np.diag(product) - dtype(1)))
            except ValueError:
                error = relative = np.inf
            errors.extend((error, relative))
            if (not np.isfinite(error) or error > 0.01) and first_bad[dtype] is None:
                first_bad[dtype] = n
        print(f"{n:3d}" + "".join(f" {error:12.3e}" for error in errors))

    for dtype in (np.float32, np.float64):
        print(f"{dtype.__name__}: первое N с ошибкой > 0.01: {first_bad[dtype]}")


if __name__ == "__main__":
    main()
