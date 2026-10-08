import numpy as np


# Задание 1, вариант 2
inf = np.float64(np.inf)
zero = np.float64(0.0)
minus_zero = np.float64(-0.0)

print("Task 1")
with np.errstate(divide="ignore", invalid="ignore"):
    print("inf / 0 =", inf / zero)
    print("0 / inf =", zero / inf)
    print("inf / inf =", inf / inf)
    print("-inf / (-0) =", -inf / minus_zero)
    print("0 / (-0) =", zero / minus_zero)

print("0 == -0:", zero == minus_zero)
print("-inf == inf:", -inf == inf)


# Задание 2, вариант 2: общий случай и пример из условия.
print("\nTask 2")
a = np.float32(np.inf)
b = np.float32(0.0)
c = np.float32(-0.0)
with np.errstate(invalid="ignore"):
    left = a * b
    right = a * c
print("General case: b == c:", b == c)
print(f"a*b = {left}, a*c = {right}, a*b == a*c: {left == right}")

a = np.float32(2)
b = np.float32(1)

for k in (23, 24):
    c = np.float32(1 + 2 ** (-k))
    print(f"k = {k}: b = {b}, c = {c}, b == c: {b == c}")
    print(f"       a*b = {a * b}, a*c = {a * c}")

print("For a=2, b=1, c=1+2^(-k), no suitable k exists.")
