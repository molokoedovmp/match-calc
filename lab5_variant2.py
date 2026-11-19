from fractions import Fraction

import numpy as np
from mpmath import mp


def convergents(x):
    """Подходящие дроби через целочисленное деление и рекуррентные формулы."""
    a, b = x.numerator, x.denominator
    p0, p1, q0, q1 = 0, 1, 1, 0
    while b:
        c, remainder = divmod(a, b)
        p0, p1 = p1, c * p1 + p0
        q0, q1 = q1, c * q1 + q0
        yield c, Fraction(p1, q1)
        a, b = b, remainder


def dot_float(a, b, dtype):
    total = dtype(0)
    for x, y in zip(a, b):
        x = dtype(x.numerator) / dtype(x.denominator)
        y = dtype(y.numerator) / dtype(y.denominator)
        total = dtype(total + dtype(x * y))
    return Fraction(float(total))


def dot_mp(a, b, bits):
    with mp.workprec(bits):
        total = mp.mpf(0)
        for x, y in zip(a, b):
            x = mp.mpf(x.numerator) / x.denominator
            y = mp.mpf(y.numerator) / y.denominator
            total += x * y
        # Внутри mpf хранится (-1)^sign * mantissa * 2^exponent.
        sign, mantissa, exponent, _ = total._mpf_
        return Fraction((-1) ** sign * mantissa) * Fraction(2) ** exponent


def reachable(graph, start):
    seen, todo = set(), [start]
    while todo:
        state = todo.pop()
        if state not in seen:
            seen.add(state)
            todo.extend(graph[state])
    return seen


def eventually(graph, target, all_paths):
    """EF: есть следующий путь; AF: все следующие пути ведут к цели."""
    result = set(target)
    while True:
        expanded = result | {
            s for s, next_states in graph.items()
            if (all if all_paths else any)(t in result for t in next_states)
        }
        if expanded == result:
            return result
        result = expanded


def main():
    print("ЗАДАНИЕ 1. Дробь 13/34; модуль 10007")
    x, modulus = Fraction(13, 34), 10007
    items = list(convergents(x))
    print("Цепная дробь:", [c for c, _ in items])
    print("Дробь     Порядок    Остаток    Абсолютная ошибка")
    for _, f in items:
        order = max(abs(f.numerator), f.denominator)
        if order < max(abs(x.numerator), x.denominator):
            residue = f.numerator * pow(f.denominator, -1, modulus) % modulus
            print(f"{str(f):>6} {order:10d} {residue:10d} {str(abs(x-f)):>20}")

    print("\nЗАДАНИЕ 2. Учебные векторы: (10^8, 1/3, -10^8) и (1, 1, 1)")
    a = [Fraction(10**8), Fraction(1, 3), Fraction(-10**8)]
    b = [Fraction(1), Fraction(1), Fraction(1)]
    exact = sum((x * y for x, y in zip(a, b)), Fraction(0))
    print("Точное скалярное произведение:", exact)
    for dtype in (np.float32, np.float64):
        baseline = dot_float(a, b, dtype)
        base_error = abs(baseline - exact)
        print(f"{dtype.__name__}: {float(baseline):.17g}; ошибка {float(base_error):.6e}")
        for bits in range(2, 101):
            value = dot_mp(a, b, bits)
            error = abs(value - exact)
            if error < base_error:
                print(f"Первое улучшение mpmath: {bits} бит; результат {float(value):.17g}; ошибка {float(error):.6e}")
                break
        else:
            print("Улучшение в диапазоне 2–100 бит не найдено")

    print("\nВАРИАНТ 2. Проверка модели сервера")
    graph = {"idle": ["req"], "req": ["wait"], "wait": ["ack", "wait"], "ack": ["idle"]}
    for label in ("До исправления", "После исправления"):
        states = reachable(graph, "idle")
        ef = eventually(graph, {"ack"}, False)
        af = eventually(graph, {"ack"}, True)
        print(label, graph)
        print("EF(ack) =", sorted(ef), "; AF(ack) =", sorted(af))
        print("AG(req -> EF ack):", all(s in ef for s in states if s == "req"))
        print("AG(req -> AF ack):", all(s in af for s in states if s == "req"))
        if label == "До исправления":
            print("Контрпример гарантии: idle -> req -> wait -> wait -> ...")
        graph["wait"] = ["ack"]


if __name__ == "__main__":
    main()
