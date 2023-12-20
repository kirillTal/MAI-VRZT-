import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import constants
import math

# Шаг 1: Определение функции для уравнений ракеты
def rocket_equations(t, v, M, m0, Ft, Cf, ro, S, g, k):
    # Расчет параметров для изменения скорости сброса топлива
    Ve = Ft / (M - k * t) - (Cf * ro * S) / (2 * (M - k * t)) * v**2
    # Вычисление производной скорости по времени
    dvdt =  (math.log (m0 / (m0 - k * t)) - 0.5 * (ro / (m0 - k)) ) * Ve
    return dvdt
#
# Шаг 2: Определение функции для симуляции полета ракеты
def simulate_rocket_flight(m0, M, Ft, Cf, ro, S, g, k, duration, num_points):
    v0 = 0
    # Создание массива времени для оценки
    t_span = (0, duration)
    t_eval = np.linspace(*t_span, num_points)

    # Параметры уравнений ракеты
    rocket_eq_args = (M, m0, Ft, Cf, ro, S, g, k)

    # Решение дифференциального уравнения для скорости
    solve = integrate.solve_ivp(
        rocket_equations,
        t_span=t_span,
        y0=[v0],
        t_eval=t_eval,
        args=rocket_eq_args
    )

    # Возвращение времени и скорости для построения графика
    return solve.t, solve.y[0]


rocket_data_kerbin = {
    "m0": 46904,
    "M": 46904 + 41482,
    "Ft": 7268861.02,
    "Cf": 0.5,
    "ro": 1.293,
    "S": constants.pi * ((6.6/2)**2),
    "g": 1.00034 * constants.g,
    "k": (46904 + 41482) / (3 * 60 + 5)
}

# Выбор данных для симуляции
rocket_data = rocket_data_kerbin

# Шаг 4: Симуляция полета с выбранными данными
t, v = simulate_rocket_flight(**rocket_data, duration=51, num_points=1080)

# Шаг 5: Построение графика
plt.figure(figsize=(7, 6))
plt.plot(t, v, '-r', label="v(t)")
plt.legend()
plt.grid(True)
plt.xlabel('Time (s)')  # Добавление подписи к оси X
plt.ylabel('Velocity (m/s)')  # Добавление подписи к оси Y
plt.title('Rocket Velocity vs Time')  # Добавление заголовка графика
plt.show()
