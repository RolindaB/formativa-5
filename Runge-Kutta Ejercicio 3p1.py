import numpy as np

# Definimos la función que representa la ecuación diferencial
def f(x, y):
    return 2 * y - 6

# Implementación del método de Runge-Kutta de orden 4
def runge_kutta_4(f, y0, x0, h, n):
    x = x0
    y = y0
    for i in range(n):
        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2)
        k4 = h * f(x + h, y + k3)
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x += h
    return y

# Parámetros iniciales
y0 = 1  # Condición inicial y(0) = 1
x0 = 0  # Valor inicial de x
h = 0.1  # Paso de integración (puedes ajustar esto)
x_final = 1  # Queremos aproximar y(1)
n = int((x_final - x0) / h)  # Número de pasos

# Aproximación de y(1) usando el método de Runge-Kutta de orden 4
resultado = runge_kutta_4(f, y0, x0, h, n)
print(f"La aproximación de y(1) es: {resultado:.6f}")