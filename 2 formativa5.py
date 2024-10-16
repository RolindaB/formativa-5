import numpy as np
import matplotlib.pyplot as plt

# Definimos la ecuación diferencial y'(x) = 4e^(-0.8x) - 0.5y
def f(x, y):
    return 4 * np.exp(-0.8 * x) - 0.5 * y

# Método de Runge-Kutta de cuarto orden
def runge_kutta_4th_order(x0, y0, x_final, h):
    n = int((x_final - x0) / h)  # Número de pasos
    x = np.linspace(x0, x_final, n+1)  # Valores de x
    y = np.zeros(n+1)  # Valores de y
    y[0] = y0  # Condición inicial

    for i in range(n):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6  # Fórmula RK4

    return x, y

# Parámetros iniciales
x0 = 0  # Valor inicial de x
y0 = 2  # Condición inicial y(0) = 2
x_final = 10  # Valor final de x (puedes ajustar este valor)
h = 0.1  # Tamaño del paso (puedes ajustarlo)

# Llamamos al método de Runge-Kutta
x_values, y_values = runge_kutta_4th_order(x0, y0, x_final, h)

# Mostramos el valor aproximado en x_final
print(f"El valor de y({x_final}) es aproximadamente: {y_values[-1]}")

# Graficamos los resultados
plt.plot(x_values, y_values, label="Solución RK4")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Método de Runge-Kutta de 4to orden')
plt.legend()
plt.grid(True)
plt.show()

