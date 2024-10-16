import numpy as np
import matplotlib.pyplot as plt

# Definimos la función que representa la ecuación diferencial dy/dx = f(x, y)
def f(x, y):
    return x * np.sin(y)  # Puedes cambiar esta función según tu problema

# Método de Euler mejorado (Heun's method o Predictor-Corrector)
def euler_mejorado(x0, y0, x_final, h):
    n = int((x_final - x0) / h)  # Número de pasos
    x = np.linspace(x0, x_final, n+1)  # Valores de x
    y = np.zeros(n+1)  # Valores de y
    y[0] = y0  # Condición inicial

    for i in range(n):
        # Predicción con método de Euler
        y_predictor = y[i] + h * f(x[i], y[i])

        # Cálculo de la pendiente promedio (corrector)
        y[i+1] = y[i] + (h / 2) * (f(x[i], y[i]) + f(x[i+1], y_predictor))

    return x, y

# Parámetros iniciales
x0 = 0  # Valor inicial de x
y0 = 1  # Valor inicial de y (condición inicial)
x_final = 10  # Valor final de x
h = 0.1  # Tamaño del paso

# Llamamos al método de Euler mejorado
x_values, y_values = euler_mejorado(x0, y0, x_final, h)

# Graficamos los resultados
plt.plot(x_values, y_values, label="Solución Euler Mejorado")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Método de Euler Mejorado (Heun\'s Method)')
plt.legend()
plt.grid(True)
plt.show()
