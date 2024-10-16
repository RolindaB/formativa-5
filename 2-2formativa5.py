import numpy as np
import matplotlib.pyplot as plt

# Definimos la ecuación diferencial y'(x) = 4 / (1 + x^2)
def f(x, y):
    return 4 / (1 + x**2)

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
y0 = 0  # Condición inicial y(0) = 0
x_final = 1  # Valor final de x (queremos el valor en x = 1)
h = 0.01  # Tamaño del paso 

# Llamamos al método de Runge-Kutta
x_values, y_values = runge_kutta_4th_order(x0, y0, x_final, h)

# Encuentra el valor de y cuando x = 1 (aproximación de pi)
pi_approx = y_values[-1]
print(f"La aproximación de pi es: {pi_approx}")

# Graficamos los resultados
plt.plot(x_values, y_values, label="Solución RK4")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Método de Runge-Kutta de 4to orden para aproximar pi')
plt.legend()
plt.grid(True)
plt.show()
