import numpy as np

# Definimos la función que representa la ecuación diferencial
def f(x, y):
    return 4 * np.exp(0.8 * x) - 0.5 * y

# Implementación del método de Euler mejorado (predictor-corrector)
def euler_mejorado(f, y0, x0, h, n):
    x = x0
    y = y0
    for i in range(n):
        # Predictor (Euler explícito)
        y_predict = y + h * f(x, y)
        
        # Corrector (Euler implícito)
        y_correct = y + h * f(x + h, y_predict)
        
        # Actualización del valor de y
        y = (y_predict + y_correct) / 2
        x += h
    return y

# Parámetros iniciales
y0 = 2  # Condición inicial y(0) = 2
x0 = 0  # Valor inicial de x
h = 0.1  # Paso de integración (puedes ajustar esto)
x_final = 4  # Queremos aproximar y(4)
n = int((x_final - x0) / h)  # Número de pasos

# Aproximación de y(4) usando el método de Euler mejorado
resultado = euler_mejorado(f, y0, x0, h, n)
print(f"La aproximación de y(4) es: {resultado:.6f}")