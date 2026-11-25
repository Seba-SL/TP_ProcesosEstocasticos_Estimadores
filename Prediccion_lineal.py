import numpy as np
import matplotlib.pyplot as plt

# Parámetros del proceso AR(1)
N = 200
alpha = 0.9
sigma = 1

# Generación del proceso AR(1)
np.random.seed(10)
v = np.random.normal(0, sigma, N)
x = np.zeros(N)

for n in range(1, N):
    x[n] = alpha*x[n-1] + v[n]

# Predictor lineal de orden 2
w0 = alpha
w1 = 0

x_pred = np.zeros(N)

for n in range(1, N-1):
    x_pred[n+1] = w0*x[n] + w1*x[n-1]

# Gráfico
plt.figure(figsize=(11, 5))
plt.plot(x,linewidth = 3, label='Proceso AR(1)')
plt.plot(x_pred,linewidth=3,linestyle='--',label=r'Predicción lineal (orden 2): $\hat{x}[n+1] = w_0 x[n] + w_1 x[n-1]$')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.title(f'Predicción de un proceso AR(1) ($\\alpha={alpha}$)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()