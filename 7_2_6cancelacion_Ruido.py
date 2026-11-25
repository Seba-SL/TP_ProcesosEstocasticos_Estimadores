import matplotlib.pyplot as plt
import numpy as np


def simular_cancelacion_ruido(omega0: float,N: int = 200,L: int = 12,phi: float = 0.0,seed: int = 42,):
  rng = np.random.default_rng(seed)

  n = np.arange(N)
  d = np.sin(n * omega0 + phi)
  g = rng.standard_normal(N)

  # Procesos AR(1) acoplados por g(n)
  v1 = np.zeros(N)
  v2 = np.zeros(N)
  for k in range(N):
    v1_prev = v1[k - 1] if k > 0 else 0.0
    v2_prev = v2[k - 1] if k > 0 else 0.0
    v1[k] = 0.8 * v1_prev + g[k]
    v2[k] = -0.6 * v2_prev + g[k]

  x = d + v1

  # Estimadores muestrales para lags k = 0, ..., L-1
  lags = np.arange(L)
  r_v2 = np.zeros(L)
  r_xv2 = np.zeros(L)

  for idx, lag in enumerate(lags):
    v2_shifted = np.zeros(N)
    if lag == 0:
      v2_shifted[:] = v2
    elif lag < N:
      v2_shifted[lag:] = v2[: N - lag]

    r_v2[idx] = np.mean(v2 * v2_shifted)
    r_xv2[idx] = np.mean(x * v2_shifted)

  # Matriz de Toeplitz R_v2 y resolución de las ecuaciones normales
  R_v2 = np.array([[r_v2[abs(i - j)] for j in range(L)] for i in range(L)], dtype=float)
  w = np.linalg.solve(R_v2 + 1e-8 * np.eye(L), r_xv2)

  # Filtrado FIR para estimar v1(n) y luego d(n)
  v1_hat = np.zeros(N)
  for k in range(N):
    acc = 0.0
    for m in range(L):
      if k - m >= 0:
        acc += w[m] * v2[k - m]
    v1_hat[k] = acc

  d_hat = x - v1_hat
  mse = np.mean((d - d_hat) ** 2)
  return n, d, x, v2, d_hat, mse


omegas = [0.05 * np.pi, 0.20 * np.pi, 0.50 * np.pi]


for idx, w0 in enumerate(omegas, start=1):
  n, d, x, v2, d_hat, mse = simular_cancelacion_ruido(omega0=w0, N=200, L=12)
  N=200
  n_fine = np.linspace(0, N-1, 10*N)
  d_fine = np.sin(n_fine * w0 )
 
  plt.figure(num=f"omega0_{idx}", figsize=(9, 4))
  plt.plot(n, x, color="orange",alpha= 0.6, lw=2, label="$x(n) = d(n) + v_1(n)$")
  plt.plot(n_fine, d_fine, "k--", lw=2, label="$d(n)$ real")
  plt.plot(n,d_hat,color="tab:blue",lw=2,label=rf"$\hat{{d}}(n)$ (MSE={mse:.3f})",)
  plt.xlabel("Muestra $n$")
  plt.ylabel("Amplitud")
  plt.title(rf"Cancelación de ruido para $\omega_0 = {w0/np.pi:.2f}\pi$")
  plt.grid(True, alpha=0.3)
  plt.legend(loc="upper right", fontsize=8)
  plt.tight_layout()

plt.show()