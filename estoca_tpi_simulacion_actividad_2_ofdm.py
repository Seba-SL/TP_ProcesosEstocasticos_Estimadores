# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del sistema (Edfors et al., Sec. II.C)
N = 64
L = 4
tau_rms = 1.0
K = 1000
snr_db_vec = np.arange(0, 31, 5)
rng = np.random.default_rng(42)

# Constelación 16-QAM y constante beta
niveles = np.array([-3, -1, 1, 3])
constelacion = np.array([a + 1j * b for a in niveles for b in niveles])
Es = np.mean(np.abs(constelacion) ** 2)
beta = Es * np.mean(1.0 / np.abs(constelacion) ** 2)  # 17/9

# Matriz de autocorrelación R_hh analítica (Apéndice A)
idx = np.arange(N)
diff = idx[:, None] - idx[None, :]
a = (1.0 / tau_rms) + 1j * 2 * np.pi * diff / N
R_hh = (1.0 - np.exp(-L * a)) / (tau_rms * (1.0 - np.exp(-L / tau_rms)) * a)

# Descomposición espectral para coloreado del canal: h = U * Lambda^(1/2) * z
lam, U = np.linalg.eigh(R_hh)
lam = np.maximum(lam, 0.0)
A_color = U @ np.diag(np.sqrt(lam))
I = np.eye(N)

# Arreglos para almacenar el MSE promedio
mse_ls = np.zeros(len(snr_db_vec))
mse_lmmse = np.zeros(len(snr_db_vec))
mse_simp = np.zeros(len(snr_db_vec))
mse_teo_simp = np.zeros(len(snr_db_vec))

ejemplo = {}  # Guarda una realización puntual para graficar el canal

print(f"{'SNR [dB]':>8} | {'LS [dB]':>9} | {'LMMSE [dB]':>10} | "
      f"{'Simp [dB]':>10} | {'Teo Ec.5':>9} | {'Dif [dB]':>9}")
print("-" * 70)

for idx_s, snr_db in enumerate(snr_db_vec):
    snr = 10 ** (snr_db / 10)
    sigma_n2 = Es / snr

    # Matriz fija del estimador simplificado (Ec. 5): se calcula una sola vez
    W_simp = R_hh @ np.linalg.inv(R_hh + (beta / snr) * I)

    # MSE teórico cerrado para la Ec. (5) (Apéndice D, rango completo p = N)
    mse_teo_simp[idx_s] = np.mean((lam * (beta / snr)) / (lam + beta / snr))

    acc_ls = acc_lmmse = acc_simp = 0.0

    for k in range(K):
        # Canal h ~ CN(0, R_hh), símbolos x y ruido n ~ CN(0, sigma_n2 * I)
        z = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
        h = A_color @ z
        x = rng.choice(constelacion, size=N)
        n = np.sqrt(sigma_n2 / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))

        # Modelo OFDM (Ec. 1) y estimador LS (Ec. 4)
        y = x * h + n
        h_ls = y / x

        # Estimador LMMSE exacto (Ec. 3): invierte una matriz en cada realización
        W_lmmse = R_hh @ np.linalg.inv(R_hh + sigma_n2 * np.diag(1.0 / np.abs(x) ** 2))
        h_lmmse = W_lmmse @ h_ls

        # Estimador LMMSE simplificado (Ec. 5)
        h_simp = W_simp @ h_ls

        acc_ls += np.mean(np.abs(h - h_ls) ** 2)
        acc_lmmse += np.mean(np.abs(h - h_lmmse) ** 2)
        acc_simp += np.mean(np.abs(h - h_simp) ** 2)

        if snr_db == 10 and k == 0:
            ejemplo = {"h": h, "h_lmmse": h_lmmse, "h_simp": h_simp}

    mse_ls[idx_s] = acc_ls / K
    mse_lmmse[idx_s] = acc_lmmse / K
    mse_simp[idx_s] = acc_simp / K

    dif_db = 10 * np.log10(mse_simp[idx_s] / mse_lmmse[idx_s])
    print(f"{snr_db:8d} | {10*np.log10(mse_ls[idx_s]):9.3f} | "
          f"{10*np.log10(mse_lmmse[idx_s]):10.3f} | "
          f"{10*np.log10(mse_simp[idx_s]):10.3f} | "
          f"{10*np.log10(mse_teo_simp[idx_s]):9.3f} | {dif_db:9.3f}")

# Figura 1: Comparación de MSE vs. SNR
plt.figure(figsize=(8, 5))
plt.plot(snr_db_vec, 10 * np.log10(mse_ls), "o-", label="LS — Ec. (4)")
plt.plot(snr_db_vec, 10 * np.log10(mse_lmmse), "^-", label="LMMSE exacto — Ec. (3)")
plt.plot(snr_db_vec, 10 * np.log10(mse_simp), "s--", label="LMMSE simplificado — Ec. (5)")
plt.plot(snr_db_vec, 10 * np.log10(mse_teo_simp), "kx", markersize=7, label="Teórico Ec. (5) — Ap. D")
plt.title("Comparación de MSE: LMMSE exacto vs. simplificado (16-QAM, N=64, L=4)")
plt.xlabel("SNR [dB]")
plt.ylabel("MSE normalizado [dB]")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# Figura 2: Seguimiento del canal en una realización puntual (SNR = 10 dB)
tonos = np.arange(N)
plt.figure(figsize=(8, 4.5))
plt.plot(tonos, np.abs(ejemplo["h"]), "k-", linewidth=1.8, label="Canal real $|h_k|$")
plt.plot(tonos, np.abs(ejemplo["h_lmmse"]), "^--", markersize=4, label="LMMSE exacto — Ec. (3)")
plt.plot(tonos, np.abs(ejemplo["h_simp"]), "s:", markersize=4, label="LMMSE simplificado — Ec. (5)")
plt.title("Estimación de magnitud del canal por subportadora (SNR = 10 dB)")
plt.xlabel("Índice de subportadora $k$")
plt.ylabel("Magnitud $|h_k|$")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()