# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:42:07 2025

@author: Alejandro

Algoritmo FFT-BPM (Vacío)
"""

import numpy as np
import matplotlib.pyplot as plt

# Definir parámetros
N = 100**2 # puntos de la malla x
L = 200 # longitud de la caja Aumentarla elimina las reflexiones internas
dx = L / N  # intervalo en la malla de posiciones
x = np.arange(-L/2 + 1/N, L/2, dx)  # malla de posiciones (centrada en el origen)
xmax = max(x)

wl = 1 # longitud de onda en micras
n0 = 1  # índice de refracción del vacío
k0 = 2 * np.pi / wl  # número de onda en el vacío

dkx = 2 * np.pi / L  # intervalo en la malla de momento
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # malla de momentos con FFT
kx = np.fft.fftshift(kx)  # Centrar la malla de momentos en cero
K = n0 * k0  # número de onda en el medio

w0 = 4 # ancho de la gaussiana
E_z = np.exp(-x**2 / (2 * w0**2))

dz = 2 # paso de propagación
zmax = 1000 #distancia de propagación en micras
z = np.arange(0, zmax + dz, dz)  # vector de propagación

# Termino de la propagación en el dominio de la frecuencia
kz = np.sqrt(K**2 - kx**2)
kz[np.isnan(kz)] = 0  # Asignar kz=0 donde haya valores negativos bajo la raíz
# Propagación usando el método Split Step
I_z = np.zeros((len(z), N))  # matriz para almacenar la intensidad

# Paso inicial (propagación en medio paso)
E_z = np.fft.ifft(np.fft.fftshift(np.fft.fft(E_z)) * np.exp(-1j * kz * dz / 2))

# Propagación completa usando el método FFT-BPM
for n in range(len(z)):
    E_z = np.fft.ifft(np.exp(-1j * kz * dz) * np.fft.fft(E_z))    
    I_z[n, :] = np.abs(E_z)**2

# Graficar
plt.figure(figsize=(8, 6))
plt.imshow(I_z, extent=[-xmax, xmax, 0, zmax], aspect='auto', origin='lower', cmap='viridis')
plt.xlabel('x $(\\mu m)$')
plt.ylabel('z $(\\mu m)$')
plt.colorbar(label='Intensidad')
plt.title('Propagación usando el método FFT BPM')
plt.show()
