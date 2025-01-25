# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:42:07 2025

@author: Alejandro

Algoritmo FFT-BPM (Guía de ondas con perfil de índice de refracción escalón)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time
plt.style.use(['science','notebook'])

# Definir parámetros
N = 200**2  # puntos de la malla x
L = 200  # longitud de la caja
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

w0 = 4# ancho de la gaussiana
E_z = np.exp(-x**2 / (2 * w0**2))

dz = 0.5# paso de propagación (ESTO SI ES MUY GRUESO METE ARTEFACTOS EN LA PROPAGACIÓN OJO)
zmax = 1000 #distancia de propagación en micras
z = np.arange(0, zmax + dz, dz)  # vector de propagación

# Termino de la propagación en el dominio de la frecuencia
kz = np.sqrt(K**2 - kx**2)
kz[np.isnan(kz)] = 0  # Asignar kz=0 donde haya valores negativos bajo la raíz

# Propagación usando el método Split Step
I_z = np.zeros((len(z), N))  # matriz para almacenar la intensidad

# Crear el perfil del índice de refracción
n0 = 2.2  # índice de refracción del medio
dn = 0.003 #modificación en el índice de refracción
n1 = n0 - dn  # índice de refracción de la franja

# Crear un perfil de índice de refracción con una franja rectangular
width = 10  # Ancho de la franja rectangular en micras
smooth_width = 1.5  # Ancho de la transición suave en micras

n_profile = np.ones(len(x)) * n0  # Inicializar el perfil con n0


# Usar una función sigmoide (tanh) para suavizar los bordes
for i, xi in enumerate(x):
    # Definir el índice de refracción dentro de la franja rectangular
    # y aplicar una transición suave con tanh
    if np.abs(xi - 15) < width / 2:
        n_profile[i] = n1 + dn * 0.5 * (1 + np.tanh((np.abs(xi - 15) - (width / 2 - smooth_width)) / smooth_width))
    elif np.abs(xi + 15) < width / 2:
        n_profile[i] = n1 + dn * 0.5 * (1 + np.tanh((np.abs(xi + 15) - (width / 2 - smooth_width)) / smooth_width))
                
# for i, xi in enumerate(x):
#     # Definir el índice de refracción dentro de la franja rectangular
#     if np.abs(xi-15) < width / 2 or  np.abs(xi+15) < width / 2:
#         n_profile[i] = n1  
        

dn2 = n_profile**2 - n0

lens_corrector_op = np.exp((-1j*k0*dn2*dz)/(2*n0))  #Operador correcto lente

# Paso inicial (propagación en medio paso)
E_z = np.fft.ifft(np.fft.fftshift(np.fft.fft(E_z)) * np.exp(-1j * kz * dz / 2))

# =============================================================================
# ALGORITMO DE FFT BPM
# =============================================================================

# Medir tiempo de inicio
start_time = time.time()

# Propagación completa usando el método FFT-BPM
for n in range(len(z)):
    E_z = np.fft.ifft(np.fft.fft(lens_corrector_op*np.fft.ifft(np.exp(-1j * kz * dz/2) * np.fft.fft(E_z))) * np.exp(-1j * kz * dz/2))
    I_z[n, :] = np.abs(E_z)**2
    
# Medir tiempo de finalización
end_time = time.time()
# Imprimir tiempo de ejecución
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

# =============================================================================
# PLOTS
# =============================================================================

# Graficar
plt.figure(figsize=(10, 6))
plt.imshow(I_z, extent=[-xmax, xmax, 0, zmax], aspect='auto', origin='lower', cmap='viridis')
plt.xlabel('x $(\\mu m)$')
plt.ylabel('z $(\\mu m)$')
plt.colorbar(label='Intensidad')
plt.title('Propagación usando el método FFT BPM')
plt.show()

plt.figure(figsize=(8,4))
plt.plot(x,n_profile)
plt.xlabel('x $(\\mu m)$')
plt.ylabel('n(x)')
plt.show()