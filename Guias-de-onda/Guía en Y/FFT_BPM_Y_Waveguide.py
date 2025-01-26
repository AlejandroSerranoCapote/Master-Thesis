
# -*- coding: utf-8 -*-
"""
Algoritmo FFT-BPM (Guía de ondas con perfil de índice de refracción en forma de Y)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
plt.style.use(['science', 'notebook'])

# =============================================================================
# Definir parámetros
# =============================================================================
N = 2**14 # puntos de la malla x
L = 200  # longitud de la caja
dx = L / N  # intervalo en la malla de posiciones
x = np.arange(-L/2 + 1/N, L/2, dx)  # malla de posiciones (centrada en el origen)
xmax = max(x)

wl = 1.064  # longitud de onda en micras
n0 = 1.5  # índice de refracción base del medio
k0 = 2 * np.pi / wl  # número de onda en el vacío

dkx = 2 * np.pi / L  # intervalo en la malla de momento
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # malla de momentos con FFT
kx = np.fft.fftshift(kx)  # Centrar la malla de momentos en cero
K = n0 * k0  # número de onda en el medio

w0 = 2.5 # ancho de la gaussiana
E_z = np.exp(-x**2 / (2 * w0**2))

dz = 0.5  # paso de propagación (debe ser pequeño para evitar artefactos)
zmax = 1000  # distancia de propagación en micras
z = np.arange(0, zmax + dz, dz)  # vector de propagación

# Término de la propagación en el dominio de la frecuencia
kz = np.sqrt(K**2 - kx**2)
kz[np.isnan(kz)] = 0  # Asignar kz=0 donde haya valores negativos bajo la raíz

I_z = np.zeros((len(z), N))  # matriz para almacenar la intensidad

# =============================================================================
# Crear un perfil de índice de refracción n(x, z) en forma de Y
# =============================================================================
dn = 0.01 # modificación en el índice de refracción
n1 = n0 + dn  # índice de refracción más bajo

n_profile = np.ones((len(z), len(x))) * n0  # Inicializar matriz del índice de refracción

waveguide_width = 4 #Micras

# Crear la estructura en forma de "Y"
for zi, zi_val in enumerate(z):
    if zi_val < 300:  # Región inicial recta
        mask = np.abs(x) < waveguide_width  # Guía central recta
    elif 300 <= zi_val < 600:  # Región de bifurcación
        mask = (np.abs(x - (zi_val - 300) / 15) < waveguide_width) | (np.abs(x + (zi_val - 300) / 15) < waveguide_width)
    else:  # Región final con dos ramas
        mask = (np.abs(x - 20) < waveguide_width) | (np.abs(x + 20) < waveguide_width)

    n_profile[zi, mask] = n1

# Precalcular dn²
dn2_profile = n_profile**2 - n0**2

# Paso inicial (propagación en medio paso)
E_z = np.fft.ifft(np.fft.fftshift(np.fft.fft(E_z)) * np.exp(-1j * kz * dz / 2))

# =============================================================================
# ALGORITMO FFT-BPM
# =============================================================================

# Medir tiempo de inicio
start_time = time.time()

# Propagación completa usando FFT-BPM
for n in range(len(z)):
    dn2 = dn2_profile[n, :]  # Perfil dn2 para el paso actual
    lens_corrector_op = np.exp((-1j * k0 * dn2 * dz) / (2 * n0))  # Operador lente dinámico
    
    E_z = np.fft.ifft(np.fft.fft(lens_corrector_op * np.fft.ifft(
        np.exp(-1j * kz * dz / 2) * np.fft.fft(E_z))) * np.exp(-1j * kz * dz / 2))
    I_z[n, :] = np.abs(E_z)**2

# Medir tiempo de finalización
end_time = time.time()
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

# =============================================================================
# PLOTS
# =============================================================================

# Graficar intensidad
plt.figure(figsize=(10, 6)) 
plt.imshow(I_z, extent=[-xmax, xmax, 0, zmax], aspect='auto', origin='lower', cmap='viridis')
plt.xlabel('x $(\\mu m)$')
plt.ylabel('z $(\\mu m)$')
plt.colorbar(label='Intensidad')
plt.title('Propagación en estructura en forma de Y')
plt.show()

# Graficar índice de refracción n(x, z)
plt.figure(figsize=(10, 6))
plt.imshow(n_profile, extent=[-xmax, xmax, 0, zmax], aspect='auto', origin='lower', cmap='coolwarm')
plt.colorbar(label='Índice de refracción n(x, z)')
plt.xlabel('x $(\\mu m)$')
plt.ylabel('z $(\\mu m)$')
plt.title('Perfil del índice de refracción en forma de Y')
plt.show()

#Graficamos el perfil de intensidad a la salida de la guía de onda
plt.figure()
plt.plot(I_z[-1,:])
plt.show()