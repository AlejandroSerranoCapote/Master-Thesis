# -*- coding: utf-8 -*-
"""
Algoritmo FFT-BPM (Guía de ondas con perfil de índice de refracción dependiente de x y z)
Optimizado para permitir un número arbitrario de tracks.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from math import *
plt.style.use(['science', 'notebook'])

# =============================================================================
# Definir parámetros
# =============================================================================
N = 2**10 #Puntos de la malla x
L = 200    #Longitud de la caja
dx = L / N  # intervalo en la malla de posiciones
x = np.arange(-L/2 + 1/N, L/2, dx)  # malla de posiciones (centrada en el origen)
xmax = max(x)

wl = 1.064 # longitud de onda en micras
k0 = 2 * np.pi / wl  # número de onda en el vacío

n0 = 2.2  # índice de refracción base del medio
dn = 0.01  # modificación en el índice de refracción
n1 = n0 - dn  # índice de refracción más bajo

dkx = 2 * np.pi / L  # intervalo en la malla de momento
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # malla de momentos con FFT
kx = np.fft.fftshift(kx)  # Centrar la malla de momentos en cero
K = n0 * k0  # número de onda en el medio

w0 = 2 # ancho de la gaussiana
E_z = np.exp(-x**2 / (2 * w0**2))

dz = 0.5  # paso de propagación (debe ser pequeño para evitar artefactos)
zmax = 1000  # distancia de propagación en micras
z = np.arange(0, zmax + dz, dz)  # vector de propagación

# Término de la propagación en el dominio de la frecuencia
kz = np.sqrt(K**2 - kx**2)
kz[np.isnan(kz)] = 0  # Asignar kz=0 donde haya valores negativos bajo la raíz

I_z = np.zeros((len(z), N))  # matriz para almacenar la intensidad

# =============================================================================
# Crear un perfil de índice de refracción n(x, z)
# =============================================================================

n_profile = np.ones((len(z), len(x)))*n0  # Inicializar matriz del índice de refracción

width = 0.3  # Ancho de cada track en micras
separation = 1.5  # Separación entre los tracks en micras
offset = 2 # Desplazamiento inicial (micras)
num_tracks = 5  # Número de tracks a cada lado del centro

# Función que define los tracks de la guía de onda
def f(z_val):
    """
    FUNCION QUE MODELA LOS TRACKS DE LA GUÍA DE ONDA
    """
    # return 5*asin(z_val /zmax)
    # return 10*np.sin(z_val /800)**2
    return (z_val/400)**2
    # return (z_val/200)
    # return 5*np.sqrt(z_val/500)

# Medir tiempo de inicio
start_time = time.time()

# Crear tracks simétricos y con apertura en las guías de la derecha
for zi, zi_val in enumerate(z):
    for xi, xi_val in enumerate(x):
        # Definir los centros de los tracks
        centers_left = [-(offset + i * separation) - f(zi_val) for i in range(num_tracks)]
        centers_right = [(offset + i * separation) + f(zi_val) for i in range(num_tracks)]
        
        # Combinar los centros de las guías izquierda y derecha
        centers = centers_left + centers_right

        # Verificar si el punto pertenece a algún track
        for center in centers:
            if np.abs(xi_val - center) < width:
                n_profile[zi, xi] = n1
                break  # Salir del bucle si ya pertenece a un track
                
# Precalcular dn2
dn2_profile = n_profile**2 - n0**2

# =============================================================================
# ALGORITMO FFT-BPM
# =============================================================================
# Paso inicial (propagación en medio paso)
E_z = np.fft.ifft(np.fft.fftshift(np.fft.fft(E_z)) * np.exp(-1j * kz * dz / 2))

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

# Graficar intensidad en 2D
plt.figure(figsize=(10, 6))
plt.imshow(I_z, extent=[-xmax, xmax, 0, zmax], aspect='auto', origin='lower', cmap='viridis')
plt.xlabel('x $(\\mu m)$')
plt.ylabel('z $(\\mu m)$')
plt.colorbar(label='Intensidad')
plt.title('Propagación usando el método FFT BPM')
plt.show()

# Graficar índice de refracción n(x, z)
plt.figure(figsize=(10, 6))
plt.imshow(n_profile, extent=[-xmax, xmax, 0, zmax], aspect='auto', origin='lower', cmap='coolwarm')
plt.colorbar(label='Índice de refracción n(x, z)')
plt.xlabel('x $(\\mu m)$')
plt.ylabel('z $(\\mu m)$')
plt.title('Perfil del índice de refracción n(x, z)')
plt.show()

#Graficamos el perfil de intensidad a la salida de la guía de onda
plt.figure()
plt.plot(I_z[-1,:])
plt.show()