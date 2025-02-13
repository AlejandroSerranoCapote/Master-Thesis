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
N = 2**12# puntos de la malla x
L = 1000  # longitud de la caja
dx = L / N  # intervalo en la malla de posiciones
x = np.arange(-L/2 + 1/N, L/2, dx)  # malla de posiciones (centrada en el origen)
xmax = max(x)

wl = 1.064  # longitud de onda en micras
n0 = 2.2  # índice de refracción base del medio
k0 = 2 * np.pi / wl  # número de onda en el vacío

dkx = 2 * np.pi / L  # intervalo en la malla de momento
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # malla de momentos con FFT
kx = np.fft.fftshift(kx)  # Centrar la malla de momentos en cero
K = n0 * k0  # número de onda en el medio

w0 = 5 # ancho de la gaussiana
E_z = np.exp(-x**2 / (w0**2))

dz = 2 # paso de propagación (debe ser pequeño para evitar artefactos)
zmax = 10000 # distancia de propagación en micras
z = np.arange(0, zmax + dz, dz)  # vector de propagación

# Término de la propagación en el dominio de la frecuencia
kz = np.sqrt(K**2 - kx**2)
kz[np.isnan(kz)] = 0  # Asignar kz=0 donde haya valores negativos bajo la raíz


I_z = np.zeros((len(z), N))  # matriz para almacenar la intensidad

# =============================================================================
# Crear un perfil de índice de refracción n(x, z) en forma de Y
# =============================================================================
# Parámetros de la guía
W0 = 5 # Ancho de la guía de entrada
W = 40   # Ancho de las ramas

dn = 0.004 # modificación en el índice de refracción
n1 = n0 - dn  # índice de refracción más bajo


track_width = 0.5  # Ancho de cada track
track_sep = 1.0  # Separación entre tracks
num_tracks = 4  # Número de tracks en cada lado


# Creación del perfil de índice de refracción
n_profile = np.ones((len(z), len(x))) * n0

# Función para la curva de bifurcación
def curve(z_val):
    return 50* np.tanh((z_val - 2000) / zmax*5)  # Curvatura suave # Curvatura suave 350 ta de locos

# Medir tiempo de inicio
start_time = time.time()
# Generar la guía óptica con tracks
for zi, zi_val in enumerate(z):
    for xi, xi_val in enumerate(x):
        if zi_val < 2000:
            # Parte recta
            for i in range(1, num_tracks + 1):
                if abs(xi_val - (W0 / 2 + i * track_sep)) < track_width / 2 or abs(xi_val + (W0 / 2 + i * track_sep)) < track_width / 2:
                    n_profile[zi, xi] = n1
                    
        elif zi_val < 6000:
            # Expansión y curvatura
            width = W0 / 2 + curve(zi_val)

            for i in range(1, num_tracks + 1):
                if abs(xi_val - (width + i * track_sep)) < track_width / 2 or abs(xi_val + (width + i * track_sep)) < track_width / 2:
                    n_profile[zi, xi] = n1
                    
        else:
            # División en tres ramas
            centers = [-60,60]  # Posiciones de las ramas
            for center in centers:
                for i in range(1, num_tracks + 1):
                    if abs(xi_val - (center + W / 2 + i * track_sep)) < track_width / 2 or abs(xi_val - (center - W / 2 - i * track_sep)) < track_width / 2:
                          n_profile[zi, xi] = n1

# Precalcular dn²
dn2_profile = n_profile**2 - n0**2

# Paso inicial (propagación en medio paso)
E_z = np.fft.ifft(np.fft.fftshift(np.fft.fft(E_z)) * np.exp(-1j * kz * dz / 2))

# =============================================================================
# ALGORITMO FFT-BPM
# =============================================================================



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