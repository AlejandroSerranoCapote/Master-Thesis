# -*- coding: utf-8 -*-
"""
Simulación de propagación de ondas en una guía óptica con cuatro tracks que siguen rectos en la última parte.
Se usa el método FFT-BPM para resolver la evolución del campo electromagnético.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
plt.style.use(['science', 'notebook'])

# =============================================================================
# Definir parámetros
# =============================================================================
N = 2**12  # Puntos en la malla espacial
L = 1000   # Longitud de la caja (micras)
dx = L / N  # Espaciado de la malla
x = np.arange(-L/2 + 1/N, L/2, dx)  # Vector de posiciones
dkx = 2 * np.pi / L  # Intervalo en la malla de momento
kx = np.fft.fftshift(np.fft.fftfreq(N, d=dx) * 2 * np.pi)  # Malla de momentos
xmax = max(x)

# Parámetros ópticos
wl = 1.064  # Longitud de onda (micras)
n0 = 2.2    # Índice de refracción base
k0 = 2 * np.pi / wl  # Número de onda en el vacío
K = n0 * k0  # Número de onda en el medio

# Parámetros de la fuente
w0 = 2.5  # Ancho de la gaussiana inicial
E_z = np.exp(-x**2 / (w0**2))  # Perfil inicial del campo eléctrico

# Parámetros de propagación
dz = 2   # Paso de propagación
zmax = 10000  # Distancia total de propagación (micras)
z = np.arange(0, zmax + dz, dz)  # Vector de propagación

# Cálculo del término de propagación
dz2 = dz / 2
kz = np.sqrt(K**2 - kx**2)
kz[np.isnan(kz)] = 0  # Corrección para valores negativos bajo la raíz

# =============================================================================
# Definir perfil de índice de refracción
# =============================================================================

# Parámetros de la guía
W0 = 5  # Ancho de la guía de entrada
W = 20  # Ancho de los canales finales
dn = 0.004  # Cambio en el índice de refracción
n1 = n0 - dn  # Índice de refracción más bajo

# Parámetros de la bifurcación
num_tracks = 4  # Número de guías laterales
track_width = 0.5  # Ancho de cada track
track_sep = 1.2  # Separación entre tracks

# Función de curvatura suave
def curve(z_val):
    return 50 * np.tanh((z_val - 2000) / zmax * 5)

# Creación de la malla del índice de refracción
n_profile = np.ones((len(z), len(x))) * n0

# Generación de la estructura óptica
start_time = time.time()
# Vectorización del perfil de índice de refracción
X, Z = np.meshgrid(x, z, indexing="xy")  # Crear malla 2D

# Sección recta inicial
mask_initial = Z < 2000

# Generar posiciones de los tracks a ambos lados de 0
x_positions = np.array([(W0 / 2 + i * track_sep) for i in range(1, num_tracks + 1)])
x_positions = np.concatenate([-x_positions, x_positions])  # Agregar los negativos para la izquierda

# Crear máscara booleana para los tracks en ambos lados
mask_tracks = (np.abs(X[..., None] - x_positions) < track_width / 2).any(axis=-1)

# Aplicar condiciones
n_profile = np.full_like(X, n0)  # Inicializar con n0
n_profile[mask_initial & mask_tracks] = n1  # Asignar n1 en las regiones deseadas

# Aplicar solo en la región donde zi_val < 6000
mask_curve = (Z >= 2000) & (Z < 6000)  

# Calcular la anchura variable en función de z
width = W0 / 2 + curve(Z)  # Vectorizamos la curva en toda la malla Z

# Generar posiciones de los tracks curvados (a ambos lados de 0)
x_positions_curve = width[..., None] + np.array([i * track_sep for i in range(1, num_tracks + 1)])
x_positions_curve = np.concatenate([-x_positions_curve, x_positions_curve], axis=-1)  # Simetría

# Crear máscara para los tracks en la región curva
mask_tracks_curve = (np.abs(X[..., None] - x_positions_curve) < track_width / 2).any(axis=-1)

# Aplicar la condición en la región curva
n_profile[mask_curve & mask_tracks_curve] = n1

# Máscara para la región de transición (zi_val >= 6000)
mask_transition = Z >= 6000  

# Cálculo del ancho variable para la transición
width_final = W0 / 2 + curve(Z)  # Vectorizamos la curva en toda la malla Z

# Generar posiciones de los tracks en la transición
x_positions_final = width_final[..., None] + np.array([i * track_sep for i in range(1, num_tracks + 1)])
x_positions_final = np.concatenate([-x_positions_final, x_positions_final], axis=-1)  # Simetría

# Generar posiciones para la parte central (última región)
center = 0
x_positions_center = center + np.array([W / 2 + i * track_sep for i in range(1, num_tracks + 1)])
x_positions_center = np.concatenate([-x_positions_center, x_positions_center], axis=-1)  # Simetría

# Crear máscaras booleanas
mask_tracks_final = (np.abs(X[..., None] - x_positions_final) < track_width / 2).any(axis=-1)
mask_tracks_center = (np.abs(X[..., None] - x_positions_center) < track_width / 2).any(axis=-1)

# Aplicar la condición en la región de transición
n_profile[mask_transition & (mask_tracks_final | mask_tracks_center)] = n1
# Precalcular dn²
dn2_profile = n_profile**2 - n0**2

# Paso inicial (propagación en medio paso)
E_z = np.fft.ifft(np.fft.fftshift(np.fft.fft(E_z)) * np.exp(-1j * kz * dz2))

# =============================================================================
# Algoritmo FFT-BPM
# =============================================================================
I_z = np.zeros((len(z), N))  # Matriz de intensidades

for n in range(len(z)):
    dn2 = dn2_profile[n, :]
    lens_corrector_op = np.exp((-1j * k0 * dn2 * dz) / (2 * n0))  # Corrección de lente dinámica
    
    E_z = np.fft.ifft(np.fft.fft(lens_corrector_op * np.fft.ifft(
        np.exp(-1j * kz * dz2) * np.fft.fft(E_z))) * np.exp(-1j * kz * dz2))
    I_z[n, :] = np.abs(E_z)**2

# Tiempo de ejecución
end_time = time.time()
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")


# # Intensidad de propagación
# plt.figure(figsize=(10, 6))
# plt.imshow(I_z, extent=[-L/2, L/2, 0, zmax],
#            aspect='auto', origin='lower', cmap='viridis')
# plt.xlabel('x $(\mu m)$')
# plt.ylabel('z $(\mu m)$')
# plt.colorbar(label='Intensidad')
# plt.title('Propagación en guía con tres canales')
# plt.show()

# Índice de refracción
# plt.figure(figsize=(10, 6))
# plt.imshow(n_profile, extent=[-L/2, L/2, 0, zmax],
#             aspect='auto', origin='lower', cmap='coolwarm')
# plt.colorbar(label='Índice de refracción n(x, z)')
# plt.xlabel('x $(\mu m)$')
# plt.ylabel('z $(\mu m)$')
# plt.title('Perfil del índice de refracción')
# plt.show()

# # Perfil de intensidad a la salida
# plt.figure()
# plt.plot(I_z[-1, :])
# plt.title('Perfil de intensidad a la salida')
# plt.xlabel('x $(\mu m)$')
# plt.ylabel('Intensidad')
# plt.show()

#Graficamos el perfil de intensidad a la salida de la guía de onda
x = np.linspace(-xmax,xmax,len(I_z[-1,:]))
plt.figure(figsize=(14,7))
plt.subplot(2,2,1)
plt.imshow(I_z, extent=[-xmax, xmax, 0, zmax], aspect='auto', origin='lower', cmap='viridis')
plt.xlabel('x $(\\mu m)$')
plt.ylabel('z $(\\mu m)$')
plt.colorbar(label='Intensidad')
plt.title('Propagación usando el método FFT BPM')
plt.subplot(2,2,3)
plt.imshow(n_profile, extent=[-xmax, xmax, 0, zmax], aspect='auto', origin='lower', cmap='coolwarm')
plt.colorbar(label='Índice de refracción n(x, z)')
plt.xlabel('x $(\\mu m)$')
plt.ylabel('z $(\\mu m)$')
plt.title('Perfil del índice de refracción n(x, z)')
plt.subplot(2,2,2)
plt.title("Intensidad a la entrada")
plt.grid()
plt.plot(x,I_z[0,:])
plt.xlim(xmin=-50,xmax=50)
plt.ylabel('Intensidad (u.arb)')
plt.subplot(2,2,4)
plt.plot(x,I_z[-1,:])
plt.grid()
plt.title("Intensidad a la salida")
plt.xlabel('x $(\\mu m)$')
plt.ylabel('Intensidad (u.arb)')
plt.xlim(xmin=-200,xmax=200)
plt.show()