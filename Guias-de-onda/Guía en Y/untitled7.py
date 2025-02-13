# -*- coding: utf-8 -*-
"""
Simulación de propagación de ondas en una guía óptica con transición suave y división en tres canales finales.
Se usa el método FFT-BPM para resolver la evolución del campo electromagnético.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
plt.style.use(['science', 'notebook'])

# =============================================================================
# Definir parámetros
# =============================================================================
N = 2**11 # Puntos en la malla espacial
L = 1000   # Longitud de la caja (micras)
dx = L / N  # Espaciado de la malla
x = np.arange(-L/2 + 1/N, L/2, dx)  # Vector de posiciones
dkx = 2 * np.pi / L  # Intervalo en la malla de momento
kx = np.fft.fftshift(np.fft.fftfreq(N, d=dx) * 2 * np.pi)  # Malla de momentos

# Parámetros ópticos
wl = 1.064  # Longitud de onda (micras)
n0 = 2.2    # Índice de refracción base
k0 = 2 * np.pi / wl  # Número de onda en el vacío
K = n0 * k0  # Número de onda en el medio

# Parámetros de la fuente
w0 = 5  # Ancho de la gaussiana inicial
E_z = np.exp(-x**2 / (2 * w0**2))  # Perfil inicial del campo eléctrico

# Parámetros de propagación
dz = 5   # Paso de propagación
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
W = 30  # Ancho de los canales finales
dn = 0.004  # Cambio en el índice de refracción
n1 = n0 - dn  # Índice de refracción más bajo

# Parámetros de la bifurcación
num_tracks = 4  # Número de guías laterales
track_width = 0.5  # Ancho de cada track
track_sep = 1.0  # Separación entre tracks

# Función de curvatura suave
def curve(z_val):
    return 50 * np.tanh((z_val - 2000) / zmax * 5)

# Creación de la malla del índice de refracción
n_profile = np.ones((len(z), len(x))) * n0

# Generación de la estructura óptica
start_time = time.time()
for zi, zi_val in enumerate(z):
    for xi, xi_val in enumerate(x):
        if zi_val < 2000:
            # Sección recta inicial
            for i in range(1, num_tracks + 1):
                if abs(xi_val - (W0 / 2 + i * track_sep)) < track_width / 2 or \
                   abs(xi_val + (W0 / 2 + i * track_sep)) < track_width / 2:
                    n_profile[zi, xi] = n1
        
        elif zi_val < 6000:
            # Expansión curva intermedia
            width = W0 / 2 + curve(zi_val)
            for i in range(1, num_tracks + 1):
                if abs(xi_val - (width + i * track_sep)) < track_width / 2 or \
                   abs(xi_val + (width + i * track_sep)) < track_width / 2:
                    n_profile[zi, xi] = n1
        
        else:
            # División en tres canales en la zona final
            width_final = W0 / 2 + curve(zi_val)  # Continuación progresiva de la expansión
            centers = [-width_final, 0, width_final]  # Posiciones centrales de los 3 canales
            for center in centers:
                for i in range(1, num_tracks + 1):
                    if abs(xi_val - (center + i * track_sep)) < track_width / 2 or \
                       abs(xi_val - (center - i * track_sep)) < track_width / 2:
                        n_profile[zi, xi] = n1

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




# Intensidad de propagación
plt.figure(figsize=(10, 6)) 
plt.imshow(I_z, extent=[-L/2, L/2, 0, zmax], aspect='auto', origin='lower', cmap='viridis')
plt.xlabel('x $(\mu m)$')
plt.ylabel('z $(\mu m)$')
plt.colorbar(label='Intensidad')
plt.title('Propagación en guía con tres canales')
plt.show()

# Índice de refracción
plt.figure(figsize=(10, 6))
plt.imshow(n_profile, extent=[-L/2, L/2, 0, zmax], aspect='auto', origin='lower', cmap='coolwarm')
plt.colorbar(label='Índice de refracción n(x, z)')
plt.xlabel('x $(\mu m)$')
plt.ylabel('z $(\mu m)$')
plt.title('Perfil del índice de refracción')
plt.show()

# Perfil de intensidad a la salida
plt.figure()
plt.plot(I_z[-1, :])
plt.title('Perfil de intensidad a la salida')
plt.xlabel('x $(\mu m)$')
plt.ylabel('Intensidad')
plt.show()