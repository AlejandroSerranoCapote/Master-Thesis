# -*- coding: utf-8 -*-
"""
Simulación de propagación de ondas en una guía óptica mediante el método
FFT-BPM.

@autor: Alejandro Serrano Capote 

email: alejandro.serrano1610@gmail.com
"""

'''
    IMPORTS DE LOS PAQUETES NECESARIOS
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.integrate as integrate

start_time = time.time() #Empezamos a medir el tiempo de ejecución del programa.

'''
    INICIALIZACIÓN DE PARÁMETROS ÓPTICOS Y COMPUTACIONALES
'''
# =============================================================================
#  Parámetros computacionales
# =============================================================================
N = 4000                                                   # Puntos en la malla espacial
L = 200                                                    # Longitud de la caja (micras)
dx = L / N                                                 # Espaciado de la malla
x = np.arange(-L/2 + 1/N, L/2, dx)                         # Vector de posiciones
dkx = 2 * np.pi / L                                        # Intervalo en la malla de momento
kx = np.fft.fftshift(np.fft.fftfreq(N, d=dx) * 2 * np.pi)  # Malla de momentos
xmax = max(x)                                              # Valor máximo de x

# =============================================================================
# Parámetros ópticos
# =============================================================================
wl = 0.633                      # Longitud de onda (micras)

########
# Ytrium Aluminium Garnet (YAG) Isótropo
########

#n0 = 1.8295 # 633  nm
#n0 = 1.8198 # 850  nm
#n0 = 1.8077 # 1500 nm
#n0 = 1.7656# 3700 nm

########
# Niobatio de litio (LiNbO3) Anisótropo
########

n0 = 2.2864 # 633  nm
#n0 = 2.2494 # 850  nm
#n0 = 2.2128 # 1500 nm
#n0 = 2.1299 # 3700 nm

k0 = 2 * np.pi / wl                 # Número de onda en el vacío
K = n0 * k0                         # Número de onda en el medio
dn = 0.004                          # Cambio en el índice de refracción
n1 = n0 - dn                        # Índice de refracción más bajo

# =============================================================================
# Parámetros de la fuente
# =============================================================================

w0 = 2.5                         # Ancho de la gaussiana inicial
E_z = np.exp(-x**2 / (w0**2))       # Perfil inicial del campo eléctrico

# =============================================================================
# Parámetros de propagación
# ArithmeticError=========================================
W0 = 15                           # Ancho de la guía recta de entrada
dz = 0.25                              # Paso de propagación (micras)
zmax = 1000                      # Distancia total de propagación (micras)
z = np.arange(0, zmax + dz, dz)     # Vector de propagación

# =============================================================================
# Cálculo del término de propagación
# =============================================================================
dz2 = dz / 2
kz = np.sqrt(K**2 - kx**2)          # Vector de propagación
kz[np.isnan(kz)] = 0                # Corrección para valores negativos bajo la raíz


'''
    DEFINIR EL PERFIL DE ÍNDICE DE REFRACCIÓN (CONSTRUCCIÓN DE LA GUÍA)
'''

# =============================================================================
# Parámetros asociados a los tracks que conforman la guía
# =============================================================================
num_tracks =  4                # Número de tracks
track_width = 0.5              # Ancho de cada track
track_sep = 2                  # Separación entre tracks

# =============================================================================
# Creación de la malla 2D inicializada con un índice de refracción n0.
# =============================================================================
X, Z = np.meshgrid(x, z, indexing="xy")             # Crear malla 2D
n_profile = np.full_like(X, n0)                     # Inicializar con n0
zona_recta = 12000                                  # Longitud en micras de la zona recta

# =============================================================================
# ZONA RECTA INICIAL
# =============================================================================

mask_initial = Z <  zona_recta       # Sección recta inicial

# Generar posiciones de los tracks a ambos lados de 0
x_positions = np.array([(W0 / 2 + i * track_sep) for i in range(1, num_tracks+1)])
x_positions = np.concatenate([-x_positions, x_positions])  # Agregar los negativos para la izquierda

# Crear máscara booleana para los tracks en ambos lados
mask_tracks = (np.abs(X[..., None] - x_positions) < track_width / 2).any(axis=-1)
n_profile[mask_initial & mask_tracks] = n1                 # Asignar n1 en las regiones que nos interesen


'''
    ALGORITMO FFT-BPM
'''
# Precalcular dn2
dn2_profile = n_profile**2 - n0**2

# Paso inicial
E_z = np.fft.ifft(np.fft.fftshift(np.fft.fft(E_z)) * np.exp(-1j * kz * dz2))
I_inicial = np.abs(E_z)**2

I_z = np.zeros((len(z), N))  # Matriz de intensidades

for n in range(len(z)):
    dn2 = dn2_profile[n, :]
    lens_corrector_op = np.exp((-1j * k0 * dn2 * dz) / (2 * n0))  #Operador de lente
    
    E_z = np.fft.ifft(np.fft.fft(lens_corrector_op * np.fft.ifft(
        np.exp(-1j * kz * dz2) * np.fft.fft(E_z))) * np.exp(-1j * kz * dz2)) #Actualización del campo por FFT
    I_z[n, :] = np.abs(E_z)**2

# Tiempo de ejecución
end_time = time.time()
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

'''
    GRÁFICOS
'''

# epsilon = 0.001
# log_I_z = np.log10(I_z / np.max(I_z) + epsilon)

# plt.figure(figsize=(10, 6))
# plt.imshow(log_I_z, extent=[-xmax, xmax, 0, zmax], aspect='auto', origin='lower', cmap='jet',interpolation=None)
# plt.xlabel('x $(\\mu m)$')
# plt.ylabel('z $(\\mu m)$')
# plt.colorbar(label='Log(Intensidad)')
# plt.title('Propagación usando el método FFT BPM')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.imshow(I_z, extent=[-xmax, xmax, 0, zmax], aspect='auto', origin='lower', cmap='jet',interpolation=None)
# plt.xlabel('x $(\\mu m)$')
# plt.ylabel('z $(\\mu m)$')
# plt.colorbar(label='Intensidad')
# plt.title('Propagación usando el método FFT BPM')
# plt.show()


# plt.figure(figsize=(10, 6))
# plt.imshow(n_profile, extent=[-xmax, xmax, 0, zmax], aspect='auto', origin='lower', cmap='coolwarm')
# plt.colorbar(label='Índice de refracción n(x, z)')
# plt.xlabel('x $(\\mu m)$')
# plt.ylabel('z $(\\mu m)$')
# plt.title('Perfil del índice de refracción n(x, z)')
# plt.show()

# plt.figure()
# plt.plot(x,I_z[-1,:])
# plt.grid()
# plt.title("Intensidad a la salida")
# plt.xlabel('x $(\\mu m)$')
# plt.ylabel('Intensidad (u.arb)')
# plt.xlim(xmin=-150,xmax=150)

I1 = integrate.trapz(I_z[-1,:][1550:2500]) #Integramos I(x)*dx
I2 = integrate.trapz(I_inicial)            #Integramos I(x)*dx
print('P/P_0 -->',I1/I2)
plt.figure()
plt.plot(x,I_inicial,'b--',label='Espectro a la entrada')
plt.plot(x,I_z[-1,:],'r',label='Espectro a la salida')
# plt.plot(x[1550:2500],I_z[-1,:][1550:2500],'r',label='Espectro a la salida')
plt.xlim(xmin=-100,xmax=100)
plt.legend(frameon=True)
plt.grid()
plt.xlabel('x $(\\mu m)$')
plt.ylabel('Intensidad (u.arb)')
plt.show()
print('#'*40)