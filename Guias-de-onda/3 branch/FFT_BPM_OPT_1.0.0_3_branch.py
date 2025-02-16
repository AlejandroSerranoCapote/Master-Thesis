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
plt.style.use(['science', 'notebook'])

start_time = time.time() #Empezamos a medir el tiempo de ejecución del programa.

'''
    INICIALIZACIÓN DE PARÁMETROS ÓPTICOS Y COMPUTACIONALES
'''

# =============================================================================
#  Parámetros computacionales
# =============================================================================
N = 2000                                                   # Puntos en la malla espacial
L = 1000                                                   # Longitud de la caja (micras)
dx = L / N                                                 # Espaciado de la malla
x = np.arange(-L/2 + 1/N, L/2, dx)                         # Vector de posiciones
dkx = 2 * np.pi / L                                        # Intervalo en la malla de momento
kx = np.fft.fftshift(np.fft.fftfreq(N, d=dx) * 2 * np.pi)  # Malla de momentos
xmax = max(x)                                              # Valor máximo de x

# =============================================================================
# Parámetros ópticos
# =============================================================================
wl = 1.064                         # Longitud de onda (micras)
n0 = 2.2321                          # Índice de refracción base
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
# =============================================================================
dz = 2                          # Paso de propagación (micras)
zmax = 12000                        # Distancia total de propagación (micras)
z = np.arange(0, zmax + dz, dz)     # Vector de propagación

# =============================================================================
# Cálculo del término de propagación
# =============================================================================
dz2 = dz / 2
kz = np.sqrt(K**2 - kx**2)          # Vector de propagación
kz[np.isnan(kz)] = 0                # Corrección para valores negativos bajo la raíz


'''
    DEFINIR EL PERFIL DE ÍNDICE DE REFRACCIÓN (CONSTRUCCIÓN DE LA GUÍA)
    
    La guía de ondas se construye en tres partes:
        -Guía recta inicial
        -Apertura suave para que el haz se expanda.
        -Divisor del haz para obtener diferentes canales por los que la luz salga.
'''

# =============================================================================
# Parámetros asociados a los tracks que conforman la guía
# =============================================================================
num_tracks = 4                    # Número de tracks
track_width = 0.5                   # Ancho de cada track
track_sep = 1.2                     # Separación entre tracks

# =============================================================================
# Creación de la malla 2D inicializada con un índice de refracción n0.
# =============================================================================
X, Z = np.meshgrid(x, z, indexing="xy")             # Crear malla 2D
n_profile = np.full_like(X, n0)                     # Inicializar con n0


#Separación de zonas de la guía
zona_recta = 2000      #Longitud en micras de la zona recta
zona_apertura = 6000   #Longitud en micras de la zona apertura
zona_split = 9000      #Longitud en micras donde comienza a bifurcarse la segunda etapa de la guía

# =============================================================================
# ZONA RECTA INICIAL
# =============================================================================
W0 = 18                           # Ancho de la guía recta de entrada
mask_initial = Z < zona_recta       # Sección recta inicial

# Generar posiciones de los tracks a ambos lados de 0
x_positions = np.array([(W0 / 2 + i * track_sep) for i in range(1, num_tracks + 1)])
x_positions = np.concatenate([-x_positions, x_positions])  # Agregar los negativos para la izquierda

# Crear máscara booleana para los tracks en ambos lados
mask_tracks = (np.abs(X[..., None] - x_positions) < track_width / 2).any(axis=-1)

n_profile[mask_initial & mask_tracks] = n1                 # Asignar n1 en las regiones deseadas

# =============================================================================
# ZONA CURVA PRIMERA BIFURCACIÓN
# =============================================================================
W = 18                            # ancho de los canales de la guía

# Función de curvatura suave
def curve(z_val):
    return 70 * np.sin((z_val - zona_recta) / zmax * 4)**2

# Aplicar solo en la región que tengamos la apertura suave
mask_curve = (Z >= zona_recta) & (Z < zona_apertura)

width = W0/2 + curve(Z) #anchura variable en función de z

# Generar posiciones de los tracks exteriores curvados
x_positions_curve = width[..., None] + np.array([i * track_sep for i in range(1, num_tracks + 1)])
x_positions_curve = np.concatenate([-x_positions_curve,x_positions_curve], axis=-1)        # Simetría

# Crear máscara para los tracks en la región curva
mask_tracks_curve = (np.abs(X[..., None] - x_positions_curve) < track_width / 2).any(axis=-1)

# Aplicar la condición en la región curva
n_profile[mask_curve & mask_tracks_curve] = n1


# =============================================================================
# SECTORES RECTOS DESPUÉS DE LA APERTURA
# =============================================================================
mask_transition = (Z >= zona_apertura) & (Z< zmax)           # Máscara para la región del splitter
mask_transition2 = (Z >= 7000) & (Z < zmax) 

width_final = W0 / 2 + curve(zona_apertura)            # Cálculo del ancho variable para la transición


# Generar posiciones de los tracks en la transición
x_positions_final = width_final + np.array([i * track_sep for i in range(1, num_tracks + 1)])
x_positions_final_2 = width_final - W + np.array([i * track_sep for i in range(1, num_tracks + 1)])

x_positions_final = np.concatenate([-x_positions_final,x_positions_final], axis=-1)  # Simetría

x_positions_final_2 = np.concatenate([x_positions_final_2,-x_positions_final_2], axis=-1)  

# Crear máscaras booleanas
mask_tracks_final = (np.abs(X[..., None] - x_positions_final) < track_width / 2).any(axis=-1)
mask_tracks_final_2 = (np.abs(X[..., None] - x_positions_final_2) < track_width / 2).any(axis=-1)

n_profile[mask_transition & (mask_tracks_final)] = n1
n_profile[mask_transition2 & (mask_tracks_final_2)] = n1

# =============================================================================
#  SEGUNDA BIFURCACIÓN O SPLITTER
# =============================================================================

xf = 100                #Posición final de las guías de onda rectas del splitter
W_split = W/1.35        #Anchura de las guías de onda del splitter
offset2 = 400           #Offset dado para que los tracks interiores no solapen a los exteriores


pi = (W0 / 2 + curve(zona_apertura)-W,7000)      #Posición inicial donde comienza la guía recta final
pf = (33,6700)                                   #Posición final donde termina la guía recta final

def recta(z,pi,pf,sim = False):
    xf,zf = pf
    xi,zi = pi
    m = (xf - xi) / (zf - zi) 
    if sim == True:   
        return xi - m * (z - zi)
    elif sim == False:
        return xi + m * (z - zi)

mask_transition_splitter = (Z >= 6700) & (Z< 7000)   # Máscara para la región del splitter

width_final = recta(Z,pi,pf)             # Cálculo del ancho variable para la transición

x_positions_splitter = width_final[..., None] + np.array([i * track_sep for i in range(1, num_tracks + 1)])
x_positions_splitter_2 = width_final[..., None] + np.array([i * track_sep for i in range(1, num_tracks + 1)])

x_positions_splitter = np.concatenate([x_positions_splitter,-x_positions_splitter_2], axis=-1)  # Simetría


mask_tracks_splitter = (np.abs(X[..., None] - x_positions_splitter) < track_width / 2).any(axis=-1)

n_profile[mask_transition_splitter & (mask_tracks_splitter)] = n1 


pf = (W0 / 2 + curve(zona_apertura)-3.7*W,7000)      #Posición inicial donde comienza la guía recta final
pi = (33,6700)                                #Posición final donde termina la guía recta final


mask_transition_splitter = (Z >= 6700) & (Z< 7000)   # Máscara para la región del splitter


width_final = recta(Z,pi,pf)             # Cálculo del ancho variable para la transición

x_positions_splitter = width_final[..., None] + np.array([i * track_sep for i in range(1, num_tracks + 1)])
x_positions_splitter_2 = width_final[..., None] + np.array([i * track_sep for i in range(1, num_tracks + 1)])

x_positions_splitter = np.concatenate([x_positions_splitter,-x_positions_splitter_2], axis=-1)  # Simetría


mask_tracks_splitter = (np.abs(X[..., None] - x_positions_splitter) < track_width / 2).any(axis=-1)

n_profile[mask_transition_splitter & (mask_tracks_splitter)] = n1 

# Generar posiciones para la parte central (última región)
mask_transition2 = (Z >= 7000) & (Z< zmax)
center = 0
x_positions_center = center + np.array([W / 2 + i * track_sep for i in range(1, num_tracks + 1)])
x_positions_center = np.concatenate([-x_positions_center, x_positions_center], axis=-1)  # Simetría
mask_tracks_center = (np.abs(X[..., None] - x_positions_center) < track_width / 2).any(axis=-1)

# Aplicar la condición en la región de transición
n_profile[mask_transition2 & (mask_tracks_center)] = n1

'''
    ALGORITMO FFT-BPM
'''
# Precalcular dn2
dn2_profile = n_profile**2 - n0**2

# Paso inicial
E_z = np.fft.ifft(np.fft.fftshift(np.fft.fft(E_z)) * np.exp(-1j * kz * dz2))

I_z = np.zeros((len(z), N))  # Matriz de intensidades

for n in range(len(z)):
    dn2 = dn2_profile[n, :]
    lens_corrector_op = np.exp((-1j * k0 * dn2 * dz) / (2 * n0))  # Corrección de lente dinámica
    
    E_z = np.fft.ifft(np.fft.fft(lens_corrector_op * np.fft.ifft(
        np.exp(-1j * kz * dz2) * np.fft.fft(E_z))) * np.exp(-1j * kz * dz2)) #Actualización del campo por FFT
    I_z[n, :] = np.abs(E_z)**2

# Tiempo de ejecución
end_time = time.time()
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

'''
    GRÁFICOS
'''
epsilon = 0.001
log_I_z = np.log10(I_z / np.max(I_z) + epsilon)

plt.figure(figsize=(10, 6))
plt.imshow(log_I_z, extent=[-xmax, xmax, 0, zmax], aspect='auto', origin='lower', cmap='jet',interpolation='gaussian')
plt.xlabel('x $(\\mu m)$')
plt.ylabel('z $(\\mu m)$')
plt.colorbar(label='Log(Intensidad)')
plt.title('Propagación usando el método FFT BPM')
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(I_z, extent=[-xmax, xmax, 0, zmax], aspect='auto', origin='lower', cmap='jet',interpolation='gaussian')
plt.xlabel('x $(\\mu m)$')
plt.ylabel('z $(\\mu m)$')
plt.colorbar(label='Intensidad')
plt.title('Propagación usando el método FFT BPM')
plt.show()


plt.figure(figsize=(10, 6))
plt.imshow(n_profile, extent=[-xmax, xmax, 0, zmax], aspect='auto', origin='lower', cmap='coolwarm')
plt.colorbar(label='Índice de refracción n(x, z)')
plt.xlabel('x $(\\mu m)$')
plt.ylabel('z $(\\mu m)$')
plt.title('Perfil del índice de refracción n(x, z)')
plt.show()

plt.figure()
plt.plot(x,I_z[-1,:])
plt.grid()
plt.title("Intensidad a la salida")
plt.xlabel('x $(\\mu m)$')
plt.ylabel('Intensidad (u.arb)')
plt.xlim(xmin=-150,xmax=150)