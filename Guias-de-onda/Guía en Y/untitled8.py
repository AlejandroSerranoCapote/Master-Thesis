# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:05:18 2025

@author: Alejandro
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la malla
N = 1024  # Puntos en la malla x
L = 200   # Longitud de la caja
x = np.linspace(-L/2, L/2, N)

# Parámetros ópticos
wl = 1.064  # Longitud de onda en micras
n0 = 2.2  # Índice de refracción base
dn = 0.004
n1 = n0 - dn  # Índice más bajo

# Parámetros de la guía
W0 = 10   # Ancho de la guía de entrada
W = 5     # Ancho de las ramas
track_width = 0.5  # Ancho de cada track
track_sep = 1.0  # Separación entre tracks
num_tracks = 3  # Número de tracks en cada lado
zmax = 1000  # Longitud total

# Creación del perfil de índice de refracción
z = np.linspace(0, zmax, int(zmax/0.5))
n_profile = np.ones((len(z), len(x))) * n0

# Función para la curva de bifurcación
def curve(z_val):
    return 15 * np.tanh((z_val - 300) / 150)  # Curvatura suave

# Generar la guía óptica con tracks
for zi, zi_val in enumerate(z):
    for xi, xi_val in enumerate(x):
        if zi_val < 300:
            # Parte recta
            if abs(xi_val) < W0 / 2:
                n_profile[zi, xi] = n0
            for i in range(1, num_tracks + 1):
                if abs(xi_val - (W0 / 2 + i * track_sep)) < track_width / 2 or abs(xi_val + (W0 / 2 + i * track_sep)) < track_width / 2:
                    n_profile[zi, xi] = n1
        elif zi_val < 600:
            # Expansión y curvatura
            width = W0 / 2 + curve(zi_val)
            if abs(xi_val) < width:
                n_profile[zi, xi] = n0
            for i in range(1, num_tracks + 1):
                if abs(xi_val - (width + i * track_sep)) < track_width / 2 or abs(xi_val + (width + i * track_sep)) < track_width / 2:
                    n_profile[zi, xi] = n1
        # else:
        #     # División en tres ramas
        #     centers = [-26, 0, 26]  # Posiciones de las ramas
        #     for center in centers:
        #         if abs(xi_val - center) < W / 2:
        #             n_profile[zi, xi] = n0
        #         for i in range(1, num_tracks + 1):
        #             if abs(xi_val - (center + W / 2 + i * track_sep)) < track_width / 2 or abs(xi_val - (center - W / 2 - i * track_sep)) < track_width / 2:
        #                 n_profile[zi, xi] = n1

# Visualización
plt.figure(figsize=(10, 6))
plt.imshow(n_profile, extent=[-L/2, L/2, 0, zmax], aspect='auto', origin='lower', cmap='coolwarm')
plt.colorbar(label='Índice de refracción n(x, z)')
plt.xlabel('x $(\mu m)$')
plt.ylabel('z $(\mu m)$')
plt.title('Guía de onda en forma de Y con tracks')
plt.show()
