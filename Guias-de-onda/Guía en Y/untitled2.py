# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:10:14 2025

@author: Alejandro
"""

import numpy as np
import matplotlib.pyplot as plt

'''
Pruebas potencia vs número de tracks en guía de ondas recta

wl = 0.633                       # Longitud de onda (micras)
n0 = 2.2864                      # Índice de refracción base
dn = 0.004                       # Cambio en el índice de refracción
w0 = 2.5                         # Ancho de la gaussiana inicial
W0 = 15                          # Ancho de la guía recta de entrada
track_sep = 2                    # Separación entre tracks

'''
p = [0.5659,0.7986,0.9066,0.9240,0.9281,0.9338]
n = [1,2,4,6,8,10]

plt.figure()
plt.plot(n,p,'k--')
plt.plot(n,p,'b.',markersize=12)
plt.xlabel('$N_{tracks}$',fontsize=20)
plt.ylabel('$P/P_0$',fontsize=20)
plt.grid()
plt.show()

'''
Pruebas potencia vs separacion entre tracks en guía de ondas recta

wl = 0.633                       # Longitud de onda (micras)
n0 = 2.2864                      # Índice de refracción base
dn = 0.004                       # Cambio en el índice de refracción
w0 = 2.5                         # Ancho de la gaussiana inicial
W0 = 15                          # Ancho de la guía recta de entrada
N_tracks = 4                     # Número de tracks

'''

p = [0.90659,0.83179,0.84327,0.79125,0.82448]
d = [2,2.5,3,3.5,4]

plt.figure()
plt.plot(d,p,'k--')
plt.plot(d,p,'b.',markersize=12)
plt.xlabel('Separación entre tracks ($\\mu m$)',fontsize=15)
plt.ylabel('$P/P_0$',fontsize=20)
plt.grid()
plt.show()


'''
Pruebas potencia vs tamaño del core entre tracks en guía de ondas recta

wl = 0.633                       # Longitud de onda (micras)
n0 = 2.2864                      # Índice de refracción base
dn = 0.004                       # Cambio en el índice de refracción
w0 = 2.5                         # Ancho de la gaussiana inicial
d_tracks = 2                     # distancia entre tracks
N_tracks = 4                     # Número de tracks

'''

p = [0.9691446158212045,0.8062305070671199,0.9653750962621406, 0.8524996656173491, 0.9073361428644506,0.8076115278573058,
 0.8737209194188258, 0.7449085754199527, 0.9388903770364728, 0.70589189055383,
 0.9402233567461148, 0.7506225669041265, 0.916827981016676, 0.7933733193303169,
 0.906598023609969, 0.7986502655391652, 0.9338159883136855, 0.7828693177077214,
 0.9362573730192467, 0.7650080749285671]
W0 = np.linspace(1,20,20)

plt.figure()
plt.plot(W0,p,'k--')
plt.plot(W0,p,'b.',markersize=12)
plt.xlabel('Ancho del core ($\\mu m$)',fontsize=15)
plt.ylabel('$P/P_0$',fontsize=20)
plt.grid()
plt.show()

'''
Pruebas potencia vs tamaño del core entre tracks en guía de ondas recta

wl = 0.633                       # Longitud de onda (micras)
n0 = 2.2864                      # Índice de refracción base
w0 = 2.5                         # Ancho de la gaussiana inicial
d_tracks = 2                     # distancia entre tracks
N_tracks = 4                     # Número de tracks
W0 = 1                           # Ancho de la guía recta de entrada
'''

p = [0.22570030158218388,
 0.6779504345399698,
 0.9009228690873713,
 0.9691446158212045,
 0.9883270783182967,
 0.99225100423196,
 0.9909527466152884,
 0.9877623894618803]
dn = np.arange(0.001,0.009,0.001)

plt.figure()
plt.plot(dn,p,'k--')
plt.plot(dn,p,'b.',markersize=12)
plt.xlabel('$\\Delta n$',fontsize=20)
plt.ylabel('$P/P_0$',fontsize=20)
plt.grid()
plt.show()