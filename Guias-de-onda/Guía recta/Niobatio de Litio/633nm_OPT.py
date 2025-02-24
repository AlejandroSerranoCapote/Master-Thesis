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
p = [0.002129297791131253,
 0.2533066988678949,
 0.9691446158212045,
 0.9854887007888359,
 0.9859937527178092,
 0.9864276664072964]
n = [1,2,4,6,8,10]

plt.figure()
plt.plot(n,p,'k--')
plt.plot(n,p,'r.',markersize=12)
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

p = [0.9691446158212045,
 0.9090610163322319,
 0.9432388085842602,
 0.8409084869333187,
 0.9071129947717912]
d = [2,2.5,3,3.5,4]

plt.figure()
plt.plot(d,p,'k--')
plt.plot(d,p,'r.',markersize=15)
plt.xlabel('Separación entre tracks ($\\mu m$)',fontsize=15)
plt.ylabel('$P/P_0$',fontsize=20)
plt.xticks([2,2.5,3,3.5,4])
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


p2 = [0.9691446158212045,
 0.8062305070671199,
 0.9653750962621406,
 0.8524996656173491,
 0.9073361428644506,
 0.8076115278573058,
 0.8737209194188258,
 0.7449085754199527,
 0.9388903770364728,
 0.70589189055383,
 0.9402233567461148,
 0.7506225669041265,
 0.916827981016676,
 0.7933733193303169,
 0.906598023609969,
 0.7986502655391652,
 0.9338159883136855,
 0.7828693177077214,
 0.9362573730192467,
 0.7650080749285671]
W0 = np.linspace(1,20,20)

plt.figure()
plt.plot(W0,p2,'k--')
plt.plot(W0,p2,'r.',markersize=12)
plt.xlabel('Ancho del core ($\\mu m$)',fontsize=15)
plt.ylabel('$P/P_0$',fontsize=20)
plt.grid()
plt.xticks(np.arange(2,21,2))
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
plt.plot(dn,p,'r.',markersize=12)
plt.xlabel('$\\Delta n$',fontsize=20)
plt.ylabel('$P/P_0$',fontsize=20)
plt.grid()
plt.show()