# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:04:47 2025

@author: Alejandro

1.5 micras
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# ANCHURA guia recta (3)
# =============================================================================
p =[0.03406179457686392,
 0.03846282239361267,
 0.07294556794844508,
 0.046204749339439716,
 0.2112131483719787,
 0.0726591524856899,
 0.35798374676913797,
 0.1368563385039053,
 0.44476418413555197,
 0.19392274876782914,
 0.4863249945170969,
 0.20728027869029972,
 0.4763145453597929,
 0.2741611702135247,
 0.4610023060258572,
 0.34008428209125285,
 0.46683955617454714,
 0.3490566374883409,
 0.44966000079856056,
 0.3495196344883261]
W0 = np.linspace(1,20,20)

plt.figure()
plt.plot(W0,p,'k--')
plt.plot(W0,p,'b.',markersize=12)
plt.xlabel('Ancho del core ($\\mu m$)',fontsize=15)
plt.ylabel('$P/P_0$',fontsize=20)
plt.grid()
plt.show()
# =============================================================================
# NUMERO TRACKS (4)
# =============================================================================
p = [0.036858437083961325,
 0.1251789978852274,
 0.4863249945170969,
 0.5883323577933643,
 0.59614963504422,
 0.5939678094410393]

n = [1,2,4,6,8,10]

plt.figure()
plt.plot(n,p,'k--')
plt.plot(n,p,'b.',markersize=12)
plt.xlabel('$N_{tracks}$',fontsize=20)
plt.ylabel('$P/P_0$',fontsize=20)
plt.grid()
plt.show()

# =============================================================================
# Distancia entre tracks (2)
# =============================================================================

p =[0.5883323577933643,
 0.5012287994435136,
 0.49817968129736107,
 0.3710432357642936,
 0.4651609159260854]

d = [2,2.5,3,3.5,4]

plt.figure()
plt.plot(d,p,'k--')
plt.plot(d,p,'b.',markersize=12)
plt.xlabel('Separación entre tracks ($\\mu m$)',fontsize=15)
plt.ylabel('$P/P_0$',fontsize=20)
plt.xticks([2,2.5,3,3.5,4])
plt.grid()
plt.show()

# =============================================================================
# DELTA N
# =============================================================================
p = [0.23302064101773384,
 0.4150389346993188,
 0.5293687520683746,
 0.5883323577933643,
 0.6187616493651819,
 0.635714892596161,
 0.6465936534180164,
 0.6556959673948861]

dn = np.arange(0.001,0.009,0.001)

plt.figure()
plt.plot(dn,p,'k--')
plt.plot(dn,p,'b.',markersize=12)
plt.xlabel('$\\Delta n$',fontsize=20)
plt.ylabel('$P/P_0$',fontsize=20)
plt.grid()
plt.show()

