# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:54:39 2025

@author: Alejandro

633 nm
"""
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# ANCHURA guia recta 
# =============================================================================
p =[0.9062671573449882,
 0.6284135134534579,
 0.9438724994800586,
 0.7760678587302662,
 0.8913884709489431,
 0.7684879484440197,
 0.8212641899174047,
 0.7193313926603458,
 0.848575494437764,
 0.6652087498088661,
 0.9015091349297889,
 0.6415949164019118,
 0.8975312240981485,
 0.6771253821118491,
 0.8728147953334144,
 0.7158978659910696,
 0.8614369581648085,
 0.7286619803943499,
 0.8884856249181021,
 0.7220794266240244]

W0 = np.linspace(1,20,20)

import numpy as np
import matplotlib.pyplot as plt

# Suponiendo que tienes los datos de la simulación:
# ancho_core --> array con los anchos del core en micras (eje x)
# P_norm     --> array con los valores de P/P0 (eje y)
ancho_core = W0
P_norm = p
# 1. Aplicamos la FFT a la señal
fft_signal = np.fft.fft(P_norm)  # Transformada de Fourier de la señal
freqs = np.fft.fftfreq(len(P_norm), d=(ancho_core[1] - ancho_core[0]))  # Frecuencias espaciales

# 2. Tomamos el módulo (espectro de frecuencias)
fft_magnitude = np.abs(fft_signal)

# 3. Graficamos el resultado
plt.figure(figsize=(8,5))
plt.plot(freqs[:len(freqs)//2], fft_magnitude[:len(freqs)//2])  # Tomamos solo la parte positiva
plt.xlabel("Frecuencia espacial (1/µm)")
plt.ylabel("Amplitud de Fourier")
plt.title("Análisis de Fourier de las Oscilaciones")
plt.grid()
plt.show()


W0 = np.linspace(1,20,20)

plt.figure()
plt.plot(W0,p,'k--')
plt.plot(W0,p,'b.',markersize=12)
plt.xlabel('Ancho del core ($\\mu m$)',fontsize=15)
plt.ylabel('$P/P_0$',fontsize=20)
plt.xticks(np.arange(2,21,2))
plt.grid()
plt.show()
# =============================================================================
# NUMERO TRACKS 
# =============================================================================
p = [0.00265531324686375,
 0.4334922302765487,
 0.9438724994800586,
 0.9525013041834138,
 0.9527181999259565,
 0.9531575029412648]
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

p = [0.9438724994800586,
 0.8473977502560189,
 0.8745013810983602,
 0.7645414523066141,
 0.804102341396811]
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
p =[0.31172508132593435,
 0.7362454552518521,
 0.892207903459595,
 0.9438724994800586,
 0.9653334394814884,
 0.9762799560148191,
 0.9826699052749371,
 0.9867242187340616]

dn = np.arange(0.001,0.009,0.001)

plt.figure()
plt.plot(dn,p,'k--')
plt.plot(dn,p,'b.',markersize=12)
plt.xlabel('$\\Delta n$',fontsize=20)
plt.ylabel('$P/P_0$',fontsize=20)
plt.grid()
plt.show()