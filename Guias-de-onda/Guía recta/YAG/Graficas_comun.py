# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:49:06 2025

@author: Alejandro

Simulaciones para 633,850,1500 y 3500 nm.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science', 'notebook'])
# =============================================================================
# P/P0 EN FUNCIÓN DE W0
# =============================================================================

p633 =[0.9062671573449882,
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


p850 =[0.3819629617413778,
 0.13411248446098936,
 0.7427554718630245,
 0.37208532256491395,
 0.7979730272053058,
 0.5209885012892285,
 0.7610201456163985,
 0.5690597345767071,
 0.7090664938103468,
 0.5719609445793362,
 0.6582865981740683,
 0.5503642827139791,
 0.6702180367405677,
 0.5218534455212634,
 0.7205152362460071,
 0.49600802007349637,
 0.7442030590603271,
 0.5161025277253254,
 0.7395822783786717,
 0.5011284714618062]

p1500 =[0.062440649922969986,
 0.055693798542646146,
 0.05061531852511311,
 0.0573183519920103,
 0.09854816133661781,
 0.06750377740364111,
 0.21075887288319659,
 0.09296531514731528,
 0.32390803593937845,
 0.1280192312026867,
 0.37913980915783946,
 0.1654278355709412,
 0.41971767510352276,
 0.21223229768705953,
 0.44246946188868563,
 0.26153218998516936,
 0.45453816270645675,
 0.29765157366532424,
 0.44756685040109584,
 0.3144729850236054]


p3700 =[0.1400498537305328,
 0.13413226448222626,
 0.1289924101131759,
 0.12939177806418728,
 0.1199970970393282,
 0.1258039002925965,
 0.114401245490149,
 0.1244767833545643,
 0.11354124888660759,
 0.1255846312484775,
 0.11630475180174167,
 0.1279678237442403,
 0.1203508942266113,
 0.13053397862250823,
 0.12550916708226034,
 0.13359140216047982,
 0.1328420682035621,
 0.13742923380860309,
 0.14064243801227658,
 0.1413873146285585]

W0 = np.linspace(1,20,20)

plt.figure()
plt.plot(W0,p633,'--',color='tab:red')
plt.plot(W0,p633,'r.',markersize=12,label='633')
plt.plot(W0,p850,'--',color='tab:blue')
plt.plot(W0,p850,'b.',markersize=12,label='850')
plt.plot(W0,p1500,'--',color='tab:green')
plt.plot(W0,p1500,'g.',markersize=12,label='1500')
plt.plot(W0,p3700,'--',color='orange')
plt.plot(W0,p3700,'.',color='orange',markersize=12,label='3700')
plt.xticks(np.arange(2,21,2))
plt.xlabel('Ancho del core ($\\mu m$)',fontsize=15)
plt.ylabel('$P/P_0$',fontsize=20)
plt.legend(loc='best',frameon=True,title='$\\bf{\\lambda \ (nm)}$',fontsize=12,title_fontsize=12)
plt.grid()
plt.show()

# =============================================================================
# P/P0  EN FUNCION DE NUMERO TRACKS 
# =============================================================================
p633 = [0.00265531324686375,
 0.4334922302765487,
 0.9438724994800586,
 0.9525013041834138,
 0.9527181999259565,
 0.9531575029412648]

p850 = [0.009040272355011779,
 0.1730144319117388,
 0.7979730272053058,
 0.8424900059532389,
 0.8435840885151421,
 0.8444323640773557]

p1500 = [0.09907566450302624,
 0.2031988391797946,
 0.45453816270645675,
 0.5099689210055266,
 0.531708411654314,
 0.5473130709166948]

p3700 = [0.1553131073364289,
 0.14929755816772403,
 0.1413873146285585,
 0.13231834582511046,
 0.142084441935112,
 0.1720023100908653]

n = [1,2,4,6,8,10]

plt.figure()
plt.plot(n,p633,'--',color='tab:red')
plt.plot(n,p633,'r.',markersize=12,label='633')
plt.plot(n,p850,'--',color='tab:blue')
plt.plot(n,p850,'b.',markersize=12,label='850')
plt.plot(n,p1500,'--',color='tab:green')
plt.plot(n,p1500,'g.',markersize=12,label='1500')
plt.plot(n,p3700,'--',color='orange')
plt.plot(n,p3700,'.',color='orange',markersize=12,label='3700')
plt.xlabel('$N_{tracks}$',fontsize=20)
plt.ylabel('$P/P_0$',fontsize=20)
plt.grid()
plt.legend(loc='best',frameon=True,title='$\\bf{\\lambda \ (nm)}$',fontsize=12,title_fontsize=12)
plt.xticks([2,4,6,8,10])
plt.show()

# =============================================================================
# P/P0  EN FUNCION DE DISTANCIA TRACKS 
# =============================================================================
p633 = [0.9438724994800586,
 0.8473977502560189,
 0.8745013810983602,
 0.7645414523066141,
 0.804102341396811]

p850 = [0.7979730272053058,
 0.6814214665110757,
 0.7418704147992021,
 0.6392801579300353,
 0.679627434494576]

p1500 = [0.5473130709166948,
 0.455886569557807,
 0.48416555452605764,
 0.4290078495340538,
 0.46147143885648545]

p3700 = [0.1720023100908653,
 0.23456653353835458,
 0.21441413105775492,
 0.26281863642388453,
 0.23673970870474487]
d = [2,2.5,3,3.5,4]

plt.figure()
plt.plot(d,p633,'--',color='tab:red')
plt.plot(d,p633,'r.',markersize=12,label='633')
plt.plot(d,p850,'--',color='tab:blue')
plt.plot(d,p850,'b.',markersize=12,label='850')
plt.plot(d,p1500,'--',color='tab:green')
plt.plot(d,p1500,'g.',markersize=12,label='1500')
plt.plot(d,p3700,'--',color='orange')
plt.plot(d,p3700,'.',color='orange',markersize=12,label='3700')
plt.xlabel('Separación entre tracks $(\\mu m)$',fontsize=15)
plt.ylabel('$P/P_0$',fontsize=20)
plt.grid()
plt.xticks(d)
plt.legend(loc='best',frameon=True,title='$\\bf{\\lambda \ (nm)}$',fontsize=12,title_fontsize=12)
plt.show()

# =============================================================================
# P/P0  EN FUNCION DE DN
# =============================================================================
p633 = [0.31172508132593435,
 0.7362454552518521,
 0.892207903459595,
 0.9438724994800586,
 0.9653334394814884,
 0.9762799560148191,
 0.9826699052749371,
 0.9867242187340616]

p850 = [0.13265500089071744,
 0.4579455148132581,
 0.6881350598465092,
 0.7979730272053058,
 0.8483511153345923,
 0.8770135727750286,
 0.8956075285978793,
 0.9064599939330881]

p1500 = [0.3635427914535391,
 0.45802122764131414,
 0.5003317340035229,
 0.5473130709166948,
 0.5962282234085737,
 0.6872500216555655,
 0.7705209261926886,
 0.8159575554758374]

p3700 = [0.18096143934424805,
 0.20028211906973256,
 0.22835269320220566,
 0.26281863642388453,
 0.29932727825640953,
 0.3334982208881832,
 0.36307885356253766,
 0.38775503913236387]

dn = np.arange(0.001,0.009,0.001)

plt.figure()
plt.plot(dn,p633,'--',color='tab:red')
plt.plot(dn,p633,'r.',markersize=12,label='633')
plt.plot(dn,p850,'--',color='tab:blue')
plt.plot(dn,p850,'b.',markersize=12,label='850')
plt.plot(dn,p1500,'--',color='tab:green')
plt.plot(dn,p1500,'g.',markersize=12,label='1500')
plt.plot(dn,p3700,'--',color='orange')
plt.plot(dn,p3700,'.',color='orange',markersize=12,label='3700')
plt.xlabel('$\\Delta n$',fontsize=20)
plt.ylabel('$P/P_0$',fontsize=20)
plt.legend(loc='best',frameon=True,title='$\\bf{\\lambda \ (nm)}$',fontsize=12,title_fontsize=12)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.gca().xaxis.get_offset_text().set_size(12)  # Ajustar tamaño de la notación científica
plt.gca().xaxis.get_offset_text().set_fontsize(12)  # Opcional: cambiar el tamaño de fuente
plt.grid()
plt.show()