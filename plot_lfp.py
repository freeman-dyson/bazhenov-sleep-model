# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 06:40:56 2019

@author: finkt
"""
from matplotlib import pyplot as plt
import numpy as np

dt=0.025
lfp=np.loadtxt('lfp_nhost=1.txt')
time=dt*np.arange(0,len(lfp))

plt.plot(time,lfp)

