'''
Plot simulated LFP. Data file should either be lfp_nhost=*.txt or vcort_nhost=*.txt
'''

from matplotlib import pyplot as plt
import numpy as np

dt=0.025
lfp=np.loadtxt('lfp_nhost=1.txt')
time=dt*np.arange(0,len(lfp))

plt.plot(time,lfp)

