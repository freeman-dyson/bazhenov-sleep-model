'''
Generates spectrogram similar to that in Fig. 2 of 
'''
import numpy as np
from matplotlib import pyplot as plt
from morlet_def import morlet_wav

flo = 1
fhi = 100
deltaf = 0.1
freqvals=np.arange(flo,fhi+deltaf,deltaf)

dt=0.025 #ms
sigma = 1.0 #width of gaussian window (in seconds) for frequency-time analysis
cut_start=1000; #number of milliseconds to cut out of beginning
cut_end=1000; #number of milliseconds to cut out of end
dsample=100; #downsample by factor 'dsample'

temp=np.loadtxt('lfp_nhost=10.txt')
data=temp[0:len(temp):dsample] #downsample data
time=dsample*dt*np.arange(0,len(data))
srate = 1000/(dsample*dt) #Hz

Modulus, Phases, Transform = morlet_wav(data,srate,sigma,flo,fhi,deltaf)

plt.pcolormesh(time[round(cut_start/(dsample*dt)):len(time)-round(cut_end/(dsample*dt))], freqvals, Modulus[:,round(cut_start/(dsample*dt)):len(time)-round(cut_end/(dsample*dt))], rasterized='True', cmap='jet')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.colorbar()
#plt.clim((0,250))
plt.xlim([10000, 360000])

