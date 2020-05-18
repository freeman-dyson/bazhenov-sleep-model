# -*- coding: utf-8 -*-
"""
This is a Python translation of morlet_wav.m, which I wrote many years ago.

this function performs time-frequency analysis on a signal using the
Morlet wavelet. I implemented this using the textbook 'The Illustrated
Wavelet Transform Handbook' (Paul Addison), pp. 33ff., and the paper
'Comparison of the Hilbert transform and wavelet methods...' (Le Van
Quyen, 2001). The parameter 'sigma' is simply the standard deviation of
the gaussian window (in seconds). The larger this is, the worse the
temporal resolution, but the better the frequency resolution. If you set sigma=1/(2*pi) seconds,
then you get a standard deviation in the power spectrum of the Morlet
wavelet of 1 Hz (but check p. 85 Le Van Quyen paper: they say that
2/3*(1/sigma) gives frequency resolution).
'srate' is the sampling rate of the signal (in Hz).'flo' is the lowest frequency
component we wish to analyze, 'fhi' the highest, and 'deltaf' is the step
size in frequency (all in Hz).
"""

import numpy as np

def morlet_wav(x, srate, sigma, flo, fhi, deltaf):
    N_orig = len(x)
    #zero-pad x so that the number of entries is a power of 2, so that the fft will be computationally efficient
    N=int( 2**(  np.ceil(  np.log(N_orig) / np.log(2)  )  )  )
    x=np.concatenate([x,np.zeros(N-len(x))])
    Xk=np.fft.fft(x)
    
    #figure out number of total frequency values at which you will be sampling
    #for the time-frequency analysis, and allocate space in 'Transform' (first
    #row of 'Transform' contains the power as a function of time for the lowest frequency
    freqvals=np.arange(flo,fhi+deltaf,deltaf)
    num_freqvals=len(freqvals)
    Transform=np.zeros((num_freqvals,N), dtype=complex)
    
    freq_samples=srate*np.arange(-N/2,N/2)/N #construct array of frequency values at which you sample the Fourier Transform of the wavelet function (Addison Eq. 2.38); don't need '-1' (as in Matlab code) bc. of how arange works; also, can assume N is divisible by 2 because of above
    
    for i_f, freq in enumerate(freqvals):
        #construct fourier transform of the Morlet wavelet in such a form that we
        #can use Eq. 2.35 (p. 33, Addison) along with iFFT to determine Transform
        #for specific frequency band. Note that my normalization is not the
        #same as in Addison's textbook.
        W = np.sqrt(2*np.pi)*sigma*np.exp(-2*np.pi**2*sigma**2*(freq_samples-freq)**2)
        Transform[i_f:i_f+1, :] = np.fft.ifft(Xk * np.fft.ifftshift(W))
        
    #throw away the part of Transform that corresponded to zero-padded portion of 'x'
    Transform=Transform[:,1:N_orig+1]
    #compute phases and modulus 
    Phases = np.arctan2(np.imag(Transform), np.real(Transform))
    Modulus = np.abs(Transform)
    
    return Modulus, Phases, Transform