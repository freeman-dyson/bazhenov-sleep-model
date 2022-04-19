# bazh-model
THIS CODE WRITTEN BY CHRISTIAN G. FINK, 2019

**This is a revised version of fink_elife_recode_lfp. The previous version would in some cases hang when run in parallel. This version
fixes that issue.**

This simulation code is meant to qualitatively replicate the raster plot, LFP, and spectrogram from Fig. 2 
of Krishnan et al's "Cellular and neurochemical basis of sleep stages in the thalamocortical network" 
(eLife, 2016). It also adds the capability of computing the LFP from biophysical first principles, rather
than simply averaging cellular voltage traces.

This was programmed using an alpha version of NEURON 7.7.2 (using nrn-7.7.1-22-g49a9ca1b.w64-mingw-py-37-36-35-27-setup.exe from https://neuron.yale.edu/ftp/neuron/versions/alpha/ ). But it now works in the latest version of NEURON (see https://www.neuron.yale.edu/neuron/download)

After installing NEURON and downloading this code, go to the 'mod' directory and type 'nrnivmodl'. 
This will compile the mod files. Then move the output (nrnmech.dll in Windows,
x86_64 directory in Linux) up to the main directory.

Then run bazh_net_v6.py. 
To run in serial, use the command 'nrniv -python bazh_net.py'
To run in parallel, use 'mpiexec -n 2 nrniv -mpi -python bazh_net.py' 
(you can swap out the '2' for whatever number of processors you wish to use in parallel).

This will generate a raster txt file, a v_cort txt file containing the sum of the voltages in all
cortical cell compartments over time, and lfp txt file with the biophysical LFP trace.

To plot the raster plot, run plot_raster.py. To plot the LFP, run plot_lfp.py (you can change the file that's opened to choose between
the averaged voltage trace and the biophysical LFP0. And to generate the spectrogram, run analyze_time_freq.py. 
(you will need to make sure that the file names in all these routines match the name of the data file in question).

Note that the network starts out in a highly synchronous state, and it is recommended to throw out the first 10 seconds
of the simulation.

The default parameters will generate data similar to Fig. 2 of the eLife paper.
This simulation took approximately 14 hours to run in serial on a laptop, with significant speed-ups in parallel.

If you wish to simulate just a single state of vigilance, rather than running through all of them, then 
set the parameter 'do_sleepstates' in config.py to 0, and set the appropriate parameters at the end of config.py.
Most other high-level parameters can also be modified in config.py.
