# Sleep-stages

This simulation code is meant to qualitatively replicate the raster plot, LFP,
and spectrogram from Fig. 2 of Krishnan et al's "Cellular and neurochemical
basis of sleep stages in the thalamocortical network", eLife, 2016 (doi:
10.7554/eLife.18607). It also adds the capability of computing the LFP from
biophysical first principles, rather than simply averaging cellular voltage
traces. This ports the original C++ code (found here:
https://github.com/bazhlab-ucsd/sleep-stage-transition/blob/main, commit a77fc58)
to NEURON. The comments contain many references to this C++ code. References to 
"Bazhenov 2002" in the comments are to "Model of Thalamocortical Slow-Wave Sleep 
Oscillations and Transitions to Activated States," J. Neuro., 2002 
(doi: https://doi.org/10.1523/JNEUROSCI.22-19-08691.2002) The C++ code for that
model may be found here: https://modeldb.science/28189?tab=1

This code was developed in NEURON 8.2.2. Users should first install Python,
then install the latest version of NEURON (https://www.neuron.yale.edu/neuron/download).
To run this code in parallel, also install MPI.

To run this code, first go to the 'mod' directory and type 'nrnivmodl'. 
This will compile the mod files. Then move the output (nrnmech.dll in Windows,
x86\_64 directory in Linux) up to the main directory.

Then run bazh\_net.py. 
To run in serial, use the command 'nrniv -python bazh\_net.py'
To run in parallel, use 'mpiexec -n 2 nrniv -mpi -python bazh\_net.py' 
(you can swap out the '2' for whatever number of processors you wish to use
in parallel).

This will generate a raster txt file, a v\_cort txt file containing the sum of
the voltages in all cortical cell compartments over time, and lfp txt file with
the biophysical LFP trace.

To plot the raster plot, run plot\_raster.py. To plot the LFP, run plot\_lfp.py
(you can change the file that's opened to choose between the averaged voltage
trace and the biophysical LFP). And to generate the spectrogram, run
analyze\_time\_freq.py. (You will need to make sure that the file names in all
these routines match the name of the data file in question).

Note that the network starts out in a highly synchronous state, and it is
recommended to disregard the first 10 seconds of the simulation.

The default parameters will generate data similar to Fig. 2 of the eLife paper.
This simulation took approximately 14 hours to run in serial on a laptop, with
significant speed-ups in parallel.

If you wish to simulate just a single state of vigilance, rather than running
through all of them, then set the parameter 'do\_sleepstates' in config.py to
False, and set the appropriate parameters at the end of config.py.
Most other high-level parameters can also be modified in config.py.

To run on Neuroscience Gateway's scripting system (NSG-R), first zip code into a directory nsg_sub.zip. 

Submit the job by issuing:
curl -u [username]:$NSG_PSWD -H cipres-appkey:$NSG_KEY $NSG_URL/job/[username] -F tool=NEURON_EXPANSE -F input.infile_=@./nsg_sub.zip -F vparam.filename_=bazh_net.py -F vparam.runtime_=[run time, in hours] -F metadata.statusEmail=true

To check on status:
curl -u [username]:$NSG_PSWD -H cipres-appkey:$NSG_KEY (self URL)

To check on files in output folder:
curl -u [username]:$NSG_PSWD -H cipres-appkey:$NSG_KEY (results URL output by above command)
This will then output a download URL, which you then use to download the results

To download results:
curl -u [username]:$NSG_PSWD -H cipres-appkey:$NSG_KEY -O -J jobfile.downloadUri.url
*Note that each different output file has a different download URL, unless you just download the very last entry, output.tar.gz 
Then that will give you everything.
To extract it, first do 'mkdir output', then 'tar -xvf output.tar.gz -C ./output'

## Author

Fink, Christian G.

Contact: finkt@gonzaga.edu

## Version

1.0

## License

GPL-2
