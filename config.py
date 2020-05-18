# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 22:47:31 2019

@author: finkt
"""

from neuron import h, gui
#from matplotlib import pyplot
#import numpy as np

#### New ParallelContext object 
pc = h.ParallelContext()
pc.set_maxstep(10) #see https://www.neuron.yale.edu/neuron/static/new_doc/modelspec/programmatic/network/parcon.html#ParallelContext.set_maxstep, as well as section 2.4 of the Lytton/Salvador paper
idhost = int(pc.id())
nhost = int(pc.nhost())

#duration of simulation
duration=360000.0 #ms
t_seg=50.0 #(ms) simulation time between each data dump to the node 0

# set randomizer seed
randSeed = 1 #controls randomness in connectivity; this is different from Random123
h.Random().Random123_globalindex(randSeed) #this changes ALL Random123 streams

#this is '1' if you want to run through sleep states, according to Wake->N2->N3->REM->N2
#make this '0' if you want to just simulated one state of vigilance (in which case, select 
#that state by setting the appropriate parameters at the end of this file)
do_sleepstates = 1

#determine whether or not to record LFP
doextra = 1
area_cell=100 #determines cell density (micrometers^2 per cell) for when cells are placed in concentric rings (only need this if doextra==1)
XE=2000.0 #x coordinate of recording electrode (in micrometers); see setCellLocations method in Net class
YE=0.0 #y coordinate of recording electrode (in micrometers)
ZE=0.0 #z coordinate of recording electrode (in micrometers) 

if doextra==1:
    # the following code allows for Python to call a function at every time step,
    # which will allow us to compute both the summed cortical voltage and the cortical biophysical LFP at every time step. code taken from
    # https://www.neuron.yale.edu/phpBB/viewtopic.php?f=2&t=3389&p=14342&hilit=extracellular+recording+parallel#p14342
    v_rec=[]
    lfp_rec=[]
    def callback(cort_secs):
        v_cort = 0
        lfp_cort = 0
        for sec in cort_secs:
                for seg in sec:
                    v_cort = v_cort + seg.v #add up voltages in all segments of cortical cells
                    lfp_cort = lfp_cort + seg.er_xtra #add up biophysical LFP contributions in all segments of cortical cells
        v_rec.append(v_cort)
        lfp_rec.append(lfp_cort)

#set numbers of each cell type (see Krishnan's network.cfg); note that version 2 of this
#code assumes that Ninh=Nre=Ntc (in the method connectCells)
Npyr = 500
Ninh = 100
Nre = 100
Ntc = 100

thresh=0 # threshold for detecting spikes (for recording) and for initiating NetCon events (mV); see associatedGid and createNetcon and connect2Source in Cell class, and connectCells in Net class; this is equivalent to 'Prethresh' in Krishnan's currents.cpp

## synaptic connectivity parameters (see Krishnan's network.cfg)
## note that all strengths are the TOTAL synaptic strength impinging on each post-synaptic cell; the actual weight of any one synapse will be the total syn strength DIVIDED BY the number of presynaptic connections received by the particular neuron species in question
## note that all strenghts are by default prescribed in NEURON in units of microSiemens, while Krishnan's currents.cpp uses milliSiemens
## also, Krishnan's currents.cpp prescribes a delay of 0 for all synapses, but NEURON does not like that, so they have all been set to 0.1 ms
## there are many factors of "0.75" because I found that my NEURON model gives similar to results to Bazhenov 2002 C++ code when the synaptic strength
## is 0.75 that of the C++ code for pyramidal and inhibitory post-synatpic neurons (presumably because I used a full model for these cells, while
## Bazhenov et al used a reduced model)
re2tc_gaba_a_rad = 8
re2tc_gaba_a_str = 0.05 #uS
re2tc_gaba_a_del = 0.1 #ms 

re2tc_gaba_b_rad = 8
re2tc_gaba_b_str = 0.002 #uS
re2tc_gaba_b_del = 0.1 #ms

re2re_gaba_a_rad = 5
re2re_gaba_a_str = 0.1 #uS
re2re_gaba_a_del = 0.1 #ms 

tc2re_ampa_rad = 8
tc2re_ampa_str = 0.05 #uS
tc2re_ampa_del = 0.1 #ms

#Krishnan's currents.cpp says this is a D2 synapse (with depression and stochastic EPSP's), but it appears to in fact be just a normal AMPA synapse (withOUT depression or stochastic EPSP's)
tc2pyr_ampa_rad = 10
tc2pyr_ampa_str = 0.75*0.2 #uS
tc2pyr_ampa_del = 0.1 #ms

#Krishnan's currents.cpp says this is a D2 synapse (with depression and stochastic EPSP's), but it appears to in fact be just a normal AMPA synapse (withOUT depression or stochastic EPSP's)
tc2inh_ampa_rad = 2
tc2inh_ampa_str = 0.75*0.2 #uS
tc2inh_ampa_del = 0.1 #ms

## D2 synapses have both short-term depression and stochastic EPSP's
## NOTE: if pyr2pyr_ampa_d2_str is set to zero, then the program will force pyr2pyr2_ampa_d2_mini_str to zero as well (see createSynapses methods in cell_classes.py)
pyr2pyr_ampa_d2_rad = 5
pyr2pyr_ampa_d2_str = 0.75*0.24 #uS; 0.24 uS is the value from network.cfg, which is 10x larger than that quoted in the paper (this seems quite high, but it may be correct when you consider that this strength will be divided by the number of presynaptic connections)
pyr2pyr_ampa_d2_del = 0.1 #ms
pyr2pyr2_ampa_d2_mini_str = 0.75*0.33 #/ (2*pyr2pyr_ampa_d2_rad) #uS; strength of stochastic EPSP's; Krishnan's network.cfg prescribes 0.33 uS, but I think this should be normalized by the typical in-degree, just as the regular connections are. Doing so gives EPSP's of about 0.6 mV, which is close to prescription of 0.75 mV on p. 4 of Bazhenov 2002
pyr2pyr2_ampa_d2_mini_f = 20.0 #ms; this parameter is involved in calculating the stochastic EPSP times

#D1 synapses have short-term depression, but no stochastic EPSP's
pyr2pyr_nmda_d1_rad = 5
pyr2pyr_nmda_d1_str = 0.75*0.01 #uS
pyr2pyr_nmda_d1_del = 0.1 #ms
pyr2pyr_nmda_d1_Use = 0.0 #unitless factor (between 0 and 1) which determines the degree of short-term depression experienced with each presynaptic spike; Krishnan sets this to 0.0, which means there is no depression

## D2 synapses have both short-term depression and stochastic EPSP's
## NOTE: if pyr2inh_ampa_d2_str is set to zero, then the program will force pyr2inh_ampa_d2_mini_str to zero as well (see createSynapses methods in cell_classes.py)
pyr2inh_ampa_d2_rad = 1 
pyr2inh_ampa_d2_str = 0.75*0.12 #uS
pyr2inh_ampa_d2_del = 0.1 #ms
pyr2inh_ampa_d2_mini_str = 0.75*0.20 #/ ( Npyr*(2*pyr2inh_ampa_d2_rad+1)/Ninh ) #uS; strength of stochastic EPSP's; it would make sense to normalize by the typical in-degree of an INH cell, but I'm not sure whether this is what Krishnan actually did
pyr2inh_ampa_d2_mini_f = 20.0 #ms; this parameter is involved in calculating the stochastic EPSP times

#D1 synapses have short-term depression, but no stochastic EPSP's
pyr2inh_nmda_d1_rad = 1
pyr2inh_nmda_d1_str = 0.75*0.01 #uS
pyr2inh_nmda_d1_del = 0.1 #ms
pyr2inh_nmda_d1_Use = 0.0 #unitless factor (between 0 and 1) which determines the degree of short-term depression experienced with each presynaptic spike; Krishnan sets this to 0.0, which means there is no depression

pyr2tc_ampa_rad = 10
pyr2tc_ampa_str = 0.05 #uS
pyr2tc_ampa_del = 0.1 #ms

pyr2re_ampa_rad = 8
pyr2re_ampa_str = 0.15 #uS
pyr2re_ampa_del = 0.1 #ms

## D2 synapses have both short-term depression and stochastic IPSP's
## NOTE: if inh2pyr_gaba_a_d2_str is set to zero, then the program will force inh2pyr_gaba_a_d2_mini_str to zero as well (see createSynapses methods in cell_classes.py)
inh2pyr_gaba_a_d2_rad = 5
inh2pyr_gaba_a_d2_str = 0.75*0.24 #uS
inh2pyr_gaba_a_d2_del = 0.1 #ms
inh2pyr_gaba_a_d2_mini_str = 0.75*0.20 #/ (Ninh*(2*inh2pyr_gaba_a_d2_rad+1)/Npyr)  #uS; strength of stochastic IPSP's; Krishnan's network.cfg specifies 0.20 uS, but I think this should be normalized by the typical in-degree ("+1" included because each PYR cell receives a connection from the "central" INH cell, plus those 'rad' away from it on either side)
inh2pyr_gaba_a_d2_mini_f = 20.0 #ms; this parameter is involved in calculating the stochastic EPSP times

#### state of vigilance parameters (lines 555f. in Krishnan's main.cpp)
s2_scale=1.2;
s3_scale=1.8;
rem_scale=0.9;
  
gkl_pyr_baseline  = 0.000011 #S/cm2; Krishnan CellSyn.h line 362 says 0.011 mS/cm2 (Krishnan currents.h line 1201 says 0.005 mS/cm2, but I'm pretty sure CellSyn.h takes precedence)
gkl_inh_baseline  = 0.000009 #S/cm2 (Krishnan CellSyn.h line 525)
gkl_awake         = 0.19
gkl_s2            = gkl_awake*s2_scale
gkl_s3            = gkl_awake*s3_scale
gkl_rem           = gkl_awake*.9

gkl_TC_baseline   = 0.000024 # S/cm2; Krishnan CellSyn.h line 241
gkl_TC_awake      = 0.79
gkl_TC_s2         = gkl_TC_awake*s2_scale
gkl_TC_s3         = gkl_TC_awake*s3_scale
gkl_TC_rem        = gkl_TC_awake*.9

gkl_RE_baseline   = 0.000012 # S/cm2; Krishnan CellSyn.h line 177
gkl_RE_awake      = 0.9
gkl_RE_s2         = gkl_RE_awake*((2-s2_scale/2)-0.5)
gkl_RE_s3         = gkl_RE_awake*((2-s3_scale/2)-0.5)
gkl_RE_rem        = gkl_RE_awake*1.1

awake_AMPAd2      =0.2
s2_AMPAd2         =awake_AMPAd2*(s2_scale + (s2_scale-1)*0.2)
s3_AMPAd2         =awake_AMPAd2*(s3_scale + (s3_scale-1)*0.2)
rem_AMPAd2        =awake_AMPAd2*(rem_scale + (rem_scale-1)*0.2)

gh_TC_awake       =-8.0 #mV
gh_TC_s2          =-3.0
gh_TC_s3          =-2.0
gh_TC_rem         = 0.0

awake_GABAd2      =0.22
s2_GABAd2         =awake_GABAd2*1.15
s3_GABAd2         =awake_GABAd2*1.3
rem_GABAd2        =awake_GABAd2*0.7

awake_GABA_TC     =0.55
s2_GABA_TC        =awake_GABA_TC*1.15
s3_GABA_TC        =awake_GABA_TC*1.3
rem_GABA_TC       =awake_GABA_TC*0.7

awake_AMPA_TC     =0.5
s2_AMPA_TC        =0.5
s3_AMPA_TC        =0.5
rem_AMPA_TC       =0.5

if do_sleepstates == 1:
    #this is where you specify the initial state of vigilance; these values are used to instantiate the network in the 'connectCells' method
    init_AMPA_D2 = awake_AMPAd2
    init_AMPA_TC = awake_AMPA_TC
    init_GABA_D2 = awake_GABAd2
    init_GABA_TC = awake_GABA_TC
    init_gkl_RE  = gkl_RE_awake
    init_gkl_TC  = gkl_TC_awake
    init_gkl     = gkl_awake
    init_gh_TC   = gh_TC_awake
    
    #specify transition times between sleep states (in order to replicate Figs. 1 and 2 in Bazhenov 2016)
    #this assumes all the 'init' variables in the block above are set to the 'awake' state  
    awake_to_s2_start = 80000
    awake_to_s2_end = 97500
    s2_to_s3_start = 150000
    s2_to_s3_end = 167500
    s3_to_rem_start = 220000
    s3_to_rem_end = 237500
    rem_to_s2_start = 290000
    rem_to_s2_end = 307500
    
else: #if do_sleepstates != 1, then just simulate one sleep state
    #this is where you specify the initial state of vigilance; these values are used to instantiate the network in the 'connectCells' method
    #make sure all of these parameters are specified as being in the same state
    init_AMPA_D2 = s3_AMPAd2
    init_AMPA_TC = s3_AMPA_TC
    init_GABA_D2 = s3_GABAd2
    init_GABA_TC = s3_GABA_TC
    init_gkl_RE  = gkl_RE_s3
    init_gkl_TC  = gkl_TC_s3
    init_gkl     = gkl_s3
    init_gh_TC   = gh_TC_s3