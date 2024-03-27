'''
Define baseline parameters for model. References to C++ code are to the code 
originally used in the paper "Cellular and neurochemical basis of sleep stages 
in the thalamocortical network" 
(Krishnan et. al., eLife, 2016, https://doi.org/10.7554/eLife.18607) 
 
The C++ code may be accessed here: 
https://github.com/bazhlab-ucsd/sleep-stage-transition/blob/main
'''

from neuron import h
#from neuron import gui # enable when not running on cluster

#### New ParallelContext object 
pc = h.ParallelContext()
# see https://www.neuron.yale.edu/neuron/static/new_doc/modelspec/programmatic/network/parcon.html#ParallelContext.set_maxstep
# as well as section 2.4 of "Simulation Neurotechnologies for Advancing Brain 
# Research: Parallelizing Large Networks in NEURON" (Lytton et. al, 2016)
pc.set_maxstep(10) 
idhost = int(pc.id())
nhost = int(pc.nhost())

# duration of simulation
duration=360000.0 #ms
t_seg=50.0 #(ms) simulation time between each data dump to node 0

# set randomizer seed
randSeed = 1 #global seed for random number generation
h.Random().Random123_globalindex(randSeed) #this changes ALL Random123 streams

# this is True if you want to run through sleep states, according to
# Wake->N2->N3->REM->N2 make this False if you want to just simulate one state
# of vigilance (in which case, select that state by setting the appropriate
# value for 'sleep_state')

# if do_sleepstates is set to True, then sleep_state will be ignored
do_sleepstates = True
sleep_state = 3  #0 for wake, 1 for S2, 2 for S3, and 3 for REM

# determine whether or not to record LFP
doextra = True

# determines cell density (micrometers^2 per cell) for when cells are placed in
# concentric rings (only need this if doextra==True)
area_cell=100
# x coordinate of recording electrode (in micrometers); see setCellLocations
# method in Net class
XE=2000.0
YE=0.0 #y coordinate of recording electrode (in micrometers)
ZE=0.0 #z coordinate of recording electrode (in micrometers) 

if doextra:
    # the following code allows for Python to call a function at every time
    # step, which will allow us to compute both the summed cortical voltage and
    # the cortical biophysical LFP at every time step. code taken from
    # https://www.neuron.yale.edu/phpBB/viewtopic.php?f=2&t=3389&p=14342&hilit=extracellular+recording+parallel#p14342
    v_rec=[]
    lfp_rec=[]
    def callback(cort_secs):
        v_cort = 0
        lfp_cort = 0
        for sec in cort_secs:
                for seg in sec:
                    # add up voltages in all segments of cortical cells
                    v_cort = v_cort + seg.v
                    # add up biophysical LFP contributions in all segments of cortical cells
                    lfp_cort = lfp_cort + seg.er_xtra
        v_rec.append(v_cort)
        lfp_rec.append(lfp_cort)

# set numbers of each cell type (see C++ code network.cfg); note 
# that the method 'connectCells' assumes Npyr=500, Ninh=100, Nre=100, and 
# Ntc=100

Npyr = 500
Ninh = 100
Nre = 100
Ntc = 100

# threshold for detecting spikes (for recording) and for initiating NetCon
# events (mV); see associatedGid and createNetcon and connect2Source in Cell
# class, and connectCells in Net class; this is equivalent to 'Prethresh' in
# C++ currents.cpp
thresh=0

# Synaptic connectivity parameters (see C++ network.cfg)
# note that all strengths are the TOTAL synaptic strength impinging on each
# post-synaptic cell; the actual weight of any one synapse will be the total
# syn strength DIVIDED BY the number of presynaptic connections received by the
# particular neuron species in question
#
# note that all strengths are by default prescribed in NEURON in units of
# microSiemens, while C++ currents.cpp uses milliSiemens. Also,
# currents.cpp prescribes a delay of 0 for all synapses, but this is difficult 
# to implement in NEURON, so they all delays have been set to 0.1 ms
#
# There are many factors of "0.75" because we found that the NEURON model gives
# similar to results to the C++ code when the synaptic strength is
# 0.75 that of the C++ code for pyramidal and inhibitory post-synatpic neurons
# (because we used a full model for these cells, while Bazhenov et. al.
# used a reduced model)

# These factors are equal to 2.0 - ach_level, with ach_level ranging 
# from 0.2 (for s3) to 1.1 (for REM) (It is equal to 1.0 for wake)

s2_scale=1.2  #from C++ main.cpp line 556
s3_scale = 1.8  #from C++ main.cpp line 557
rem_scale=0.9  #from C++ main.cpp line 558

# following factors apply to connections terminating in thalamus (RE->TC GABA-A, 
# RE->TC GABA-B, and RE->RE GABA-A connections, defined below)
# originally referred to as awake_GABA_TC, s2_GABA_TC, s3_GABA_TC, and 
# rem_GABA_TC in lines 597-601 of main.cpp of C++ code
awake_GABA_thal     =0.55
s2_GABA_thal        =awake_GABA_thal*1.15
s3_GABA_thal        =awake_GABA_thal*1.3
rem_GABA_thal       =awake_GABA_thal*0.7

re2tc_gaba_a_rad = 8 #C++ line 23 of network.cfg
re2tc_gaba_a_str = 0.05 #uS
re2tc_gaba_a_del = 0.1 #ms 

re2tc_gaba_b_rad = 8 #C++ line 24 of network.cfg
re2tc_gaba_b_str = 0.002 #uS
re2tc_gaba_b_del = 0.1 #ms

re2re_gaba_a_rad = 5 #C++ line 25 of network.cfg
re2re_gaba_a_str = 0.1 #uS
re2re_gaba_a_del = 0.1 #ms 

# following factors apply to AMPA connections terminating in the thalamus
# (PYR->TC AMPA, PYR->RE AMPA, and TC->RE AMPA connections, defined below)
# note the following parameters were originally named awake_AMPA_TC, 
# s2_AMPA_TC, s3_AMPA_TC, and rem_AMPA_TC in lines 605-608 of main.cpp
# of the original C++ code. They were all originally set to 0.5, but I here set
# them to 1.0, and then multiply the originals synaptic strength values by 0.5
awake_AMPA_thal     = 1.0
s2_AMPA_thal        = 1.0
s3_AMPA_thal        = 1.0
rem_AMPA_thal       = 1.0

pyr2tc_ampa_rad = 10 #C++ line 35 of network.cfg
pyr2tc_ampa_str = 0.5*0.05 #uS; original value of 0.05 is multiplied by 0.5, as noted above
pyr2tc_ampa_del = 0.1 #ms

pyr2re_ampa_rad = 8 #C++ line 36 of network.cfg
pyr2re_ampa_str = 0.5*0.15 #uS; original value of 0.05 is multiplied by 0.5, as noted above
pyr2re_ampa_del = 0.1 #ms

tc2re_ampa_rad = 8 #C++ line 26 of network.cfg
tc2re_ampa_str = 0.5*0.05 #uS; original value of 0.05 is multiplied by 0.5, as noted above
tc2re_ampa_del = 0.1 #ms

# following factors apply to all AMPA connections termining in cortex, other
# than PYR->PYR connections (so this includes TC->PYR, TC->INH, and PYR->INH
# connections, as defined below)
awake_AMPA_cort     =1.0 #this factor is named 'awake_AMPAd2' in C++ main.cpp line 580 and was set to 0.2; here we make this 1.0 instead of 0.2, to emphasize that this is the baseline value
s2_AMPA_cort        =awake_AMPA_cort*(s2_scale + (s2_scale-1)*0.2) # see C++ main.cpp line 582
s3_AMPA_cort        =awake_AMPA_cort*(s3_scale + (s3_scale-1)*0.2) # see C++ main.cpp line 583
rem_AMPA_cort       =awake_AMPA_cort*(rem_scale + (rem_scale-1)*0.2) # see C++ main.cpp line 584

# these are connected with "normal" AMPA synapses (using ampa.mod, not ampa_D2.mod)
tc2pyr_ampa_rad = 10 #C++ line 28 of network.cfg
tc2pyr_ampa_str = 0.75*0.2/5.0 #uS; divide by 5 because we increased awake_AMPA_cort from 0.2 ('awake_AMPAd2' in the C++ code) to 1.0
tc2pyr_ampa_del = 0.1 #ms

# these are connected with "normal" AMPA synapses (using ampa.mod, not ampa_D2.mod)
tc2inh_ampa_rad = 2 #C++ line 29 of network.cfg
tc2inh_ampa_str = 0.75*0.2/5.0 #uS, divide by 5 because we increased awake_AMPA_cort from 0.2 ('awake_AMPAd2' in the C++ code) to 1.0
tc2inh_ampa_del = 0.1 #ms

# These are AMPA_D2 synapses, which have both short-term depression and 
# stochastic EPSP's
# NOTE: if pyr2inh_ampa_d2_str is set to zero, then the program will force
# pyr2inh_ampa_d2_mini_str to zero as well (see createSynapses methods in
# cell_classes.py)
pyr2inh_ampa_d2_rad = 1  #C++ line 29 of network.cfg
pyr2inh_ampa_d2_str = 0.75*0.12/5.0 #uS, divide by 5 because we increased awake_AMPA_cort from 0.2 ('awake_AMPAd2' in the C++ code) to 1.0
pyr2inh_ampa_d2_del = 0.1 #ms
# strength of stochastic EPSP's
pyr2inh_ampa_d2_mini_str = 0.75*0.20/5.0  #uS; divide by 5 because we increased awake_AMPA_cort from 0.2 ('awake_AMPAd2' in the C++ code) to 1.0
# this corresponds to 'mini_fre' in the C++ code (see line 520 of currents.cpp)
# despite the name implying this is a frequency, it is really value of time. The
# largeer this value is, the less frequent the stochastic EPSP's (see function 
# 'gen_nextpsp' in ampa_D2.mod)
pyr2inh_ampa_d2_mini_f = 20.0 #ms; 

# scaling of PYR->PYR AMPA D2 synapses for each sleep stage. Note that this is 
# treated differently than in the C++ code, and these values were selected 
# in order to get obtain better spindles in N2
awake_AMPA_pyrpyr = 1.0
s2_AMPA_pyrpyr = 1.24
s3_AMPA_pyrpyr = 2.7048
rem_AMPA_pyrpyr = 1.056 

# These are AMPA_D2 synapses, which have both short-term depression and 
# stochastic EPSP's
# NOTE: if pyr2pyr_ampa_d2_str is set to zero, then the program will force
# pyr2pyr2_ampa_d2_mini_str to zero as well (see createSynapses methods in
# cell_classes.py)
pyr2pyr_ampa_d2_rad = 5 #C++ line 31 of network.cfg
pyr2pyr_ampa_d2_str = 0.03 #this value deviates from the C++ code in order to obtain better spindles in N2
pyr2pyr_ampa_d2_del = 0.1 #ms
pyr2pyr2_ampa_d2_mini_str = (0.33/0.24) * 0.03 #this value deviates from the C++ code in order to obtain better spindles in N2; the ratio 0.33/0.24 comes from the original scaling in the C++ code (see line 31 of network.cfg)
# note that ratio pyr2pyr2_ampa_d2_mini_str/pyr2pyr_ampa_d2_str determines PSP
# weight in AMPA_D2.mod, so that gmax associated with stochastic EPSP's changes 
# when gmax associated with pyr2pyr_ampa_d2_str changes (as it does in the 
# full sleep states simulation)
# this corresponds to 'mini_fre' in the C++ code (see line 520 of currents.cpp)
# despite the name implying this is a frequency, it is really value of time. The
# largeer this value is, the less frequent the stochastic EPSP's (see function 
# 'gen_nextpsp' in ampa_D2.mod)
pyr2pyr2_ampa_d2_mini_f = 20.0 #ms

# D1 synapses have short-term depression, but no stochastic EPSP's (these
# values do not vary with sleep stage)
pyr2pyr_nmda_d1_rad = 5 #C++ line 32 of network.cfg
pyr2pyr_nmda_d1_str = 0.75*0.01 #uS
pyr2pyr_nmda_d1_del = 0.1 #ms
# unitless factor (between 0 and 1) which determines the degree of short-term
# depression experienced with each presynaptic spike; C++ code sets this to
# 0.0, which means there is no depression
pyr2pyr_nmda_d1_Use = 0.0

# D1 synapses have short-term depression, but no stochastic EPSP's (these
# values do not vary with sleep stage)
pyr2inh_nmda_d1_rad = 1 #C++ line 34 of network.cfg
pyr2inh_nmda_d1_str = 0.75*0.01 #uS
pyr2inh_nmda_d1_del = 0.1 #ms
# unitless factor (between 0 and 1) which determines the degree of short-term
# depression experienced with each presynaptic spike; Krishnan sets this to
# 0.0, which means there is no depression
pyr2inh_nmda_d1_Use = 0.0

# following factors apply to INH->PYR GABA-A connections. See lines 591-595 
# of C++ main.cpp
awake_GABA_D2      =0.22
s2_GABA_D2         =awake_GABA_D2*1.15
s3_GABA_D2         =awake_GABA_D2*1.3
rem_GABA_D2        =awake_GABA_D2*0.7

# These are GABA_A_D2 synapses, which have both short-term depression and 
# stochastic EPSP's
# NOTE: if inh2pyr_gaba_a_d2_str is set to zero, then the program will force
# inh2pyr_gaba_a_d2_mini_str to zero as well (see createSynapses methods in
# cell_classes.py)
inh2pyr_gaba_a_d2_rad = 5 #C++ line 38 of network.cfg
inh2pyr_gaba_a_d2_str = 0.75*0.24 #uS
inh2pyr_gaba_a_d2_del = 0.1 #ms
# strength of stochastic IPSP's
inh2pyr_gaba_a_d2_mini_str = 0.75*0.20  #uS
# this corresponds to 'mini_fre' in the C++ code (see line 520 of currents.cpp)
# despite the name implying this is a frequency, it is really value of time. The
# largeer this value is, the less frequent the stochastic EPSP's (see function 
# 'gen_nextpsp' in gaba_A_D2.mod)
inh2pyr_gaba_a_d2_mini_f = 20.0 #ms; this parameter is involved in calculating the stochastic EPSP times 


# cellular properties that vary with sleep stage
gkl_pyr_awake         = 0.19 * 0.000011 #S/cm2; factor of 0.19 from C++ main.cpp line 561, and 0.000011 from C++ CellSyn.h line 362
gkl_pyr_s2            = gkl_pyr_awake*s2_scale
gkl_pyr_s3            = gkl_pyr_awake*s3_scale
gkl_pyr_rem           = gkl_pyr_awake*.9

gkl_inh_awake         = 0.19 * 0.000009 #S/cm2; factor of 0.19 from C++ main.cpp line 561, and 0.000011 from C++ CellSyn.h line 525
gkl_inh_s2            = gkl_inh_awake*s2_scale
gkl_inh_s3            = gkl_inh_awake*s3_scale
gkl_inh_rem           = gkl_inh_awake*.9

gkl_TC_awake      = 0.79 * 0.000024 # S/cm2; factor of 0.79 from C++ main.cpp line 567, and 0.000024 from C++ CellSyn.h line 241
gkl_TC_s2         = gkl_TC_awake*s2_scale
gkl_TC_s3         = gkl_TC_awake*s3_scale
gkl_TC_rem        = gkl_TC_awake*.9

gkl_RE_awake      = 0.9 * 0.000012 # S/cm2; factor of 0.9 from C++ main.cpp line 573, and 0.000012 from C++ CellSyn.h line 177
gkl_RE_s2         = gkl_RE_awake*((2-s2_scale/2)-0.5)
gkl_RE_s3         = gkl_RE_awake*((2-s3_scale/2)-0.5)
gkl_RE_rem        = gkl_RE_awake*1.1

gh_TC_awake       =-8.0 #mV; see lines 586-589 from C++ main.cpp
gh_TC_s2          =-4.0 #C++ code uses -3.0, but -4.0 was found to give better spindles in N2
gh_TC_s3          =-2.0
gh_TC_rem         = 0.0


if do_sleepstates:
    # this is where you specify the initial state of vigilance; these values
    # are used to instantiate the network in the 'connectCells' method
    init_GABA_thal = awake_GABA_thal
    init_AMPA_thal = awake_AMPA_thal
    init_AMPA_cort = awake_AMPA_cort
    init_AMPA_pyrpyr = awake_AMPA_pyrpyr
    init_GABA_D2 = awake_GABA_D2
    init_gkl_pyr = gkl_pyr_awake
    init_gkl_inh  = gkl_inh_awake
    init_gkl_RE  = gkl_RE_awake
    init_gkl_TC  = gkl_TC_awake
    init_gh_TC   = gh_TC_awake
    # specify transition times between sleep states (in order to replicate
    # Figs. 1 and 2 in Bazhenov 2016). this assumes all the 'init' variables in
    # the block above are set to the 'awake' state  
    awake_to_s2_start = 80000
    awake_to_s2_end = 97500
    s2_to_s3_start = 150000
    s2_to_s3_end = 167500
    s3_to_rem_start = 220000
    s3_to_rem_end = 237500
    rem_to_s2_start = 290000
    rem_to_s2_end = 307500
    
else: #if do_sleepstates != True, then just simulate one sleep state
    #these values are used to instantiate the network in the 'connectCells' method
    
    if sleep_state == 0:
        init_GABA_thal = awake_GABA_thal
        init_AMPA_thal = awake_AMPA_thal
        init_AMPA_cort = awake_AMPA_cort
        init_AMPA_pyrpyr = awake_AMPA_pyrpyr
        init_GABA_D2 = awake_GABA_D2
        init_gkl_pyr = gkl_pyr_awake
        init_gkl_inh = gkl_inh_awake
        init_gkl_RE  = gkl_RE_awake
        init_gkl_TC  = gkl_TC_awake
        init_gh_TC   = gh_TC_awake
    elif sleep_state == 1: 
        init_GABA_thal = s2_GABA_thal
        init_AMPA_thal = s2_AMPA_thal
        init_AMPA_cort = s2_AMPA_cort
        init_AMPA_pyrpyr = s2_AMPA_pyrpyr
        init_GABA_D2 = s2_GABA_D2  
        init_gkl_pyr = gkl_pyr_s2
        init_gkl_inh = gkl_inh_s2
        init_gkl_RE  = gkl_RE_s2
        init_gkl_TC  = gkl_TC_s2
        init_gh_TC   = gh_TC_s2
    elif sleep_state == 2: 
        init_GABA_thal = s3_GABA_thal
        init_AMPA_thal = s3_AMPA_thal
        init_AMPA_cort = s3_AMPA_cort
        init_AMPA_pyrpyr = s3_AMPA_pyrpyr
        init_GABA_D2 = s3_GABA_D2
        init_gkl_pyr = gkl_pyr_s3
        init_gkl_inh = gkl_inh_s3
        init_gkl_RE  = gkl_RE_s3
        init_gkl_TC  = gkl_TC_s3
        init_gh_TC   = gh_TC_s3
    elif sleep_state == 3:
        init_GABA_thal = rem_GABA_thal
        init_AMPA_thal = rem_AMPA_thal
        init_AMPA_cort = rem_AMPA_cort
        init_AMPA_pyrpyr = rem_AMPA_pyrpyr
        init_GABA_D2 = rem_GABA_D2
        init_gkl_pyr = gkl_pyr_rem
        init_gkl_inh = gkl_inh_rem
        init_gkl_RE  = gkl_RE_rem
        init_gkl_TC  = gkl_TC_rem
        init_gh_TC   = gh_TC_rem
