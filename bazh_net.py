"""
Main simulation file for replicating 
"Cellular and neurochemical basis of sleep stages in the thalamocortical network," eLife, 2016.
"""

import config
from datetime import datetime
from neuron import h
import numpy as np
h('load_file("stdgui.hoc")') # need this instead of import gui to get the simulation to be reproducible and not give an LFP flatline
from network_class import Net

h('CVode[0].use_fast_imem(1)') #see use_fast_imem() at https://www.neuron.yale.edu/neuron/static/new_doc/simctrl/cvode.html  
  
def onerun(randSeed,Npyr,Ninh,Nre,Ntc):
       
    h.Random().Random123_globalindex(randSeed) #this changes ALL Random123 streams
    
    # create network
    net = Net(config.Npyr,config.Ninh,config.Nre,config.Ntc)
    
    #build list of sections in cortical cells, because only these are the ones that will generate the LFP
    cort_secs = []
    for pyr_gid in net.pyr_gidList:
        secs_to_add = config.pc.gid2cell(pyr_gid).all
        for sec in secs_to_add:
            cort_secs.append(sec)
        
    for inh_gid in net.inh_gidList:
        secs_to_add = config.pc.gid2cell(inh_gid).all
        for sec in secs_to_add:
            cort_secs.append(sec)
         
    if config.doextra:
        recording_callback = (config.callback, cort_secs)
        h.cvode.extra_scatter_gather(0,recording_callback)  #this tells NEURON to call 'callback' on every time step, in order to compute LFP
    
    '''if do_sleepstates==True, specify how parameters should change to induce different sleep states (see lines 640-777 of C++ main.cpp)'''
    if config.do_sleepstates:
        # see https://www.neuron.yale.edu/neuron/static/py_doc/programming/math/vector.html?highlight=vector#Vector.play
        t=h.Vector([config.awake_to_s2_start,config.awake_to_s2_end,config.s2_to_s3_start,config.s2_to_s3_end,config.s3_to_rem_start,config.s3_to_rem_end,config.rem_to_s2_start,config.rem_to_s2_end,config.rem_to_s2_end+h.dt]) #last entry (with "+h.dt") ensures that last parameter values remain constant to end of simulation ("If a constant outside the range is desired, make sure the last two points have the same y value and have different t values")
               
        #the next three entries are for changing GABA_A and GABA_B strengths in thalamus
        RE_TC_GABA_A_vec=h.Vector(config.re2tc_gaba_a_str * np.array([config.awake_GABA_thal,config.s2_GABA_thal,config.s2_GABA_thal,config.s3_GABA_thal,config.s3_GABA_thal,config.rem_GABA_thal,config.rem_GABA_thal,config.s2_GABA_thal,config.s2_GABA_thal]))
        RE_TC_A_veclist = []
        for tc_gid in net.tc_gidList:
            norm_vec = RE_TC_GABA_A_vec / config.pc.gid2cell(tc_gid).k_RE_TC_GABA_A
            RE_TC_A_veclist.append(norm_vec)
            RE_TC_A_veclist[-1].play(config.pc.gid2cell(tc_gid).synlist[0]._ref_gmax, t, True)
        
        RE_TC_GABA_B_vec=h.Vector(config.re2tc_gaba_b_str * np.array([config.awake_GABA_thal,config.s2_GABA_thal,config.s2_GABA_thal,config.s3_GABA_thal,config.s3_GABA_thal,config.rem_GABA_thal,config.rem_GABA_thal,config.s2_GABA_thal,config.s2_GABA_thal]))
        RE_TC_B_veclist = []
        for tc_gid in net.tc_gidList:
            norm_vec = RE_TC_GABA_B_vec / config.pc.gid2cell(tc_gid).k_RE_TC_GABA_B
            RE_TC_B_veclist.append(norm_vec)
            RE_TC_B_veclist[-1].play(config.pc.gid2cell(tc_gid).synlist[1]._ref_gmax, t, True)
        
        RE_RE_GABA_A_vec=h.Vector(config.re2re_gaba_a_str * np.array([config.awake_GABA_thal,config.s2_GABA_thal,config.s2_GABA_thal,config.s3_GABA_thal,config.s3_GABA_thal,config.rem_GABA_thal,config.rem_GABA_thal,config.s2_GABA_thal,config.s2_GABA_thal]))
        RE_RE_veclist = []
        for re_gid in net.re_gidList:
            norm_vec = RE_RE_GABA_A_vec / config.pc.gid2cell(re_gid).k_RE_RE
            RE_RE_veclist.append(norm_vec)
            RE_RE_veclist[-1].play(config.pc.gid2cell(re_gid).synlist[2]._ref_gmax, t, True)
            
        #next three entries address changing AMPA strengths for connections terminating in thalamus
        TC_RE_AMPA_vec=h.Vector(config.tc2re_ampa_str * np.array([config.awake_AMPA_thal,config.s2_AMPA_thal,config.s2_AMPA_thal,config.s3_AMPA_thal,config.s3_AMPA_thal,config.rem_AMPA_thal,config.rem_AMPA_thal,config.s2_AMPA_thal,config.s2_AMPA_thal]))
        TC_RE_veclist =[]
        for re_gid in net.re_gidList:
            norm_vec = TC_RE_AMPA_vec / config.pc.gid2cell(re_gid).k_TC_RE
            TC_RE_veclist.append(norm_vec)
            TC_RE_veclist[-1].play(config.pc.gid2cell(re_gid).synlist[0]._ref_gmax, t, True)
        
        PYR_TC_AMPA_vec=h.Vector(config.pyr2tc_ampa_str * np.array([config.awake_AMPA_thal,config.s2_AMPA_thal,config.s2_AMPA_thal,config.s3_AMPA_thal,config.s3_AMPA_thal,config.rem_AMPA_thal,config.rem_AMPA_thal,config.s2_AMPA_thal,config.s2_AMPA_thal]))
        PYR_TC_veclist = []
        for tc_gid in net.tc_gidList:
            norm_vec = PYR_TC_AMPA_vec / config.pc.gid2cell(tc_gid).k_PY_TC
            PYR_TC_veclist.append(norm_vec)
            PYR_TC_veclist[-1].play(config.pc.gid2cell(tc_gid).synlist[2]._ref_gmax, t, True)
        
        PYR_RE_AMPA_vec=h.Vector(config.pyr2re_ampa_str * np.array([config.awake_AMPA_thal,config.s2_AMPA_thal,config.s2_AMPA_thal,config.s3_AMPA_thal,config.s3_AMPA_thal,config.rem_AMPA_thal,config.rem_AMPA_thal,config.s2_AMPA_thal,config.s2_AMPA_thal]))
        PYR_RE_veclist = []
        for re_gid in net.re_gidList:
            norm_vec = PYR_RE_AMPA_vec / config.pc.gid2cell(re_gid).k_PY_RE
            PYR_RE_veclist.append(norm_vec)
            PYR_RE_veclist[-1].play(config.pc.gid2cell(re_gid).synlist[1]._ref_gmax, t, True)
            
        #next three entries address all AMPA connections termining in cortex, other than PYR->PYR connections
        TC_PYR_AMPA_D2_vec=h.Vector(config.tc2pyr_ampa_str * np.array([config.awake_AMPA_cort,config.s2_AMPA_cort,config.s2_AMPA_cort,config.s3_AMPA_cort,config.s3_AMPA_cort,config.rem_AMPA_cort,config.rem_AMPA_cort,config.s2_AMPA_cort,config.s2_AMPA_cort]))
        TC_PYR_veclist = [] #pretty sure I need to make a list of all the play vectors, bc. according to documentation, "The system maintains a set of play vectors and the vector will be removed from the list if the vector or var is destroyed." 
        for pyr_gid in net.pyr_gidList:
            norm_vec =  TC_PYR_AMPA_D2_vec / config.pc.gid2cell(pyr_gid).k_TC_PY  #need to noramlize each connection by in-degree of post-synaptic cell
            TC_PYR_veclist.append(norm_vec)
            TC_PYR_veclist[-1].play( config.pc.gid2cell(pyr_gid).synlist[0]._ref_gmax, t, True)
            
        TC_INH_AMPA_D2_vec=h.Vector(config.tc2inh_ampa_str * np.array([config.awake_AMPA_cort,config.s2_AMPA_cort,config.s2_AMPA_cort,config.s3_AMPA_cort,config.s3_AMPA_cort,config.rem_AMPA_cort,config.rem_AMPA_cort,config.s2_AMPA_cort,config.s2_AMPA_cort]))
        TC_INH_veclist = [] #pretty sure I need to make a list of all the play vectors, bc. according to documentation, "The system maintains a set of play vectors and the vector will be removed from the list if the vector or var is destroyed." 
        for inh_gid in net.inh_gidList:
            norm_vec = TC_INH_AMPA_D2_vec / config.pc.gid2cell(inh_gid).k_TC_IN
            TC_INH_veclist.append(norm_vec)
            TC_INH_veclist[-1].play(config.pc.gid2cell(inh_gid).synlist[0]._ref_gmax, t, True)
            
        PYR_INH_AMPA_D2_vec=h.Vector(config.pyr2inh_ampa_d2_str * np.array([config.awake_AMPA_cort,config.s2_AMPA_cort,config.s2_AMPA_cort,config.s3_AMPA_cort,config.s3_AMPA_cort,config.rem_AMPA_cort,config.rem_AMPA_cort,config.s2_AMPA_cort,config.s2_AMPA_cort]))
        PYR_INH_veclist = []
        for inh_gid in net.inh_gidList:
            norm_vec = PYR_INH_AMPA_D2_vec / config.pc.gid2cell(inh_gid).k_PY_IN_AMPA
            PYR_INH_veclist.append(norm_vec)
            PYR_INH_veclist[-1].play(config.pc.gid2cell(inh_gid).synlist[1]._ref_gmax, t, True)
        
        #next one addresses scaling of PYR->PYR connections only
        PYR_PYR_AMPA_D2_vec=h.Vector( config.pyr2pyr_ampa_d2_str * np.array([config.awake_AMPA_pyrpyr,config.s2_AMPA_pyrpyr,config.s2_AMPA_pyrpyr,config.s3_AMPA_pyrpyr,config.s3_AMPA_pyrpyr,config.rem_AMPA_pyrpyr,config.rem_AMPA_pyrpyr,config.s2_AMPA_pyrpyr,config.s2_AMPA_pyrpyr]))
        PYR_PYR_veclist = []
        for pyr_gid in net.pyr_gidList:
            norm_vec = PYR_PYR_AMPA_D2_vec / config.pc.gid2cell(pyr_gid).k_PY_PY_AMPA
            PYR_PYR_veclist.append(norm_vec)
            PYR_PYR_veclist[-1].play(config.pc.gid2cell(pyr_gid).synlist[1]._ref_gmax, t, True)
        
        #this last entry is for changing GABA_A_D2 strength for INH->PYR connections alone
        INH_PYR_GABA_D2_vec=h.Vector(config.inh2pyr_gaba_a_d2_str * np.array([config.awake_GABA_D2,config.s2_GABA_D2,config.s2_GABA_D2,config.s3_GABA_D2,config.s3_GABA_D2,config.rem_GABA_D2,config.rem_GABA_D2,config.s2_GABA_D2,config.s2_GABA_D2]))
        INH_PYR_veclist = []
        for pyr_gid in net.pyr_gidList:
            norm_vec = INH_PYR_GABA_D2_vec / config.pc.gid2cell(pyr_gid).k_IN_PY
            INH_PYR_veclist.append(norm_vec)
            INH_PYR_veclist[-1].play(config.pc.gid2cell(pyr_gid).synlist[3]._ref_gmax, t, True)
        
        #cellular properties
        gkl_pyr_vec=h.Vector( np.array([config.gkl_pyr_awake,config.gkl_pyr_s2,config.gkl_pyr_s2,config.gkl_pyr_s3,config.gkl_pyr_s3,config.gkl_pyr_rem,config.gkl_pyr_rem,config.gkl_pyr_s2,config.gkl_pyr_s2]))
        for pyr_gid in net.pyr_gidList:
            gkl_pyr_vec.play(config.pc.gid2cell(pyr_gid).dend(0.5)._ref_gkL_kL, t, True) #'True' makes it so values are linearly interpolated between points specified in vectors
            
        gkl_inh_vec=h.Vector( np.array([config.gkl_inh_awake,config.gkl_inh_s2,config.gkl_inh_s2,config.gkl_inh_s3,config.gkl_inh_s3,config.gkl_inh_rem,config.gkl_inh_rem,config.gkl_inh_s2,config.gkl_inh_s2]))
        for inh_gid in net.inh_gidList:
            gkl_inh_vec.play(config.pc.gid2cell(inh_gid).dend(0.5)._ref_gkL_kL, t, True)
        
        gkl_TC_vec=h.Vector( np.array([config.gkl_TC_awake,config.gkl_TC_s2,config.gkl_TC_s2,config.gkl_TC_s3,config.gkl_TC_s3,config.gkl_TC_rem,config.gkl_TC_rem,config.gkl_TC_s2,config.gkl_TC_s2]))
        for tc_gid in net.tc_gidList:
            gkl_TC_vec.play(config.pc.gid2cell(tc_gid).soma(0.5)._ref_gkL_kL, t, True)
        
        gkl_RE_vec=h.Vector( np.array([config.gkl_RE_awake,config.gkl_RE_s2,config.gkl_RE_s2,config.gkl_RE_s3,config.gkl_RE_s3,config.gkl_RE_rem,config.gkl_RE_rem,config.gkl_RE_s2,config.gkl_RE_s2]))
        for re_gid in net.re_gidList:
            gkl_RE_vec.play(config.pc.gid2cell(re_gid).soma(0.5)._ref_gkL_kL, t, True)
        
        gh_TC_vec=h.Vector([config.gh_TC_awake,config.gh_TC_s2,config.gh_TC_s2,config.gh_TC_s3,config.gh_TC_s3,config.gh_TC_rem,config.gh_TC_rem,config.gh_TC_s2,config.gh_TC_s2])
        for tc_gid in net.tc_gidList:
            gh_TC_vec.play(config.pc.gid2cell(tc_gid).soma(0.5)._ref_fac_gh_TC_iar, t, True)
    
    '''set up custom initialization to set RE voltage values'''
    def set_RE_voltages():
        #loop through all RE cells and set their initial voltages to -65 mV
        for re_gid in net.re_gidList:
            config.pc.gid2cell(re_gid).soma.v=-65 
        if(h.cvode.active()):
            h.cvode.re_init()
        else:
            h.fcurrent()
        h.frecord_init() 
       
    # run sim and gather spikes
    config.pc.set_maxstep(10) #see https://www.neuron.yale.edu/neuron/static/new_doc/modelspec/programmatic/network/parcon.html#ParallelContext.set_maxstep, as well as section 2.4 of the Lytton/Salvador paper
    h.dt = 0.025
    
    fih = h.FInitializeHandler(set_RE_voltages)
    h.finitialize(-68) #set initial voltages of all cells *except RE cells* to -68 mV
    #h.stdinit()
    
    if config.idhost==0: 
        print('Running sim...')
        startTime = datetime.now() # store sim start time
        
        raster_file = open("raster_nhost=%g.txt"%(config.nhost), 'w') # prepare file to print raster data to file
        lfp_file = open("lfp_nhost=%g.txt"%(config.nhost),'w') #prepare to print biophysical lfp data to file
        vcort_file = open("vcort_nhost=%g.txt"%(config.nhost), 'w') #prepare to print summed intracellular voltage trace data to file
    

    #actually run the simulation on all nodes
    t_curr = 0
    while (t_curr < config.duration-h.dt): # include the '-dt' to account for rounding error; otherwise, may get error in writeVoltages
        #step forward in periods of 'config.t_seg', and dump data to file after each step forward. Then resize vectors, so program does not run out of memory
        if(t_curr + config.t_seg < config.duration):
            config.pc.psolve(t_curr+config.t_seg)  
            if config.idhost==0: print("Numerically integrated through %.2f ms"%(t_curr+config.t_seg))
        else:
            config.pc.psolve(config.duration)
            
        net.gatherSpikes()  # gather spikes from all nodes onto master node
        if config.doextra: net.gatherLFP() #gather LFP data
        
        if(config.idhost==0):
            for i in range(len(net.tVecAll)): #print raster data to file
                raster_file.write("%.3f  %g\n" % (net.tVecAll[i], net.idVecAll[i])) # use the bash command 'sort -k 1n,1n -k 2n,2n raster_nhost=4 > raster_nhost=4_sorted' to sort the raster plots when nhost>1
            net.tVecAll = [] #reset the raster vectors that aggregate the results from all nodes, so they do not grow too large as the simulation progresses
            net.idVecAll = []
            
            if config.doextra:
                for i in range(len(net.v_sum)): #print cortical voltage data to file
                    vcort_file.write("%.3f \n" % net.v_sum[i])
                for i in range(len(net.lfp_sum)): #print LFP data to file
                    lfp_file.write("%.3f \n" % net.lfp_sum[i])
                    
        net.tVec.resize(0) #reset the raster vectors on each individual node, so they do not grow too large as the simulation progresses
        net.idVec.resize(0)
        if config.doextra:
            config.v_rec=[] #reset list to being empty
            config.lfp_rec=[] #reset list to being empty
            
        t_curr = t_curr + config.t_seg
    
    if config.idhost==0: 
        raster_file.close() #close raster file
        lfp_file.close()
        vcort_file.close()
        
        runTime = (datetime.now() - startTime).total_seconds()  # calculate run time
        print("Run time for %d sec sim = %.2f sec"%(int(config.duration/1000.0), runTime) )
        
    
    
    # plot net raster, save net data and plot cell 0 traces 
    '''net.gatherSpikes()  # gather spikes from all nodes onto master node
    if config.idhost==0: #if statement because we don't want every host to plot and save data (that would be redundant)
        #to plot raster data, we need to load the file, then define tVecAll and idVecAll lists so that they exist when the plotRaster method is called
        rasterdat=np.loadtxt("raster_nhost=%g.txt"%(config.nhost))
        if(len(rasterdat)>0):
            net.tVecAll=list(rasterdat[:,0])
            net.idVecAll=list(rasterdat[:,1])
            #net.plotRaster()
        else:
            print("No cells spiked, so there is no raster plot to display.")
        
        net.saveData()
        #net.cells[plot_cell].plotTraces()'''
        
    del net
    if config.doextra:
        h.cvode.extra_scatter_gather_remove(recording_callback) #removes 'callback', so that we don't have more and more callbacks on progressive iterations

onerun(config.randSeed,config.Npyr,config.Ninh,config.Nre,config.Ntc)

config.pc.barrier()
h.quit()
