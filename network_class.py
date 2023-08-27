'''
Defines network connectivity
'''

import config
import pickle
from neuron import h
import numpy as np
h('load_file("stdgui.hoc")') #for some reason, need this instead of import gui to get the simulation to be reproducible and not give an LFP flatline
from cell_classes import Cell, PyrCell, InhCell, RECell, TCCell   

class Net:
    """Creates network with prescribed number of each species of neurons (using parallelContext).
    Also ncludes methods to gather and plot spikes
    """
    def __init__(self, Npyr,Ninh,Nre,Ntc):

        self.Npyr=Npyr                  #number of pyramidal cells
        self.Ninh=Ninh                  #number of inhibitory cortical cells
        self.Nre=Nre
        self.Ntc=Ntc
        self.N = Npyr+Ninh+Nre+Ntc     # total number of cells in network
        self.cells = []                 # List of Cell objects in the net
        self.nclist = []                # List of NetCon in the net
        self.tVec = h.Vector()         # spike time of all cells on this processor
        self.idVec = h.Vector()        # cell ids of spike times
        #self.v_rec = []                #to record summed voltage values from all the cortical cells on particular host
        #self.lfp_rec = []               #to record LFP values from all the cortical cells on particular host
        self.createNet()  # Actually build the net
        
    def __del__(self):
        config.pc.gid_clear()
        
    def createNet(self):
        """Create, layout, and connect N cells."""
        self.setGids() #### set global ids (gids), used to connect cells
        self.createCells()
        self.connectCells() 
        self.createStims() 
        self.createIClamps()
        if config.doextra: self.setCellLocations() #only need to do this if recording the LFP
        
    def setGids(self):
        self.gidList = []
        self.pyr_gidList = []
        self.inh_gidList = []
        self.re_gidList = []
        self.tc_gidList = []
        #### Round-robin counting. Each host as an id from 0 to nhost - 1.
        for i in range(config.idhost, self.N, config.nhost): #start with idhost, count by nhost until you get to total number of neurons (so if idhost=2 and nhost=4, the list contains [2,6,10...])
            self.gidList.append(i)
            if i<self.Npyr:
                self.pyr_gidList.append(i)
            elif(self.Npyr <= i < self.Npyr+self.Ninh):
                self.inh_gidList.append(i)
            elif(self.Npyr+self.Ninh <= i < self.Npyr+self.Ninh+self.Nre):
                self.re_gidList.append(i)
            else:
                self.tc_gidList.append(i)
                
    def createCells(self):
        """Create and layout cells (in this host) in the network."""
        self.cells = []
        #random.seed(config.randSeed) #use Python's random number generator for randomizing g_pas values for PYR and INH cells (use Python instead of Random123 to make sure these streams don't somehow get crossed)
        
        for gid in self.gidList: #### Loop over cells in this node/host
            if gid<self.Npyr:
                cell = PyrCell(gid) # create pyramidal cell if gid is less than Npyr
                r = h.Random()
                r.Random123(gid,2,0) #set stream of random numbers; first argument is gid, make second argument 2 because 0 and 1 are already taken by AMPA_D2 and GABA_D2 synapses (see cell_classes.py)
                cell.dend.g_pas = 0.000011 + (r.uniform(0,1)+1) * 0.000003 #add cell-to-cell variability in this parameter, as prescribed in Krishnan CellSyn.h line 365
            elif(self.Npyr <= gid < self.Npyr+self.Ninh):
                cell = InhCell(gid)  # create pyramidal cell if gid is Npyr or greater
                r = h.Random()
                r.Random123(gid,1,0) #set stream of random numbers; first argument is gid, make second argument 1 because 0 is already taken by AMPA_D2 synapse (see cell_classes.py)
                cell.dend.g_pas = 0.000009 + (r.uniform(0,1)+1) * 0.000003 #add cell-to-cell variability in this parameter, as prescribed in Krishnan CellSyn.h line 526
            elif(self.Npyr+self.Ninh <= gid < self.Npyr+self.Ninh+self.Nre):
                cell = RECell(gid)
            else:
                cell = TCCell(gid)
                
            self.cells.append(cell)  # add cell object to net cell list
            cell.associateGid() # associated gid to each cell; if this line generates the error "gid=0 already exists on this process as an output port," then you just need to restart the kernel in Spyder
            config.pc.spike_record(cell.gid, self.tVec, self.idVec) # Record spikes of this cell
            
            print('Created cell %d on host %d out of %d'%(gid, config.idhost, config.nhost) )
            print('config.pc.gid2cell(%d): %s'%(gid,config.pc.gid2cell(gid)))
            
    def connectCells(self):
        """Connect cells. Assume that a "radius of 8" (as specified in Krishnan 2016) implies
        fan-out. Note that this assumes that there Npyr=500, Ninh=100, Nre=100, and Ntc=100.
        It will not work correctly for other values."""
        
        ##### specify RE->TC GABA_A connections (NOTE: It is assumed Nre=Ntc) 
        rad=config.re2tc_gaba_a_rad #number of outgoing connections from each RE cell
        if (2*rad+1) > self.Ntc : print("******WARNING: RE->TC GABA_A connectivity radius is too large.")
        for tc_gid in self.tc_gidList: #loop over all post-synaptic TC cells on this machine (have to loop over post-syn cells bc. pc.gid_connect only works with post-synaptic gid on machine)
            i_tc = tc_gid-(self.Npyr+self.Ninh+self.Nre) #easier to work by indexing the TC cells from 0 to Ntc-1, rather than from Npyr+Ninh+Nre to Npyr+Ninh+Nre+Ntc-1
            tmp = list(range(i_tc-rad,i_tc+rad+1)) #generate list of RE sources (but this will in general include negative values); note that this assumes Nre=Ntc
            re_set=[val % self.Nre for val in tmp] #apply modulus operation to eliminate negative numbers; this implements periodic boundary conditions; indexes RE cells from 0 to Nre-1
            config.pc.gid2cell(tc_gid).k_RE_TC_GABA_A = 0 #make sure in-degree is initialized to zero
            #connect each RE cell to each target TC cell
            for i_re in re_set:
                #must add Npyr+Ninh to i_re in order to get RE cells gid
                nc = config.pc.gid_connect(self.Npyr+self.Ninh+i_re, config.pc.gid2cell(tc_gid).synlist[0]) # create NetCon by associating pre gid to post synapse; synlist[0] is GABA_A synapse for TC cells
                nc.weight[0] = 1 #weight should always be 1; adjust strength by modifying synapse's gmax
                nc.delay = config.re2tc_gaba_a_del
                nc.threshold = config.thresh
                self.nclist.append((self.Npyr+self.Ninh+i_re,tc_gid,nc))
                config.pc.gid2cell(tc_gid).nclist.append((self.Npyr+self.Ninh+i_re,nc))
                config.pc.gid2cell(tc_gid).k_RE_TC_GABA_A += 1 #update in-degree for this cell
            #reduce gmax for this cell's synapse, so that the total synaptic strength is equal to that specified in the config file
            #(gmax was already set in cell_classes.py, which used the value specified in config.py)
            #for fac_GABA_TC, see currents.cpp lines 341 & 369, and main.cpp
            config.pc.gid2cell(tc_gid).synlist[0].gmax = config.init_GABA_thal * config.re2tc_gaba_a_str / config.pc.gid2cell(tc_gid).k_RE_TC_GABA_A
    
        ##### specify RE->TC GABA_B connections (NOTE: It is assumed Nre=Ntc) 
        rad=config.re2tc_gaba_b_rad #number of outgoing connections from each RE cell
        if (2*rad+1) > self.Ntc : print("******WARNING: RE->TC GABA_B connectivity radius is too large.")
        for tc_gid in self.tc_gidList: #loop over all post-synaptic TC cells on this machine (have to loop over post-syn cells bc. pc.gid_connect only works with post-synaptic gid on machine)
            i_tc = tc_gid-(self.Npyr+self.Ninh+self.Nre) #easier to work by indexing the TC cells from 0 to Ntc-1, rather than from Npyr+Ninh+Nre to Npyr+Ninh+Nre+Ntc-1
            tmp = list(range(i_tc-rad,i_tc+rad+1)) #generate list of RE sources (but this will in general include negative values); note that this assumes Nre=Ntc
            re_set=[val % self.Nre for val in tmp] #apply modulus operation to eliminate negative numbers; this implements periodic boundary conditions; indexes RE cells from 0 to Nre-1
            config.pc.gid2cell(tc_gid).k_RE_TC_GABA_B = 0 #make sure in-degree is initialized to zero
            for i_re in re_set:  #must add Npyr+Ninh to i_re in order to get RE cell's gid
                nc = config.pc.gid_connect(self.Npyr+self.Ninh+i_re, config.pc.gid2cell(tc_gid).synlist[1]) # create NetCon by associating pre gid to post synapse; synlist[1] is GABA_B synapse for TC cells
                nc.weight[0] = 1 #weight should always be 1; adjust strength by modifying synapse's gmax
                nc.delay = config.re2tc_gaba_b_del
                nc.threshold=config.thresh
                self.nclist.append((self.Npyr+self.Ninh+i_re,tc_gid,nc))
                config.pc.gid2cell(tc_gid).nclist.append((self.Npyr+self.Ninh+i_re,nc))
                config.pc.gid2cell(tc_gid).k_RE_TC_GABA_B += 1 #update in-degree for this cell
            #see note above for GABA_synapse
            #for fac_GABA_TC, see currents.cpp lines 341 & 369, and main.cpp
            config.pc.gid2cell(tc_gid).synlist[1].gmax = config.init_GABA_thal * config.re2tc_gaba_b_str / config.pc.gid2cell(tc_gid).k_RE_TC_GABA_B
            
        ##### specify RE->RE GABA_A connections
        rad=config.re2re_gaba_a_rad #number of outgoing connections from each RE cell
        if (2*rad+1) > self.Nre : print("******WARNING: RE->RE GABA_A connectivity radius is too large.")
        for re_gid in self.re_gidList: #loop over all post-synaptic RE cells on this machine (have to loop over post-syn cells bc. pc.gid_connect only works with post-synaptic gid on machine)
            i_post_re = re_gid - (self.Npyr+self.Ninh) #easier to work by indexing the RE cells from 0 to Nre-1, rather than from Npyr+Ninh to Npyr+Ninh+Nre-1
            tmp = list(range(i_post_re-rad,i_post_re+rad+1)) #generate list of RE sources (but this will in general include negative values)
            re_pre_set=[val % self.Nre for val in tmp] #apply modulus operation to eliminate negative numbers; this implements periodic boundary conditions; indexes RE cells from 0 to Nre-1
            config.pc.gid2cell(re_gid).k_RE_RE = 0 #make sure in-degree is initialized to zero
            for i_pre_re in re_pre_set:
                if i_pre_re != i_post_re: #prevent self-connection
                    nc = config.pc.gid_connect(self.Npyr+self.Ninh+i_pre_re, config.pc.gid2cell(re_gid).synlist[2]) # create NetCon by associating pre gid to post synapse; synlist[2] is GABA_A synapse for RE cells
                    nc.weight[0] = 1 #weight should always be 1; adjust strength by modifying synapse's gmax
                    nc.delay = config.re2re_gaba_a_del
                    nc.threshold = config.thresh
                    self.nclist.append((self.Npyr+self.Ninh+i_pre_re,re_gid,nc))
                    config.pc.gid2cell(re_gid).nclist.append((self.Npyr+self.Ninh+i_pre_re,nc))
                    config.pc.gid2cell(re_gid).k_RE_RE += 1 #update in-degree for this cell
            #reduce gmax for this cell's synapse, so that the total synaptic strength is equal to that specified in the config file
            #(gmax was already set in cell_classes.py, which used the value specified in config.py)
            #for fac_GABA_TC, see currents.cpp lines 341 & 369, and main.cpp
            config.pc.gid2cell(re_gid).synlist[2].gmax = config.init_GABA_thal * config.re2re_gaba_a_str / config.pc.gid2cell(re_gid).k_RE_RE
              
        ##### specify TC->RE AMPA connections (NOTE: It is assumed Nre=Ntc)  
        rad = config.tc2re_ampa_rad    #number of outgoing connections from each TC cell
        if (2*rad+1) > self.Nre : print("******WARNING: TC->RE AMPA connectivity radius is too large.")
        for re_gid in self.re_gidList: #loop over all post-synaptic RE cells on this machine (have to loop over post-syn cells bc. pc.gid_connect only works with post-synaptic gid on machine)
            i_re = re_gid - (self.Npyr+self.Ninh) #easier to work by indexing the RE cells from 0 to Nre-1, rather than from Npyr+Ninh to Npyr+Ninh+Nre-1    
            tmp = list(range(i_re-rad,i_re+rad+1)) #generate list of TC sources (but this will in general include negative values)
            tc_set=[val % self.Ntc for val in tmp] #apply modulus operation to eliminate negative numbers; this implements periodic boundary conditions; indexes TC cells from 0 to Ntc-1
            config.pc.gid2cell(re_gid).k_TC_RE = 0 #make sure in-degree is initialized to zero
            for i_tc in tc_set: 
                nc = config.pc.gid_connect(self.Npyr+self.Ninh+self.Nre+i_tc, config.pc.gid2cell(re_gid).synlist[0]) # create NetCon by associating pre gid to post synapse; synlist[0] is AMPA synapse for RE cells
                nc.weight[0] = 1 #weight should always be 1; adjust strength by modifying synapse's gmax
                nc.delay = config.tc2re_ampa_del
                nc.threshold = config.thresh
                self.nclist.append((self.Npyr+self.Ninh+self.Nre+i_tc, re_gid, nc))
                config.pc.gid2cell(re_gid).nclist.append((self.Npyr+self.Ninh+self.Nre+i_tc, nc))
                config.pc.gid2cell(re_gid).k_TC_RE += 1
            #reduce gmax for this cell's synapse, so that the total synaptic strength is equal to that specified in the config file
            #(gmax was already set in cell_classes.py, which used the value specified in config.py)
            #for fac_AMPA_TC, see Krishnan's currents.cpp line 429, and main.cpp 
            config.pc.gid2cell(re_gid).synlist[0].gmax = config.init_AMPA_thal * config.tc2re_ampa_str / config.pc.gid2cell(re_gid).k_TC_RE
        
        ##### specify TC->PYR AMPA connections
        rad=config.tc2pyr_ampa_rad #number of outgoing connections from each TC cell
        p2t_ratio = self.Npyr/self.Ntc #we will assume that Npyr > Ntc; this ratio will be important for determining the center of the set of source TC cells for each post-synaptic PYR cell
        if p2t_ratio <= 1: print("******WARNING: This code assumes that Npyr > Ntc")
        if (2*rad+1) > self.Npyr : print("******WARNING: TC->PYR AMPA connectivity radius is too large.")
        
        for pyr_gid in self.pyr_gidList: #loop over all post-synaptic PYR cells on this machine (have to loop over post-syn cells bc. pc.gid_connect only works with post-synaptic gid on machine)
            i_pyr = pyr_gid #no subtraction needed, because PYR cells are the first set of gid's
            tmp = list(range(  int(np.floor((i_pyr-rad)/p2t_ratio)), int(np.floor((i_pyr+rad)/p2t_ratio))+1))  #center_tc should be roughly round(i_pyr/p2t_ratio); then you need to consider the radius of connectivity surrounding that, and consider that larger p2t_ratio decreases the number of TC cells sending to a given PYR cell (for a particular radius of connectivity); "+1" is due to definition of Python's "range" function 
            tc_set = [val % self.Ntc for val in tmp] #apply modulus operation to eliminate negative numbers; this implements periodic boundary conditions; indexes TC cells from 0 to Ntc-1
            config.pc.gid2cell(pyr_gid).k_TC_PY = 0 #make sure in-degree is initialized to zero
            for i_tc in tc_set:
                nc = config.pc.gid_connect(self.Npyr+self.Ninh+self.Nre+i_tc, config.pc.gid2cell(pyr_gid).synlist[0]) # create NetCon by associating pre gid to post synapse; synlist[0] is AMPA synapse for PYR cells
                nc.weight[0] = 1 #weight should always be 1; adjust strength by modifying synapse's gmax
                nc.delay = config.tc2pyr_ampa_del
                nc.threshold = config.thresh
                self.nclist.append((self.Npyr+self.Ninh+self.Nre+i_tc,pyr_gid,nc))
                config.pc.gid2cell(pyr_gid).nclist.append((self.Npyr+self.Ninh+self.Nre+i_tc,nc))
                config.pc.gid2cell(pyr_gid).k_TC_PY += 1
            #reduce gmax for this cell's synapse, so that the total synaptic strength is equal to that specified in the config file
            #(gmax was already set in cell_classes.py, which used the value specified in config.py)
            #for fac_AMPA_TC, see Krishnan's currents.cpp line 429, and main.cpp; note that network.cfg lists this as a D2 synapse with mini_f=0, which
            #doesn't make sense; but if it was somehow coded as a D2 synapse, then we should instead apply fac_AMPA_D2;
            #so it is unclear whether we should apply fac_AMPA_TC or fac_AMPA_D2 here (I'm guessing D2)
            config.pc.gid2cell(pyr_gid).synlist[0].gmax = config.init_AMPA_cort * config.tc2pyr_ampa_str / config.pc.gid2cell(pyr_gid).k_TC_PY
        
        ##### specify TC->INH AMPA connections (NOTE: this assumes that Ntc=Ninh)
        rad=config.tc2inh_ampa_rad #number of outgoing connections from each TC cell
        if self.Ntc != self.Ninh: print("******WARNING: This routine assumes that Ntc=Ninh")
        if (2*rad+1) > self.Nre : print("******WARNING: TC->RE AMPA connectivity radius is too large.")
        
        for inh_gid in self.inh_gidList: #loop over all post-synaptic INH cells on this machine (have to loop over post-syn cells bc. pc.gid_connect only works with post-synaptic gid on machine)
            i_inh = inh_gid-self.Npyr #easier to work by indexing the INH cells from 0 to Ninh-1, rather than from Npyr to Npyr+Ninh-1    
            tmp = list(range(i_inh-rad,i_inh+rad+1)) #generate list of TC sources (but this will in general include negative values)
            tc_set=[val % self.Ntc for val in tmp] #apply modulus operation to eliminate negative numbers; this implements periodic boundary conditions; indexes TC cells from 0 to Ntc-1
            config.pc.gid2cell(inh_gid).k_TC_IN = 0 #make sure in-degree is initialized to zero
            for i_tc in tc_set:
                nc = config.pc.gid_connect(self.Npyr+self.Ninh+self.Nre+i_tc, config.pc.gid2cell(inh_gid).synlist[0]) # create NetCon by associating pre gid to post synapse; synlist[0] is AMPA synapse for INH cells
                nc.weight[0] = 1 #weight should always be 1; adjust strength by modifying synapse's gmax
                nc.delay = config.tc2inh_ampa_del
                nc.threshold = config.thresh
                self.nclist.append((self.Npyr+self.Ninh+self.Nre+i_tc,inh_gid,nc))
                config.pc.gid2cell(inh_gid).nclist.append((self.Npyr+self.Ninh+self.Nre+i_tc, nc))
                config.pc.gid2cell(inh_gid).k_TC_IN += 1
            #reduce gmax for this cell's synapse, so that the total synaptic strength is equal to that specified in the config file
            #(gmax was already set in cell_classes.py, which used the value specified in config.py)
            #for fac_AMPA_TC, see Krishnan's currents.cpp line 429, and main.cpp; note that network.cfg lists this as a D2 synapse with mini_f=0, which
            #doesn't make sense; but if it was somehow coded as a D2 synapse, then we should instead apply fac_AMPA_D2
            #so it is unclear whether we should apply fac_AMPA_TC or fac_AMPA_D2 here (I'm guessing D2)
            config.pc.gid2cell(inh_gid).synlist[0].gmax = config.init_AMPA_cort * config.tc2inh_ampa_str / config.pc.gid2cell(inh_gid).k_TC_IN

        ##### specify PYR->PYR AMPA_D2 connections
        rad=config.pyr2pyr_ampa_d2_rad #number of outgoing connections from each PY cell
        if (2*rad+1) > self.Npyr : print("******WARNING: PYR->PYR AMPA_D2 connectivity radius is too large.")
        for pyr_gid in self.pyr_gidList: #loop over all post-synaptic PYR cells on this machine (have to loop over post-syn cells bc. pc.gid_connect only works with post-synaptic gid on machine)
            i_post_pyr = pyr_gid #no subtraction needed, because pyramidal cells are the first set of gid's
            tmp = list(range(i_post_pyr-rad,i_post_pyr+rad+1)) #generate list of PYR sources (but this will in general include negative values)
            pyr_pre_set = [val % self.Npyr for val in tmp] #apply modulus operation to eliminate negative numbers; this implements periodic boundary conditions; indexes PYR cells from 0 to Nre-1
            config.pc.gid2cell(pyr_gid).k_PY_PY_AMPA = 0 #make sure in-degree is initialized to zero
            for i_pre_pyr in pyr_pre_set:
                if i_pre_pyr != i_post_pyr: #prevent self connections
                    nc = config.pc.gid_connect(i_pre_pyr, config.pc.gid2cell(pyr_gid).synlist[1]) # create NetCon by associating pre gid to post synapse; synlist[1] is AMPA_D2 synapse for PYR cells
                    nc.weight[0] = 1 #weight should always be 1; adjust strength by modifying synapse's gmax
                    nc.delay = config.pyr2pyr_ampa_d2_del
                    nc.threshold = config.thresh
                    self.nclist.append((i_pre_pyr,pyr_gid,nc))
                    config.pc.gid2cell(pyr_gid).nclist.append((i_pre_pyr,nc))
                    config.pc.gid2cell(pyr_gid).k_PY_PY_AMPA += 1
            #reduce gmax for this cell's synapse, so that the total synaptic strength is equal to that specified in the config file
            #(gmax was already set in cell_classes.py, which used the value specified in config.py)
            #note that this should also normalize the mini's, since their strength is set in the mod file by psp_weight, which is set in cell_classes.py
            #for fac_AMPA_D2, see Krishnan's currents.cpp line 545 and main.cpp lines 579-583, 639-776
            #config.pc.gid2cell(pyr_gid).synlist[1].gmax = config.awake_AMPAd2 * 0.4306688*(np.tanh(-config.ach_ion/0.82790214) + 1.17895946) / config.pc.gid2cell(pyr_gid).k_PY_PY_AMPA #see run_67; this used only when allowing ACh concentration to continuously vary within a simulation
            config.pc.gid2cell(pyr_gid).synlist[1].gmax = config.init_AMPA_pyrpyr * config.pyr2pyr_ampa_d2_str / config.pc.gid2cell(pyr_gid).k_PY_PY_AMPA
            
        ##### specify PYR->PYR NMDA_D1 connections   
        rad=config.pyr2pyr_nmda_d1_rad #number of outgoing connections from each PY cell
        if (2*rad+1) > self.Npyr : print("******WARNING: PYR->PYR NMDA_D1 connectivity radius is too large.")
        for pyr_gid in self.pyr_gidList: #loop over all post-synaptic PYR cells on this machine (have to loop over post-syn cells bc. pc.gid_connect only works with post-synaptic gid on machine)
            i_post_pyr = pyr_gid #no subtraction needed, because pyramidal cells are the first set of gid's
            tmp = list(range(i_post_pyr-rad,i_post_pyr+rad+1)) #generate list of PYR sources (but this will in general include negative values)
            pyr_pre_set = [val % self.Npyr for val in tmp] #apply modulus operation to eliminate negative numbers; this implements periodic boundary conditions; indexes PYR cells from 0 to Nre-1
            config.pc.gid2cell(pyr_gid).k_PY_PY_NMDA = 0 #make sure in-degree is initialized to zero 
            for i_pre_pyr in pyr_pre_set:
                if i_pre_pyr != i_post_pyr: #prevent self connections
                    nc = config.pc.gid_connect(i_pre_pyr, config.pc.gid2cell(pyr_gid).synlist[2]) # create NetCon by associating pre gid to post synapse; synlist[2] is NMDA_D1 synapse for PYR cells
                    nc.weight[0] = 1 #weight should always be 1; adjust strength by modifying synapse's gmax
                    nc.delay = config.pyr2pyr_nmda_d1_del
                    nc.threshold = config.thresh
                    self.nclist.append((i_pre_pyr,pyr_gid,nc))
                    config.pc.gid2cell(pyr_gid).nclist.append((i_pre_pyr,nc))
                    config.pc.gid2cell(pyr_gid).k_PY_PY_NMDA += 1
            #reduce gmax for this cell's synapse, so that the total synaptic strength is equal to that specified in the config file
            #(gmax was already set in cell_classes.py, which used the value specified in config.py)
            config.pc.gid2cell(pyr_gid).synlist[2].gmax = config.pc.gid2cell(pyr_gid).synlist[2].gmax / config.pc.gid2cell(pyr_gid).k_PY_PY_NMDA
        
        ##### specify PYR->INH AMPA_D2 connections 
        rad=config.pyr2inh_ampa_d2_rad #number of outgoing connnections from each PY cell
        p2i_ratio = self.Npyr/self.Ninh #we will assume that Npyr > Ninh; this ratio will be important for determining the center of the set of source PYR cells for each post-synaptic INH cell
        if p2i_ratio <= 1: print("******WARNING: This code assumes that Npyr > Ninh")
        if(2*rad+1) > self.Ninh : print("******WARNING: PYR->INH AMPA_D2 connectivity radius is too large.")
        
        for inh_gid in self.inh_gidList: #loop over all post-synaptic INH cells on this machine (have to loop over post-syn cells bc. pc.gid_connect only works with post-synaptic gid on machine)
            i_inh = inh_gid - self.Npyr #easier to work by indexing the INH cells from 0 to Ninh-1, rather than from Npyr to Npyr+Ninh-1
            tmp = list(range( int(np.floor((i_inh-rad)*p2i_ratio)), int(np.floor((i_inh+rad)*p2i_ratio)) +1) ) #center_pyr should be roughly round(i_inh*p2i_ratio); then you need to consider the radius of connectivity surrounding that, and consider that larger p2i_ratio increases the number of PYR cells sending to a given INH cell (for a particular radius of connectivity); "+1" is due to definition of Python's "range" function 
            pyr_set = [val % self.Npyr for val in tmp] #apply modulus operation to eliminate negative numbers; this implements periodic boundary conditions; indexes PYR cells from 0 to Npyr-1
            config.pc.gid2cell(inh_gid).k_PY_IN_AMPA = 0 #make sure in-degree is initialized to zero
            for i_pyr in pyr_set:
                nc = config.pc.gid_connect(i_pyr, config.pc.gid2cell(inh_gid).synlist[1]) # create NetCon by associating pre gid to post synapse; synlist[1] is AMPA_D2 synapse for INH cells
                nc.weight[0] = 1 #weight should always be 1; adjust strength by modifying synapse's gmax
                nc.delay = config.pyr2inh_ampa_d2_del
                nc.threshold = config.thresh
                self.nclist.append((i_pyr,inh_gid,nc))
                config.pc.gid2cell(inh_gid).nclist.append((i_pyr,nc))
                config.pc.gid2cell(inh_gid).k_PY_IN_AMPA += 1
            #reduce gmax for this cell's synapse, so that the total synaptic strength is equal to that specified in the config file
            #(gmax was already set in cell_classes.py, which used the value specified in config.py)
            #note that this should also normalize the mini's, since their strength is set in the mod file by psp_weight, which is set in cell_classes.py
            #for fac_AMPA_D2, see Krishnan's currents.cpp line 545 and main.cpp lines 579-583, 639-776
            config.pc.gid2cell(inh_gid).synlist[1].gmax = config.init_AMPA_cort * config.pyr2inh_ampa_d2_str / config.pc.gid2cell(inh_gid).k_PY_IN_AMPA
            
        
        ##### specify PYR->INH NMDA_D1 connections
        rad=config.pyr2inh_nmda_d1_rad #number of outgoing connnections from each PY cell
        p2i_ratio = self.Npyr/self.Ninh #we will assume that Npyr > Ninh; this ratio will be important for determining the center of the set of source PYR cells for each post-synaptic INH cell
        if p2i_ratio <= 1: print("******WARNING: This code assumes that Npyr > Ninh")
        if(2*rad+1) > self.Ninh : print("******WARNING: PYR->INH AMPA_D2 connectivity radius is too large.")
        
        for inh_gid in self.inh_gidList: #loop over all post-synaptic INH cells on this machine (have to loop over post-syn cells bc. pc.gid_connect only works with post-synaptic gid on machine)
            i_inh = inh_gid - self.Npyr #easier to work by indexing the INH cells from 0 to Ninh-1, rather than from Npyr to Npyr+Ninh-1
            tmp = list(range( int(np.floor((i_inh-rad)*p2i_ratio)), int(np.floor((i_inh+rad)*p2i_ratio)) +1) ) #center_pyr should be roughly round(i_inh*p2i_ratio); then you need to consider the radius of connectivity surrounding that, and consider that larger p2i_ratio increases the number of PYR cells sending to a given INH cell (for a particular radius of connectivity); "+1" is due to definition of Python's "range" function 
            pyr_set = [val % self.Npyr for val in tmp] #apply modulus operation to eliminate negative numbers; this implements periodic boundary conditions; indexes PYR cells from 0 to Npyr-1
            config.pc.gid2cell(inh_gid).k_PY_IN_NMDA = 0 #make sure in-degree is initialized to zero
            for i_pyr in pyr_set:
                nc = config.pc.gid_connect(i_pyr, config.pc.gid2cell(inh_gid).synlist[2]) # create NetCon by associating pre gid to post synapse; synlist[2] is NMDA_D1 synapse for INH cells
                nc.weight[0] = 1 #weight should always be 1; adjust strength by modifying synapse's gmax
                nc.delay = config.pyr2inh_nmda_d1_del
                nc.threshold = config.thresh
                self.nclist.append((i_pyr,inh_gid,nc))
                config.pc.gid2cell(inh_gid).nclist.append((i_pyr,nc))
                config.pc.gid2cell(inh_gid).k_PY_IN_NMDA += 1
            #reduce gmax for this cell's synapse, so that the total synaptic strength is equal to that specified in the config file
            #(gmax was already set in cell_classes.py, which used the value specified in config.py)
            config.pc.gid2cell(inh_gid).synlist[2].gmax = config.pc.gid2cell(inh_gid).synlist[2].gmax / config.pc.gid2cell(inh_gid).k_PY_IN_NMDA
        
        ##### specify PYR->TC AMPA connections
        rad=config.pyr2tc_ampa_rad #number of outgoing connections from each PYR cell
        p2t_ratio = self.Npyr/self.Ntc #we will assume that Npyr>Ntc; this ratio will be important for determining the center of the set of source PYR cells for each post-synaptic TC cell
        if p2t_ratio <= 1: print("*******WARNING: This code assumes that Npyr > Ntc")
        if(2*rad+1) > self.Ntc: print("*******WARNING: PYR->TC AMPA connectivity radius is too large.")
        
        for tc_gid in self.tc_gidList: #loop over all post-synaptic TC cells on this machine (have to loop over post-syn cells bc. pc.gid_connect only works with post-synaptic gid on machine)
            i_tc = tc_gid - (self.Npyr+self.Ninh+self.Nre) #easier to work by indexing the TC cells from 0 to Ntc-1, rather than from Npyr+Ninh+Nre to from Npyr+Ninh+Nre+Ntc-1
            tmp = list(range( int(np.floor((i_tc-rad)*p2t_ratio)), int(np.floor((i_tc+rad)*p2t_ratio)) +1) ) #center_pyr should be roughly round(i_tc*p2i_ratio); then you need to consider the radius of connectivity surrounding that, and consider that larger p2t_ratio increases the number of PYR cells sending to a given TC cell (for a particular radius of connectivity); "+1" is due to definition of Python's "range" function 
            pyr_set = [val % self.Npyr for val in tmp] #apply modulus operation to eliminate negative numbers; this implements periodic boundary conditions; indexes PYR cells from 0 to Npyr-1
            config.pc.gid2cell(tc_gid).k_PY_TC = 0 #make sure in-degree is initialized to zero
            for i_pyr in pyr_set:
                nc = config.pc.gid_connect(i_pyr,config.pc.gid2cell(tc_gid).synlist[2]) # create NetCon by associating pre gid to post synapse; synlist[2] is AMPA synapse for TC cells
                nc.weight[0] = 1 #weight should always be 1; adjust strength by modifying synapse's gmax
                nc.delay = config.pyr2tc_ampa_del
                nc.threshold = config.thresh
                self.nclist.append((i_pyr,tc_gid,nc))
                config.pc.gid2cell(tc_gid).nclist.append((i_pyr,nc))
                config.pc.gid2cell(tc_gid).k_PY_TC += 1
            #reduce gmax for this cell's synapse, so that the total synaptic strength is equal to that specified in the config file
            #(gmax was already set in cell_classes.py, which used the value specified in config.py)
            #for fac_AMPA_TC, see currents.cpp line 429, and main.cpp
            config.pc.gid2cell(tc_gid).synlist[2].gmax = config.init_AMPA_thal * config.pyr2tc_ampa_str / config.pc.gid2cell(tc_gid).k_PY_TC
        
        ##### specify PYR->RE AMPA connections
        rad=config.pyr2re_ampa_rad #number of outgoing connections from each PYR cell
        p2r_ratio = self.Npyr/self.Nre #we will assume that Npyr>Nre; this ratio will be important for determining the center of the set of source PYR cells for each post-synaptic RE cell
        if p2r_ratio <= 1: print("*******WARNING: This code assumes that Npyr > Nre")
        if(2*rad+1) > self.Nre: print("*******WARNING: PYR->RE AMPA connectivity radius is too large.")
        
        for re_gid in self.re_gidList: #loop over all post-synaptic RE cells on this machine (have to loop over post-syn cells bc. pc.gid_connect only works with post-synaptic gid on machine)
            i_re = re_gid - (self.Npyr+self.Ninh) #easier to work by indexing the TC cells from 0 to Nre-1, rather than from Npyr+Ninh to from Npyr+Ninh+Nre-1
            tmp = list(range( int(np.floor((i_re-rad)*p2r_ratio)), int(np.floor((i_re+rad)*p2r_ratio)) +1) ) #center_pyr should be roughly round(i_re*p2i_ratio); then you need to consider the radius of connectivity surrounding that, and consider that larger p2r_ratio increases the number of PYR cells sending to a given RE cell (for a particular radius of connectivity); "+1" is due to definition of Python's "range" function 
            pyr_set = [val % self.Npyr for val in tmp] #apply modulus operation to eliminate negative numbers; this implements periodic boundary conditions; indexes PYR cells from 0 to Npyr-1
            config.pc.gid2cell(re_gid).k_PY_RE = 0 #make sure in-degree is initialized to zero
            for i_pyr in pyr_set:
                nc = config.pc.gid_connect(i_pyr,config.pc.gid2cell(re_gid).synlist[1]) # create NetCon by associating pre gid to post synapse; synlist[1] is AMPA synapse for RE cells
                nc.weight[0] = 1 #weight should always be 1; adjust strength by modifying synapse's gmax
                nc.delay = config.pyr2re_ampa_del
                nc.threshold = config.thresh
                self.nclist.append((i_pyr,re_gid,nc))
                config.pc.gid2cell(re_gid).nclist.append((i_pyr,nc))
                config.pc.gid2cell(re_gid).k_PY_RE += 1
            #reduce gmax for this cell's synapse, so that the total synaptic strength is equal to that specified in the config file
            #(gmax was already set in cell_classes.py, which used the value specified in config.py)
            #for fac_AMPA_TC, see currents.cpp line 429, and main.cpp
            config.pc.gid2cell(re_gid).synlist[1].gmax = config.init_AMPA_thal * config.pyr2re_ampa_str / config.pc.gid2cell(re_gid).k_PY_RE
            
        ##### specify INH->PYR GABA_A_D2 connections 
        rad=config.inh2pyr_gaba_a_d2_rad #number of outgoing connections from each INH cell
        p2i_ratio = self.Npyr/self.Ninh #we will assume that Npyr > Ninh; this ratio will be important for determining the center of the set of source INH cells for each post-synaptic PYR cell
        if p2i_ratio <= 1: print("******WARNING: This code assumes that Npyr > Ninh")
        if (2*rad+1) > self.Npyr : print("******WARNING: INH->PYR GABA_A_D2 connectivity radius is too large.")
        
        for pyr_gid in self.pyr_gidList: #loop over all post-synaptic PYR cells on this machine (have to loop over post-syn cells bc. pc.gid_connect only works with post-synaptic gid on machine)
            i_pyr = pyr_gid #no subtraction needed, because pyramidal cells are the first set of gid's
            tmp = list(range( int(np.floor((i_pyr-rad)/p2i_ratio)) , int(np.floor((i_pyr+rad)/p2i_ratio))+1 ) )  #center_inh should be roughly round(i_pyr/p2i_ratio); then you need to consider the radius of connectivity surrounding that, and consider that larger p2i_ratio decreases the number of INH cells sending to a given PYR cell (for a particular radius of connectivity); "+1" is due to definition of Python's "range" function 
            inh_set = [val % self.Ninh for val in tmp] #apply modulus operation to eliminate negative numbers; this implements periodic boundary conditions; indexes INH cells from 0 to Ninh-1
            config.pc.gid2cell(pyr_gid).k_IN_PY = 0 #make sure in-degree is initialized to zero 
            for i_inh in inh_set:
                nc = config.pc.gid_connect(self.Npyr+i_inh, config.pc.gid2cell(pyr_gid).synlist[3]) # create NetCon by associating pre gid to post synapse; synlist[3] is GABA_A_D2 synapse for PYR cells
                nc.weight[0] = 1 #weight should always be 1; adjust strength by modifying synapse's gmax
                nc.delay = config.inh2pyr_gaba_a_d2_del
                nc.threshold = config.thresh
                self.nclist.append((self.Npyr+i_inh,pyr_gid,nc))
                config.pc.gid2cell(pyr_gid).nclist.append((self.Npyr+i_inh,nc))
                config.pc.gid2cell(pyr_gid).k_IN_PY += 1
            #reduce gmax for this cell's synapse, so that the total synaptic strength is equal to that specified in the config file
            #(gmax was already set in cell_classes.py, which used the value specified in config.py)
            #note that this should also normalize the mini's, since their strength is set in the mod file by psp_weight, which is set in cell_classes.py
            #for fac_GABA_D2, see Krishnan's currents.cpp line 755, and main.cpp
            config.pc.gid2cell(pyr_gid).synlist[3].gmax = config.init_GABA_D2 * config.inh2pyr_gaba_a_d2_str / config.pc.gid2cell(pyr_gid).k_IN_PY
            
            
    def createStims(self):
        pass
    
    def createIClamps(self):
        pass
        #config.pc.gid2cell(0).createIClamp(amp=50)
        
    def setCellLocations(self):
        '''set cell locations for calculating distance from recording electrode. Also set pointers necessary for
        xtra and extracellular mechanisms to work. (This code is adapted from HFO recode version 12)'''
        h.define_shape() #this assigns default position and orientation of all cell. These default positions must be changed below...
        
        dist_cell=np.sqrt(config.area_cell) #closest linear distance between cells, if laid out on a square grid
        
        #place soma where you want them (understanding that default orientation of each cell is unchanged)                     
        Npyr=self.Npyr
        Ninh=self.Ninh
        Ntot=Npyr+Ninh
        
        pyr2inh_ratio=Npyr/Ninh
        #for ease of coding, I am going to require that Npyr be a multiple of Ninh
        assert Npyr%Ninh==0, "Error: Number of pyramidal cells must be a multiple of the number of inhibitory cells."
        
        #fist column is x-coordinates, second column y-coordinates, third column z-coordinates
        pyrcoords=np.zeros((Npyr,3))
        inhcoords=np.zeros((Ninh,3))
        
        #lay out cells in rings around origin. Each ring will be a distance dist_cell further from the origin than the previous one, and within
        #each ring, cells will be spaced 'dist_cell' from one another. Inhibitory cells are interspersed amongst pyramidal cells
        ring_ind=1 #current ring index. Radius of ring will be ring_ind*dist_cell
        intraring_loc=0 #lattice location within present ring
        cells_laid=0 #number of cells laid out
        pyrcells_laid=0
        inhcells_laid=0
        cells_in_ring=int(np.floor(2.0*np.pi*ring_ind)) #cells to place in next ring; take circumference divided by cell spacing: 2*np.pi*(ring_ind*dist_cell)/dist_cell
        phi=np.linspace(0,2*np.pi,cells_in_ring+1) #calculate angular coordinate for each cell; "+1" results in a "dummy cell" being placed at angle of 2*pi, so that we don't place an actual cell there. 
        phi=phi+(phi[1]-phi[0])/2.0 #rotate everything so that first cell isn't always placed along the y-axis
        while (cells_laid<Ntot):
            
            #set x- and y- coordinates
            if(cells_laid % (pyr2inh_ratio+1) == 0):
                inhcoords[inhcells_laid,1]=(dist_cell*ring_ind)*np.cos(phi[intraring_loc])
                inhcoords[inhcells_laid,2]=(dist_cell*ring_ind)*np.sin(phi[intraring_loc])
                inhcells_laid += 1
            else:
                pyrcoords[pyrcells_laid,1]=(dist_cell*ring_ind)*np.cos(phi[intraring_loc])
                pyrcoords[pyrcells_laid,2]=(dist_cell*ring_ind)*np.sin(phi[intraring_loc])        
                pyrcells_laid += 1
                
            intraring_loc += 1
            cells_laid += 1
            
            if(intraring_loc == cells_in_ring and cells_laid != Ntot): #condition for having completed a ring
                ring_ind += 1 #move out to the next ring
                intraring_loc=0 #reset lattice location within present ring
                
                cells_in_ring=int(np.floor(2.0*np.pi*ring_ind)) #cells to place in next ring; take circumference divided by cell spacing: 2*np.pi*(ring_ind*dist_cell)/dist_cell
                if(cells_laid+cells_in_ring > Ntot): #if the next ring can accommodate more neurons than is necessary, then have it only house the remaining neurons (this will make them symmetrically distributed around the ring)
                    cells_in_ring = Ntot - cells_laid
                    
                phi=np.linspace(0,2*np.pi,cells_in_ring+1)
                phi=phi+(phi[1]-phi[0])/2.0
        
        #place soma where you want them (understanding that default orientation of each cell is unchanged) 
        #loop over cells and determine which kind they are, then find appropriate row in appropriate matrix to grab coordinates
        #need to do loop over self.cells, rather than just looping over all gid values, because if this is run in parallel
        #each processor only has a subset of all possible cells                    
        for cell in self.cells:
            if cell.gid<Npyr:
                row = cell.gid
                self.place_cell(cell,pyrcoords[row,0],pyrcoords[row,1],pyrcoords[row,2])
            elif Npyr <= cell.gid < Npyr + Ninh:
                row = cell.gid-Npyr
                self.place_cell(cell,inhcoords[row,0],inhcoords[row,1],inhcoords[row,2])
        
        h.define_shape() #call this again, so that all the other sections attached to each soma will move with it
        
        #print out locations of each section
        for sec in h.allsec():
            print("section=",sec.name()) #or use print h.secname()
            for i in range(int(h.n3d())): print(i, h.x3d(i), h.y3d(i), h.z3d(i), h.diam3d(i))
            
        #call grindaway() in order to calculate centers of all segements, and--most importantly--copy these values to the xtra mechanism of each cell, so they can be used to calculate the LFP
        h.load_file('interpxyz.hoc') #from Ted's extracellular_stim_and_rec code; see https://www.neuron.yale.edu/phpBB/viewtopic.php?f=8&t=3649
        h('grindaway()') 
        
        #this is basically replicating setpointers.hoc from Ted's extracellular_stim_and_rec code
        for sec in h.allsec():
            if h.ismembrane('xtra',sec=sec):
                for seg in sec:
                    h.setpointer(seg._ref_i_membrane_, 'im', seg.xtra)

        #h('xopen("calcrxc_a.hoc")') #h.load_file is like Python import, h('xopen...') is like Python execfile
        h.load_file('calcrxc_a.hoc')
        h.setelec(config.XE, config.YE, config.ZE) # put stim electrode at (x, y, z)); fyi, x-coordinates of neurons by default range from (CF: the following applies to HFO recode, not Bazhenov model)-200 um (tip of basilar dendrite) to +470 um (tip of apical dendrite) 
        
    def place_cell(self,cell,newx,newy,newz):
        '''places zeroth 3D coordinate of soma of 'cell' at a specified x,y,z coordinate, maintaining default orientation of cell 
        (cells appear to be parallel to the x-axis) (This code is copied from HFO recode version 12)'''
        n = int(h.n3d(sec=cell.soma))
        #create lists of x, y, and z coordinates for all segments in the soma (see p. 256, 2017 Neuron Course manual)
        xs = [h.x3d(i, sec=cell.soma) for i in range(n)]
        ys = [h.y3d(i, sec=cell.soma) for i in range(n)]
        zs = [h.z3d(i, sec=cell.soma) for i in range(n)]
        ds = [h.diam3d(i, sec=cell.soma) for i in range(n)]
              
        #iterate through each 3D coordinate and re-assign it
        i=0
        for a,b,c,d in zip(xs, ys, zs, ds):
            h.pt3dchange(i,(a-xs[0])+newx,(b-ys[0])+newy,(c-zs[0])+newz,d,sec=cell.soma ) # part in parentheses gives location of non-zeroth coordinates relative to zeroth coordinate
            i+=1
    
    def gatherSpikes(self):
        """Gather spikes from all nodes/hosts"""
        if config.idhost==0: print('Gathering spikes ...')
        
        data = [None]*config.nhost
        data[0] = {'tVec': self.tVec, 'idVec': self.idVec}
        config.pc.barrier()
        gather=config.pc.py_alltoall(data)
        config.pc.barrier()
        self.tVecAll = [] 
        self.idVecAll = [] 
        if config.idhost==0:
            for d in gather:
                self.tVecAll.extend(list(d['tVec']))
                self.idVecAll.extend(list(d['idVec']))
                
    def gatherLFP(self):
        '''Gather LFP waveforms from all nodes/hosts'''
        if config.idhost==0: print('Gathering LFP waveforms ...')
        data = [None]*config.nhost #EACH NODE has this list, the i^th element of which will be sent to node i
        data[0] = {'lfp': config.lfp_rec, 'v_rec': config.v_rec} #by making only the zeroth element something other than 'None,' this means each node will be sending data only to node 0
        config.pc.barrier()
        gather=config.pc.py_alltoall(data) #according to Lytton paper, 'gather' is a list
        config.pc.barrier() 
        if config.idhost==0:
            print(len(gather[0]['v_rec']))
            print(len(gather[0]['lfp']))
            self.v_sum=np.zeros(len(gather[0]['v_rec'])) #start sum at zeros, and make np array same length as v_rec lists
            self.lfp_sum=np.zeros(len(gather[0]['lfp'])) #start sum at zeros, and make np array same length as lfp_rec lists
            for d in gather:
                self.v_sum += d['v_rec'] #compute summed cortical voltage, summed over contributions from nodes on all hosts
                self.lfp_sum += d['lfp'] #compute cortical LFP, summed over contributions from nodes on all hosts
                
    def plotRaster(self):

        print('Plotting raster ...')
        pyr_indices=[i for (i, val) in enumerate(self.idVecAll) if val<self.Npyr] #basically the equivalent of [i,val]=find(idVeCall<Npyr) in Matlab to get the indices of pyramidal cell spikes
        inh_indices=[i for (i, val) in enumerate(self.idVecAll) if (self.Npyr<=val<(self.Npyr+self.Ninh))] #find indices (within self.idVecAll) of inhibitory cell spikes
        re_indices=[i for (i, val) in enumerate(self.idVecAll) if ((self.Npyr+self.Ninh)<=val<(self.Npyr+self.Ninh+self.Nre))]
        tc_indices=[i for (i, val) in enumerate(self.idVecAll) if ((self.Npyr+self.Ninh+self.Nre)<=val)]
        
        pyr_tVec=[net.tVecAll[val] for val in pyr_indices] #create spike time vector of just pyramidal cells
        pyr_id=[net.idVecAll[val] for val in pyr_indices] #create corresponding spike id vector for just pyramidal cells
        inh_tVec=[net.tVecAll[val] for val in inh_indices]
        inh_id=[net.idVecAll[val] for val in inh_indices]
        re_tVec=[net.tVecAll[val] for val in re_indices]
        re_id=[net.idVecAll[val] for val in re_indices]
        tc_tVec=[net.tVecAll[val] for val in tc_indices]
        tc_id=[net.idVecAll[val] for val in tc_indices]
        
        pyplot.figure()
        pyplot.scatter(pyr_tVec,pyr_id,marker="o",s=5,color='red')
        pyplot.scatter(inh_tVec,inh_id,marker="o",s=5,color='blue')
        pyplot.scatter(re_tVec,re_id,marker="o",s=5,color='green')
        pyplot.scatter(tc_tVec,tc_id,marker="o",s=5,color='orange')
        pyplot.xlabel('Time (ms)')
        pyplot.ylabel('Cell ID')
        pyplot.title('Raster Plot')
        if len(self.tVecAll)>0:
            pyplot.xlim(0,1.05*max(self.tVecAll))
        else:
            print("tVecAll is empty.")
        pyplot.ylim(0,self.N)
        pyplot.show()
        #pyplot.savefig('raster')
        
    def saveData(self):
        print('Saving data ...')
        dataSave = {'Npyr': self.Npyr, 'Nbask': self.Ninh, 'Nre':self.Nre, 'Ntc':self.Ntc, 'tVec': self.tVecAll, 'idVec': self.idVec} #may want to add in all the connectivity and stimulation parameters at some point
        with open('output.pkl', 'wb') as f:
            pickle.dump(dataSave, f)
