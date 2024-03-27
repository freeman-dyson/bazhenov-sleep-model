"""
Define classes for four types of cells (cortical pyramidal, cortical inhibitory, thalamocortical, reticular)
References to Bazhenov 2002 are to this paper: 10.1523/JNEUROSCI.22-19-08691.2002
"""

import config
from neuron import h
import numpy as np
h('load_file("stdgui.hoc")')

class Cell(object):
    '''all other classes of neurons should inherit from Cell'''
    def __init__(self,gid):
        self.gid = gid
        self.synlist = []
        self.createSections()
        self.buildTopology()
        self.defineGeometry()
        self.defineBiophysics()
        self.applyDC()
        self.createSynapses()
        self.createInDegVars()
        self.nclist = []
    #
    def createSections(self):
        pass
    #
    def buildTopology(self):
        pass
    #
    def defineGeometry(self):
        """Set the 3D geometry of the cell."""
        pass
    #
    def defineBiophysics(self):
        pass
    #
    def applyDC(self):
        pass
    #
    def createSynapses(self):
        pass
    
    def createInDegVars(self): 
        '''Keep track of number of incoming connections from each different species of neuron, for each kind of synapse.
        This is used to normalize synapse 'gmax' values in Net's 'connectCells' method'''
        pass
    
    def associateGid(self):
        print("self.gid=%d, config.idhost=%d"%(self.gid,config.idhost))
        config.pc.set_gid2node(self.gid, config.idhost)
        nc = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        nc.threshold = config.thresh
        config.pc.cell(self.gid, nc)
        del nc # discard netcon, since its only purpose is to detect spikes, and the previous line of code will now make sure that happens
    
    def createNetcon(self, thresh=0):
        """ created netcon to record spikes """
        nc = h.NetCon(self.soma(0.5)._ref_v, None, sec = self.soma)
        nc.threshold = config.thresh
        return nc
    
    def setRecording(self):
        """Set soma, dendrite, and time recording vectors on the cell. """
        self.soma_v_vec = h.Vector()   # Membrane potential vector at soma
        self.tVec = h.Vector()        # Time stamp vector
        self.soma_v_vec.record(self.soma(0.5)._ref_v)
        self.tVec.record(h._ref_t)
        
    def plotTraces(self):
        """Plot the recorded traces"""
        pyplot.figure() # Default figsize is (8,6)
        somaPlot = pyplot.plot(self.tVec, self.soma_v_vec, color='black')
        pyplot.legend(somaPlot, ['soma'])
        pyplot.xlabel('time (ms)')
        pyplot.ylabel('mV')
        pyplot.title('Cell %d voltage trace'%(self.gid))
        pyplot.show()
        #pyplot.savefig('traces')
        
    def createIClamp(self,amp):
        #add depolarizing current
        self.Iext = h.IClamp(self.soma(0.5))
        self.Iext.dur = 1e9
        self.Iext.delay = 0
        self.Iext.amp = amp*np.pi*self.soma.diam*self.soma.L*1e-8*1e3 # 'amp' is specified in uA/cm2 (for comparison with Bazhenov code); the rest of this converts to nA, since that is what NEURON requires
        
    #this method makes it so that when you print my_cell.soma, it returns Cell[#].soma, rather than an identifier based on the location in memory
    def __str__(self):
        return 'Cell[{}]'.format(self.gid)
    
class PyrCell(Cell):
    def createSections(self):
        self.soma=h.Section(name='soma', cell=self) #create object 'soma,' and make object variable 'name' equal to 'soma'
        self.dend=h.Section(name='dend', cell=self) #for some reason, it is essential to include 'cell=self' in order for 'associateGid' to work properly (for more, see https://www.neuron.yale.edu/phpBB/viewtopic.php?f=2&t=2576)
        
    def defineGeometry(self):
        self.soma.diam = self.soma.L = np.sqrt(100/np.pi) #microns (gives an area of 100 microns^2), according to Bazhenov 2002; see also variable s_cx_soma in currents.cpp and currents.hof C++ code)
        self.dend.diam = self.soma.diam #microns (might as well make it the same diameter as soma)
        self.dend.L=165*self.soma.L #comes from r=165 for cortical pyramidal cell, specified right before "synaptic currents" section of Bazhenov 2002
        self.soma.nseg=1
        self.dend.nseg=1
        self.all=self.soma.wholetree()
        
    def buildTopology(self):
        self.dend.connect(self.soma(1)) # https://neuron.yale.edu/neuron/static/docs/neuronpython/ballandstick1.html

    def defineBiophysics(self):
        
        if config.doextra: #necessary to record LFP
            for sec in self.all:
                sec.insert('xtra')
            
        # this post is extremely helpful for setting the axial resistivity to match the absolute values described in the line above: https://www.neuron.yale.edu/phpBB/viewtopic.php?f=8&t=3904&p=16841&hilit=axial+resistance#p16841
        geom_factor = (self.soma.L/2)/(np.pi * (self.soma.diam/2)**2) + (self.dend.L/2)/(np.pi * (self.dend.diam/2)**2)  #axial resistance from middle of soma to end where it meets dendrite is soma.L/2; then divide by area and multiple by axial resistivity to get resistance. Then do the same to calculate resistance from soma/dendrite junction to halfway along dendrite
        self.soma.Ra = 1e7/geom_factor * 1e-4 #1e7 is because axial resistance is 10 MegaOhms (Bazhenov 2002); factor of 1e-4 converts from Ohms*micrometers to Ohms*cm
        self.dend.Ra = 1e7/geom_factor * 1e-4

        self.soma.cm = 0.75 #currents.cpp line 1082
        self.soma.insert('kdr')
        self.soma.gkbar_kdr=0.200 # S/cm2 (200 mS/cm2); currents.h line 1235
        
        self.soma.insert('naf')
        self.soma.gnabar_naf=3.0 # S/cm2; currents.h line 1236
        
        self.soma.insert('nap')
        self.soma.gnabar_nap=0.0003 #S/cm2, currents.h line 1237 sets G_Nap=15 mS/cm2, then uses f=0.02 in activation function, which effectively divides G_Nap by 50
        
        self.dend.cm = 0.75
        self.dend.insert('naf')
        self.dend.gnabar_naf=0.0008 #S/cm2, currents.h line 1202

        self.dend.insert('nap')
        self.dend.gnabar_nap=0.000042 #S/cm2, currents.h line 1205 says 2.1 mS/cm2, but need to divide this by 50 bc. the activation function for Inap_CX (currents.h line 546) uses 0.02 instead of 1 in the numerator
      
        self.dend.insert('hva')
        self.dend.gcabar_hva=0.000012 # S/cm2, currents.h line 1208
        self.dend.eca = 140 #currents.cpp line 200 (see also comment on line 205)
        
        self.dend.insert('km')
        self.dend.gkbar_km=0.00002 # S/cm2, CellSyn.h line 364 says 0.02 mS/cm2
        
        self.dend.insert('cad')  #calcium accumulation mechanism
        self.dend.taur_cad=165 #ms; currents.h line 1204
        self.dend.depth_cad=1.0 #micrometer; currents.h line 339
        self.dend.cainf_cad=2.4e-4 #mM; currents.cpp line 133
        
        self.dend.insert('kca') #kca.mod depends on cadecay.mod
        self.dend.gkcabar_kca=0.00005 # S/cm2, currents.h line 1207
        
        self.dend.insert('kL')
        h.krev_kL = -95 #Bazhenov et. al. use -95mV as ek for potassium leak current, but -90mV for other potassium currents in PYR and INH cells (currents.cpp line 103)
        self.dend.gkL_kL = config.init_gkl_pyr # S/cm2; see CellSyn.h line 362
        
        self.dend.insert('pas')
        self.dend.g_pas = 0.000011 #S/cm2; this is set in CellSyn.h line 365. NOTE: Following C++ code, we add variability to this parameter in the createCells method of network_class.py
        self.dend.e_pas = -67 # see CellSyn.h line 363
        
        #set ena and ek for both soma and dendrite
        self.soma.ek=-90 # see currents.cpp lines 225 and 240; note that currents.cpp line 1066 uses ek=-95mV for just the potassium leak conductance
        self.soma.ena=50 # currents.cpp line 275
        self.dend.ek=-90 # currents.cpp line 260
        self.dend.ena=50 # currents.cpp line 275
        
        h.ion_style("ca_ion",3,1,0,0,0,sec=self.dend) #this ensures that cai is treated as a dynamical variable, while eca is a fixed parameter (see currents.cpp line 205)
        
    def applyDC(self):
        '''we found that 0.0417 uA/cm2 must be injected into pyramidal cell in order for its rheobase (and entire f-I curve) to match
        that of Bazhenov's reduced model'''
        self.DCstim = h.IClamp(self.dend(0.5)) #in Bazhenov 2002 C++ source code, I generated voltage traces based on current injected into dendrite, NOT soma
        self.DCstim.dur = 1e9
        self.DCstim.amp = 0.0417*np.pi*self.dend.diam*self.dend.L*1e-8*1e3  # nano-Amps; to match C++ pyramidal cell's rheobase, this should be 0.0417 uA/cm2; conversion from uA/cm2 (x) to nA: x*np.pi*soma.diam*soma.L*1e-8*1e3 
        self.DCstim.delay = 0
        
    def createSynapses(self):      
        #we found that for all synapses onto pyramidal cells, synaptic strengths in my model should be 75% the strength in the C++ model
        #(presumably due to the difference between our full model and their reduced model)
        syn = h.AMPA(self.dend(0.5)) #TC->PY synapse
        syn.gmax = config.tc2pyr_ampa_str 
        self.synlist.append(syn)
       
        syn = h.AMPA_D2(self.dend(0.5)) #PY->PY synapse
        syn.gmax = config.pyr2pyr_ampa_d2_str 
        syn.gid = self.gid
        syn.syn_index = 0 #make syn_index for PY->PY equal to 0, and syn_index for IN->PY equal to 1 (see note on syn_index in ampa_D2.mod)
        syn.setrand(syn.gid, syn.syn_index) #need to set this here, rather than in INITIAL block of mod file, bc. need to make sure that random number generator is set before gen_nextpsp is called in NET_RECEIVE INITIAL block 
        if config.pyr2pyr_ampa_d2_str > 0: #prevent division by zero
            syn.psp_weight = config.pyr2pyr2_ampa_d2_mini_str/config.pyr2pyr_ampa_d2_str #we wrote the mod file so that 'psp_weight' is a normalized value that tells you how much stronger (for values greater than 1) or weaker (values less than 1) the stochastic stimulation is than the regular presynaptic weight
        else:
            syn.psp_weight = 0 #if regular pyr2pyr_ampa_d2 connections have strength of zero, then stochastic stimulation of these synapses will also be absent
        h.SS_denom_AMPA_D2 = 250.0 #C++ hard codes this as 250.0 ms in line 520 of currents.cpp (need to use "h.*" bc. this is a GLOBAL variable in mod file); this parameter controls the mean inter-event interval of stochastic EPSP's (so smaller value will lead to more frequent stochastic EPSP's)
        h.mini_fre_AMPA_D2 = config.pyr2pyr2_ampa_d2_mini_f 
        self.synlist.append(syn)
        
        syn = h.NMDA_D1(self.dend(0.5)) #PY->PY NMDA synapse
        syn.gmax = config.pyr2pyr_nmda_d1_str 
        h.Use_NMDA_D1 = config.pyr2pyr_nmda_d1_Use 
        self.synlist.append(syn)
        
        syn = h.GABA_A_D2(self.dend(0.5)) #IN->PY synapse
        syn.gmax = config.inh2pyr_gaba_a_d2_str 
        syn.gid = self.gid
        syn.syn_index = 1 #make syn_index for PY->PY equal to 0, and syn_index for IN->PY equal to 1 (see note on syn_index in ampa_D2.mod)
        syn.setrand(syn.gid, syn.syn_index) #need to set this here, rather than in INITIAL block of mod file, bc. need to make sure that random number generator is set before gen_nextpsp is called in NET_RECEIVE INITIAL block
        if config.inh2pyr_gaba_a_d2_str > 0: #prevent division by zero
            syn.psp_weight = config.inh2pyr_gaba_a_d2_mini_str/config.inh2pyr_gaba_a_d2_str #see note for 'psp_weight' above
        else:
            syn.psp_weight = 0 #if regular inh2pyr_gaba_a_d2 connections have zero strength, then stochastic stimulation of these synapses will also be absent
        h.SS_denom_GABA_A_D2 = 250.0 #C++ hard codes this as 250.0 ms in line 520 of currents.cpp (need to use "h.*" bc. this is a GLOBAL variable in mod file)
        h.mini_fre_GABA_A_D2 = config.inh2pyr_gaba_a_d2_mini_f 
        self.synlist.append(syn)
        
    def createInDegVars(self):
        self.k_TC_PY = 0
        self.k_PY_PY_AMPA = 0
        self.k_PY_PY_NMDA = 0
        self.k_IN_PY = 0
    
class InhCell(Cell):
    '''
Almost the same as PyrCell, except for the following points:
--the length of the dendrite is 50 times the length of the soma, rather than 165 (according to Bazhenov 2002 paper and CellSyn.h line 518)
--removal of Nap current (CellSyn.h lines 519-520)
--set soma.gnabar_naf=2.5, rather than 3.0 (CellSyn.h line 521)
--set g_kL=0.000009 S/cm2 (CellSyn.h line 525)
--gkbar_km=0.000015 # S/cm2 (currents.h line 1210)
--E_L=-70 mV (currents.h line 1200)
--set G_l = 0.009 + (((double) rand() / (RAND_MAX)) + 1) * 0.003; (CellSyn.h line 526)
'''
    def createSections(self):
        self.soma=h.Section(name='soma', cell=self) #create object 'soma,' and make object variable 'name' equal to 'soma'
        self.dend=h.Section(name='dend', cell=self)
        
    def defineGeometry(self):
        self.soma.diam = self.soma.L = np.sqrt(100/np.pi) 
        self.dend.diam = self.soma.diam 
        self.dend.L=50.0*self.soma.L #comes from r=50 for cortical basket cell, specified right before "synaptic currents" section of Bazhenov 2002
        self.soma.nseg=1
        self.dend.nseg=1
        self.all=self.soma.wholetree()
        
    def buildTopology(self):
        self.dend.connect(self.soma(1)) # https://neuron.yale.edu/neuron/static/docs/neuronpython/ballandstick1.html

    def defineBiophysics(self):
        
        if config.doextra: #necessary to record LFP
            for sec in self.all:
                sec.insert('xtra')
                
        # this post is extremely helpful for setting the axial resistivity to match the absolute values described in the line above: https://www.neuron.yale.edu/phpBB/viewtopic.php?f=8&t=3904&p=16841&hilit=axial+resistance#p16841
        geom_factor = (self.soma.L/2)/(np.pi * (self.soma.diam/2)**2) + (self.dend.L/2)/(np.pi * (self.dend.diam/2)**2)  #axial resistance from middle of soma to end where it meets dendrite is soma.L/2; then divide by area and multiple by axial resistivity to get resistance. Then do the same to calculate resistance from soma/dendrite junction to halfway along dendrite
        self.soma.Ra = 1e7/geom_factor * 1e-4 #1e7 is because axial resistance is 10 MegaOhms (Bazhenov 2002); factor of 1e-4 converts from Ohms*micrometers to Ohms*cm
        self.dend.Ra = 1e7/geom_factor * 1e-4

        self.soma.cm = 0.75
        self.soma.insert('kdr')
        self.soma.gkbar_kdr=0.200 # S/cm2; currents.h line 1235
        
        self.soma.insert('naf')
        self.soma.gnabar_naf=2.5 # S/cm2; CellSyn.h line 521
        
        self.dend.cm = 0.75
        self.dend.insert('naf')
        self.dend.gnabar_naf=0.0008 #S/cm2; currents.h line 1202

        self.dend.insert('hva')
        self.dend.gcabar_hva=0.000012 # S/cm2; currents.h line 1208
        self.dend.eca = 140 # currents.cpp line 200 (see also comment on line 205)
        
        self.dend.insert('km')
        self.dend.gkbar_km=0.000015 # S/cm2, currents.h line 1210
        
        self.dend.insert('cad')  #calcium accumulation mechanism
        self.dend.taur_cad=165 #ms; currents.h line 1204
        self.dend.depth_cad=1.0 #micrometer; currents.h line 339
        self.dend.cainf_cad=2.4e-4 #mM; currents.cpp line 133
        
        self.dend.insert('kca') #kca.mod depends on cadecay.mod
        self.dend.gkcabar_kca=0.00005 # S/cm2, currents.h line 1207
        
        self.dend.insert('kL')
        h.krev_kL = -95 #Bazhenov et. al. use -95mV as ek for potassium leak current, but -90mV for other potassium currents in PYR and INH cells (currents.cpp line 103)
        self.dend.gkL_kL = config.init_gkl_inh #S/cm2; CellSyn.h line 525
        
        self.dend.insert('pas')
        self.dend.g_pas = 0.000009 #S/cm2; see CellSyn.h line 526; NOTE: Following C++ code, we add variability to this parameter in the createCells method of network_class.py
        self.dend.e_pas = -70.0 # currents.h line 1200
        
        #set ena and ek for both soma and dendrite (see corresponding comments for PYR cell above)
        self.soma.ek=-90 
        self.soma.ena=50 
        self.dend.ek=-90
        self.dend.ena=50
        
        h.ion_style("ca_ion",3,1,0,0,0,sec=self.dend) #this ensures that cai is treated as a dynamical variable, while eca is a fixed parameter (see currents.cpp line 205)
        
    def applyDC(self):
        '''we found 0.1091 uA/cm2 must be injected into inhibitory cell in order 
        for its rheobase (and entire f-I curve) to match that of C++ model'''
        self.DCstim = h.IClamp(self.dend(0.5)) 
        self.DCstim.dur = 1e9
        self.DCstim.amp = 0.1091*np.pi*self.dend.diam*self.dend.L*1e-8*1e3  # nano-Amps; conversion from uA/cm2 (x) to nA: x*np.pi*soma.diam*soma.L*1e-8*1e3 
        self.DCstim.delay = 0
        
    def createSynapses(self):    
        #we have found that for all synapses onto inhibitory cells, synaptic 
        # strengths in my model should be 75% the strength in Krishnan's model
        #(due to the difference between our full model and their reduced model)
        syn = h.AMPA(self.dend(0.5)) #TC->IN synapse
        syn.gmax = config.tc2inh_ampa_str 
        self.synlist.append(syn)
        
        syn = h.AMPA_D2(self.dend(0.5)) #for PY->INH connections
        syn.gmax = config.pyr2inh_ampa_d2_str 
        syn.gid = self.gid
        syn.syn_index = 0 #set this to zero (doesn't really matter what value this is set to, since there is only one synapse on INH cells that uses Random123)
        syn.setrand(syn.gid, syn.syn_index) #need to set this here, rather than in INITIAL block of mod file, bc. need to make sure that random number generator is set before gen_nextpsp is called in NET_RECEIVE INITIAL block
        if config.pyr2inh_ampa_d2_str > 0: #prevent division by zero
            syn.psp_weight = config.pyr2inh_ampa_d2_mini_str/config.pyr2inh_ampa_d2_str #see note on 'psp_weight' for pyramidal cell AMPA_D2 synapse
        else:
            syn.psp_weight = 0 #if regular pyr2inh_ampa_d2 connections have strength of zero, then stochastic stimulation of these synapses will also be absent
        h.SS_denom_AMPA_D2 = 250.0 #C++ uses 250.0 ms (need to use "h.*" bc. this is a GLOBAL variable in mod file)
        h.mini_fre_AMPA_D2 = config.pyr2inh_ampa_d2_mini_f
        self.synlist.append(syn)
        
        syn = h.NMDA_D1(self.dend(0.5)) #PY->INH synapse
        syn.gmax = config.pyr2inh_nmda_d1_str 
        h.Use_NMDA_D1 = config.pyr2inh_nmda_d1_Use 
        self.synlist.append(syn)
        
    def createInDegVars(self):
        self.k_TC_IN = 0
        self.k_PY_IN_AMPA = 0
        self.k_PY_IN_NMDA = 0

class RECell(Cell):    
      
    def createSections(self):
        self.soma=h.Section(name='soma', cell=self) #create object 'soma,' and make object variable 'name' equal to 'soma'
    
    def defineGeometry(self):
        self.soma.diam=np.sqrt(1.43e4/np.pi) # one-compartment of 1.43e4 um2
        self.soma.L=self.soma.diam
        self.soma.nseg=1
        self.all=self.soma.wholetree()
    
    def defineBiophysics(self):
        self.soma.cm=1 #Bazhenov 2002
        self.soma.Ra=100
        
        self.soma.insert('pas')		# leak current 
        self.soma.e_pas = -77   # CellSyn.h line 175
        self.soma.g_pas = 5e-5  
        
        self.soma.insert('kL')
        h.krev_kL = -95  # currents.cpp line 103
        self.soma.gkL_kL = config.init_gkl_RE # S/cm2; see C++ CellSyn.h line 177
        
        self.soma.insert('naf_re')		# Hodgin-Huxley INa and IK 
        self.soma.ena = 50 
        self.soma.gnabar_naf_re = 0.100 
        
        self.soma.insert('kdr_re')
        self.soma.ek = -95 
        self.soma.gkbar_kdr_re = 0.01 
        
        self.soma.insert('it_re') 		# reticular IT current  
        self.soma.cao = 2.0               #millimolar; see currents.cpp, line 86
        self.soma.gcabar_it_re = 0.0022	# CellSyn.h, line 176
        
        self.soma.insert('cad')		# calcium decay
        self.soma.depth_cad = 1      #currents.h line 339
        self.soma.taur_cad = 5       #currents.h line 339
        self.soma.cainf_cad = 2.4e-4 #currents.cpp line 133

        h.ion_style("ca_ion",3,1,0,0,0,sec=self.soma) #this ensures that cai is treated as a dynamical variable, while eca is a fixed parameter (although iT_RE.mod implements its own calcium reversal potential)
            
    def createSynapses(self):           
        syn = h.AMPA(self.soma(0.5)) #this is the TC->RE connection
        syn.gmax = config.tc2re_ampa_str #microSiemens (not uS/cm2, just uS)
        self.synlist.append(syn)
        
        syn = h.AMPA(self.soma(0.5)) #this is the PY->RE connection
        syn.gmax = config.pyr2re_ampa_str #microSiemens (not uS/cm2, just uS)
        self.synlist.append(syn)
               
        syn = h.GABA_A(self.soma(0.5)) #RE->RE synapse
        syn.gmax = config.re2re_gaba_a_str #microSiemens
        self.synlist.append(syn)
        
    def createInDegVars(self):
        self.k_TC_RE = 0
        self.k_PY_RE = 0
        self.k_RE_RE = 0
        
class TCCell(Cell):
    def createSections(self):
        self.soma=h.Section(name='soma', cell=self) #create object 'soma,' and make object variable 'name' equal to 'soma'
        
    def defineGeometry(self):
        self.soma.diam=np.sqrt(2.9e4/np.pi) # one compartment of 29,000 um2 
        self.soma.L=self.soma.diam
        self.soma.nseg=1
        self.all=self.soma.wholetree()
        
    def defineBiophysics(self): 
        self.soma.cm=1 
        self.soma.Ra=100
        
        self.soma.insert('pas')		# leak current 
        self.soma.e_pas = -70		# currents.h line 1158
        self.soma.g_pas = 1e-5      # currents.cpp line 1039
        
        self.soma.insert('kL')
        h.krev_kL = -95 # currents.cpp line 103
        self.soma.gkL_kL = config.init_gkl_TC # S/cm2; CellSyn.h line 241
        
        self.soma.insert('naf_tc')		# Hodgin-Huxley INa and IK 
        self.soma.ena = 50 
        self.soma.gnabar_naf_tc = 0.090 
        
        self.soma.insert('kdr_tc')
        self.soma.ek = -95 
        self.soma.gkbar_kdr_tc = 0.012 # CellSyn.h line 238
        
        self.soma.insert('it_tc')		# T-current 
        self.soma.cai = 2.4e-4 
        self.soma.cao = 2 
        self.soma.eca = 120 
        self.soma.qm_it_tc = 3.55		# currents.cpp line 144
        self.soma.qh_it_tc = 3.0          # currents.cpp line 144
        self.soma.gcabar_it_tc = 0.0025	# CellSyn.h line 231
        
        self.soma.insert('cad')		# calcium decay
        self.soma.depth_cad = 2.0    # CellSyn.h line 233
        self.soma.taur_cad = 5     # currents.h line 339
        self.soma.cainf_cad = 2.4e-4  #currents.cpp line 133
        
        self.soma.insert('iar')		# h-current
        self.soma.eh = -40		# currents.cpp line 165
        self.soma.ghbar_iar = 1.6e-5	# CellSyn.h line 239
        h.cac_iar = 0.0015		# half-activation of Ca++ binding; see currents.h line 376
        h.k2_iar = 0.0004		# decay of Ca++ binding on protein (see currents.cpp line 167)
        h.Pc_iar = 0.007		# half-activation of binding on Ih channel (see CellSyn.h line 234)
        h.k4_iar = 0.001		# decay of protein binding on Ih channel (see CellSyn.h line 235)
        h.nca_iar = 4		# "h." rather than "soma." because this is a GLOBAL variable in the mod file; nb of binding sites for Ca++ on protein (see currents.cpp line 168)
        h.nexp_iar = 1		# nb of binding sites on Ih channel(see currents.cpp line 168)
        h.ginc_iar = 2.0		# augm of conductance of bound Ih (see CellSyn.h line 229)
        h.taum_iar = 20.0          # see currents.cpp line 168
        self.soma.fac_gh_TC_iar = config.init_gh_TC 
        
        h.ion_style("ca_ion",3,1,0,0,0,sec=self.soma) #this ensures that cai is treated as a dynamical variable, while eca is a fixed parameter (although iT_TC.mod implements its own calcium reversal potential)
        
    def createSynapses(self):
        syn = h.GABA_A(self.soma(0.5)) #RE->TC GABA_A synapse
        syn.gmax = config.re2tc_gaba_a_str #uS
        syn.Erev = -83.0 #see line 262 of CellSyn.h
        self.synlist.append(syn)
        
        syn = h.GABA_B(self.soma(0.5)) #RE->TC GABA_B synapse
        syn.gmax = config.re2tc_gaba_b_str # NOTE: all changes to GABA_B syn strength should be made by modifying gmax, NOT netcon.weight[0]. This is because GABA_B has a nonlinearity, so doubling the weight does not double gmax
        syn.Erev = -95
        self.synlist.append(syn)
        
        syn = h.AMPA(self.soma(0.5)) #this is the PY->TC connection
        syn.gmax = config.pyr2tc_ampa_str #microSiemens (not uS/cm2, just uS)
        self.synlist.append(syn)
        
    def createInDegVars(self):
        self.k_RE_TC_GABA_A = 0
        self.k_RE_TC_GABA_B = 0
        self.k_PY_TC = 0
        
