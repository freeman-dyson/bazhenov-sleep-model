TITLE ampa_D2.mod

COMMENT
This model adds random, spontaneous EPSP's to ampa_D1.mod (which already includes depression), 
as in the model used in Krishnan et. al. 2016 (eLife) (and similar to Bazhenov 2002). See line 490 in C++ currents.cpp.
First-order synaptic dynamics originally proposed in "An efficient method for computing synaptic
conductances based on a kinetic model of receptor binding" (Destexhe et. al., 1994).
This is an updated version of a mod file originally by Alain Destexhe, ModelDB #18198.
This updated version is based primarily on Section 10.1.7 from The NEURON Book, 
revised to include the parameters Cdur and gmax, as well as a "deadtime"
--adapted by Christian G. Fink, Gonzaga University--
ENDCOMMENT

NEURON{
	POINT_PROCESS AMPA_D2
	NONSPECIFIC_CURRENT i
	POINTER ptr
	GLOBAL Rinf, Rtau, Cdur, Alpha, Beta, mini_fre, SS_denom  :values of global variables are the same within a mechanism, but not across mechanisms (e.g., AMPA's deadtime may have a different value than AMPA_D1's deadtime)
	RANGE g, gmax,  Erev, gid, syn_index, psp_weight
}

UNITS{
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
	(mM) = (milli/liter)
	(uS) = (micromho)
}

PARAMETER{ :see line 447 of currents.cpp for parameter values
	gmax   = 0.0001 (uS) :max conductance of *one* synapse (so in BREAKPOINT, g can be greater than this if there are multiple incoming connections)
	psp_weight = 1.5 :unitless factor which scales gmax for the stochastic PSP's prescribed in Bazhenov 2002 and Krishnan 2016
	Cdur   = 0.3  (ms) :transmitter duration (rising phase)
	deadtime=1.0 (ms) : minimum time between release events; this code assumes Cdur+deadtime<100 ms, because PSP's must happen at least 100 ms after last spike, which is assumed not to be during a deadtime
	Cmax   = 0.5	(mM)		: max transmitter concentration
	Alpha  = 1.1  (/ms mM):forward (binding) rate (see currents.cpp line 493)
	Beta   = 0.19 (/ms):backward (dissociation) rate (see currents.cpp line 493)
	Erev   = 0    (mV) :equilibrium potential
	Tr     = 700 (ms) :time constant for short-term depression
	afterspike_time= 100 (ms) :EPSP's can only occur a minimum of 'afterspike_time' after most recent presynaptic spike
	mini_fre=20  (ms) :parameter for generating epsp's (this is the same variable name as in currents.cpp in C++ code)
	SS_denom=250 (ms) :another parameter for generating epsp's (this is in the denominator of 'SS' in currents.cpp)
	gid = 0 :need to remember to change this when the synapse is created in NEURON
	syn_index = 0 :may change this when the synapse is created in NEURON; this is used by the random number generator to generate different streams for different synapses on the same cell; so if a cell has more than one synapse which uses Random123, this should be changed
}

ASSIGNED {
	v    (mV)   : postsynaptic voltage
	i    (nA)   : current = g*(v-Erev)
	g    (umho) : conductance
	Rtau (ms)   : time constant of channel building
	Rinf        :fraction of open channels if xmtr is present "forever"
	synon       :sum of weights of all synapses in the "onset" state (where weight is assumed to be a unitless factor which scales gmax)
	ptr
}

STATE { Ron Roff }  :initialized to 0 by default
: Ron and Roff are the total conductances of all synapses
: that are in the "onset" (transmitter pulse ON)
: and "offset" (transmitter pulse OFF) states, respectively
:declared without units, so units are specified in BREAKPOINT block

INITIAL {
	:Ron and Roff default to being initialized to zero
	synon = 0
	Rtau = 1 / ((Alpha * Cmax) + Beta)
	Rinf = Cmax*Alpha / (Cmax*Alpha + Beta)
	:setrand(gid,syn_index) :this is now set from NEURON (see createSynapses methods in cell_classes.py)
}

BREAKPOINT { 
	SOLVE release METHOD cnexp
	g = gmax * (Ron + Roff) :max value is weight*gmax*synon*Rinf (for spikes; random PSP's carry psp_weight instead of weight)
	i = g*(v - Erev)
}

DERIVATIVE release {
	Ron'  = (synon*Rinf - Ron)/Rtau
	Roff' = -Beta*Roff
}

:weight is assumed to be a unitless factor which scales gmax for presynaptic spikes
:short-term depression achieved using the factor 'E,' which is proportion of available presynaptic resources. 

NET_RECEIVE(weight, r0, toff (ms), lastspike (ms), lastpsp (ms), nextpsp (ms), E, gid) {
	INITIAL{
		net_send(gen_nextpsp(-100*Rtau,mini_fre,SS_denom),1) :put a possible PSP event in the queue, using a 'lastspike' value of -100*Rtau (very long time ago); for some reason, placing this statement in the mod file's INITIAL block caused a segmentation fault
		r0 = 0
		toff = 0 (ms) :this value doesn't matter
		lastspike = -100*Rtau :initialize to large negative val so that do not get depression on first spike
		lastpsp = -100*Rtau :initialize to large negative val so an early spike will trigger neurotransmitter release
		nextpsp = 0.0 (ms) :time until the next PSP *might* be generated (as long as the last spike occurred more than 99.99 ms ago); this value doesn't matter bc. we initiate PSP event in INITIAL block
		E  = 1 :synapse starts out at full strength
		:printf("presyn gid = %d, postsyn gid = %d \n", gid, src_gid) :useful for debugging
	}
	
	:flag is an implicit argument of NET_RECEIVE, normally 0
	if (flag == 0){ :flag==0 implies a spike is received 
		:a spike arrived; ignore it if we are already within either a spike or a psp on state, or deadtime
		:printf("Entered flag=0: t=%8.2f ms \n", t) :useful for debugging
		if((t-lastspike)>(Cdur + deadtime) && (t-lastpsp)>(Cdur+deadtime)){
			:for 'E,' see "Synaptic Currents" section of Bazhenov 2002 (and around roughly line 500 of C++ currents.cpp)
			E = 1 - (1 - E*(1-0.07)) * exp(-(t-lastspike)/Tr) :note that we are assuming the last spike in this stream occurred at t0-Cdur
			synon = synon + E*weight :weight is scaled by 'E' to implement synaptic depression
			r0 = r0*exp(-Beta*(t-toff)) :r0 at start of onset state
			Ron = Ron + r0
			Roff = Roff - r0
			lastspike = t 
			:printf("New lastspike time: %8.2f ms \n",lastspike)
			net_send(Cdur, 2) :turn off neurotransmitter in Cdur with flag = 2
		}
	}
	if (flag == 1) { :possible PSP event, as long as it is sufficiently long after spike and previous psp
		if( (t-lastpsp)<=(Cdur+deadtime) ){
			:printf("Flag=1, and (t-lastpsp)<=(Cdur+deadtime) \n")
		}
		if((t-lastspike)>(afterspike_time-0.00001) && (t-lastspike)>(nextpsp-0.00001)) { :"-0.00001" to avoid round-off issues; second clause bc. Krishnan et. al. require that last spike or psp occur at least 'nextpsp' before next psp (and by design, lastpsp occurred at least 'nextpsp' ago); third clause prevents two PSP events from overlapping
			synon = synon + psp_weight
			r0 = r0*exp(-Beta*(t-toff)) 
			Ron = Ron + r0
			Roff = Roff - r0
			lastpsp = t
			net_send(Cdur, 2) :turn off neurotransmitter
			
			nextpsp=0 :generate next epsp time (analogous to 'newrelease' in C++ code), and make sure it does not come before the end of Cdur+deadtime
			while(nextpsp < (Cdur+deadtime) ){
				nextpsp = gen_nextpsp(lastspike,mini_fre,SS_denom) 
			}
			:printf("Flag==1: New next psp: t=%8.2f, nextpsp=%8.2f, lastspike=%8.2f \n",t, nextpsp,lastspike)
			net_send(max(afterspike_time-(t-lastspike),nextpsp),1) :C++ code requires that PSP's occur at least 100 ms after most recent spike; this is earliest time that a PSP could be allowed to happen, so send an event to check at that time
		} else {
			net_send( max(afterspike_time,nextpsp)-(t-lastspike) , 1) :at this point, we know for sure that lastpsp occurred more than 'nextpsp' ago (bc. when the last psp occurred, we scheduled the next psp to occur 'nextpsp' later), so the next psp must occur more than 100 ms after lastspike or 'nextpsp' ms after lastspike (whichever is greater)
			:printf("Regenerated psp: t=%8.2f, nextpsp=%8.2f, lastspike=%8.2f \n",t,nextpsp,lastspike)
		}
	}
	if (flag == 2) {
		: "turn off transmitter," i.e. this synapse enters the offset state
		:printf("Entered flag==2: t=%8.2f ms \n",t)
		if(lastspike>lastpsp) { :need to consider that spikes carry different weights from epsp's
			synon = synon - E*weight :note we need to include 'E' in order to undo the addition to synon in the above block
			: r0 at start of offset state
			r0 = E*weight*Rinf + (r0-E*weight*Rinf)*exp(-(t-lastspike)/Rtau)
		} else {
			synon = synon - psp_weight
			r0 = psp_weight*Rinf + (r0-psp_weight*Rinf)*exp(-(t-lastpsp)/Rtau)
		}
		Ron = Ron - r0
		Roff = Roff + r0
		:printf("Flag=2: Ron=%8.2f, Roff=%8.2f \n",Ron,Roff)
		toff = t :update time of most recent offset
	}
}

FUNCTION max(x(ms),y(ms)) {
	if(x>y) {
		max = x
	} else {
		max = y
	}
}

:this code uses Random123, which requires NEURON 7.3 or higher
:uses nrnran123.c and nrnran123.h from http://www.neuron.yale.edu/hg/neuron/nrn/file/9d4ab20927bc/src/oc/
VERBATIM
#define VOIDCAST nrnran123_State** vp = (nrnran123_State**)(&(_p_ptr))
//#define VOIDCAST void** vp = (void**)(&(_p_ptr)) these revisions made in order to successfully compile in NEURON 8.2 (see https://www.neuron.yale.edu/phpBB/viewtopic.php?p=20009#p20009)
//extern void * nrnran123_newstream(int,int); 
//extern void nrnran123_deletestream(void *);
//extern double nrnran123_dblpick(void *);
ENDVERBATIM

PROCEDURE setrand(id1,id2) {
	VERBATIM
	VOIDCAST;
	if(*vp) {
		nrnran123_deletestream(*vp);
	} 
	*vp = nrnran123_newstream((uint32_t) _lid1,(uint32_t) _lid2);
	//*vp = nrnran123_newstream((int) _lid1,(int) _lid2); again, see https://www.neuron.yale.edu/phpBB/viewtopic.php?p=20009#p20009
	ENDVERBATIM
} 

FUNCTION pick() {
	VERBATIM
	VOIDCAST;
	_lpick = nrnran123_dblpick(*vp);
	ENDVERBATIM
}

FUNCTION gen_nextpsp(lastspike (ms),mini_fre (ms),SS_denom(ms)) { :adapted from http://www.neuron.yale.edu/hg/neuron/nrn/file/9d4ab20927bc/src/gnu/NegExp.cpp
	LOCAL S, SS
	SS = (2.0/(1.0+exp(-(t-lastspike)/mini_fre))-1.0)/SS_denom :this is essentially half of a sigmoid function, starting from 0 at t=0 and saturating at 1 as t->\infty
	S = pick()
	if(S < 0.000001){ :just following Krishnan here
		S = 0.000001
	}
	gen_nextpsp = -(log(S))/SS :see AMPA_D2 class in currents.cpp
}
