TITLE nmda_D1.mod

COMMENT
This model is a revised version of ampa_D1.mod, which added synaptic depression to ampa.mod, 
as in the model used in Krishnan 2016 (eLife)
First-order synaptic dynamics originally proposed in "An efficient method for computing synaptic
conductances based on a kinetic model of receptor binding" (Destexhe et. al., 1994).
This is an updated version of a mod file originally by Alain Destexhe, ModelDB #18198.
This updated version is based primarily on Section 10.1.7 from The NEURON Book, 
revised to include the parameters Cdur and gmax, as well as a "deadtime"
ENDCOMMENT

NEURON{
	POINT_PROCESS NMDA_D1
	NONSPECIFIC_CURRENT i
	GLOBAL deadtime, Cdur, Alpha, Beta,  Rinf, Rtau, Use :values of global variables are the same within a mechanism, but not across mechanisms (e.g., AMPA's deadtime may have a different value than AMPA_D1's deadtime)
	RANGE g, gmax, Erev
}

UNITS{
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
	(mM) = (milli/liter)
	(uS) = (micromho)
}

PARAMETER{ :see line 447 of currents.cpp (from Giri Krishnan) for parameter values
	gmax   = 0.0001 (uS) :max conductance of *one* synapse (so in BREAKPOINT, g can be greater than this if there are multiple incoming connections)
	Cdur   = 0.3  (ms) :transmitter duration (rising phase)
	deadtime=1.0 (ms)  : minimum time between release events
	Cmax   = 0.5	(mM)		: max transmitter concentration
	Alpha  = 1.0  (/ms mM):forward (binding) rate
	Beta   = 0.0067 (/ms):backward (dissociation) rate
	Erev   = 0    (mV) :equilibrium potential
	Use    = 0.0  :determines how quickly synaptic resources are depleted (the larger this value is, the greater the short-term depressive effect)
	Tr     = 750 (ms) :time constant for short-term depression (see Krishnan's currents.h)
}

ASSIGNED {
	v    (mV)   : postsynaptic voltage
	i    (nA)   : current = g*(v-Erev)
	g    (umho) : conductance
	fn			: for implementing NMDA receptor post-synaptic voltage dependence
	Rtau (ms)   : time constant of channel building
	Rinf        :fraction of open channels if xmtr is present "forever"
	synon       :sum of weights of all synapses in the "onset" state (where weight is assumed to be a unitless factor which scales gmax)
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
}

BREAKPOINT { : would be good to get this in terms of gmax
	SOLVE release METHOD cnexp
	g = gmax * (Ron + Roff) :max value is gmax*synon*Rinf
	fn = 1/(1+exp(-(v + 25 (mV))/12.5 (mV))) :see "Synaptic Currents" section of Bazhenov 2002 (also in Krishnan's currents.cpp)
	i = g*fn*(v - Erev)
}

DERIVATIVE release {
	Ron'  = (synon*Rinf - Ron)/Rtau
	Roff' = -Beta*Roff
}

:weight is assumed to be a unitless factor which scales gmax.
:short-term depression achieved using the factor 'E,' which is proportion 
:of available presynaptic resources. 

NET_RECEIVE(weight, r0, t0 (ms), lastspike (ms), E) {
	INITIAL{
		r0 = 0
		t0 = 0 (ms) :this value doesn't really matter
		lastspike = -100*Rtau :initialize to large neg value so that do not get depression on first spike
		E  = 1 :synapse starts out at full strength
	}
	:flag is an implicit argument of NET_RECEIVE, normally 0
	if (flag == 0){ :flag==0 implies a spike is received 
		:a spike arrived; ignore it if we are already within either a spike state, or deadtime
		if( (t-lastspike)>(Cdur + deadtime) ){
			:for 'E,' see "Synaptic Currents" section of Bazhenov 2002 (and line ~500 of Krishnan's currents.cpp)
			E = 1 - (1 - E*(1-Use)) * exp(-(t-lastspike)/Tr) :note that we are assuming the last spike in this stream occurred at t0-Cdur
			synon = synon + E*weight :weight is scaled by 'E' to implement synaptic depression
			r0 = r0*exp(-Beta*(t-t0)) :r0 at start of onset state
			Ron = Ron + r0
			Roff = Roff - r0
			t0 = t :update time of most recent state change
			lastspike = t :update most recent spike time
			:come again in Cdur with flag = 1
			net_send(Cdur, 1)
		}
	}
	if (flag == 1) {
		: "turn off transmitter"
		: i.e. this synapse enters the offset state
		synon = synon - E*weight :note we need to include 'E' in order to undo the addition to synon in the above block
		: r0 at start of offset state
		r0 = E*weight*Rinf + (r0-E*weight*Rinf)*exp(-(t-t0)/Rtau)
		Ron = Ron - r0
		Roff = Roff + r0
		t0 = t :update time of most recent state change
	}
}
