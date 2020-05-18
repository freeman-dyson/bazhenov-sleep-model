TITLE ampa.mod

COMMENT
First-order synaptic dynamics originally proposed in "An efficient method for computing synaptic
conductances based on a kinetic model of receptor binding" (Destexhe et. al., 1994).
This is an updated version of a mod file originally by Alain Destexhe, ModelDB #18198.

This updated version is based primarily on Section 10.1.7 from The NEURON Book. It is
revised (from ampa_NEURON.mod) to include the parameters Cdur and gmax, as well as to include a "deadtime," as in 
Destexhe's original model. 
ENDCOMMENT

NEURON{
	POINT_PROCESS AMPA
	NONSPECIFIC_CURRENT i
	GLOBAL deadtime, Cdur, Alpha, Beta, Rinf, Rtau :values of global variables are the same within a mechanism, but not across mechanisms (e.g., AMPA's deadtime may have a different value than AMPA_D1's deadtime)
	RANGE g, gmax, Erev
}

UNITS{
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
	(mM) = (milli/liter)
	(uS) = (micromho)
}

PARAMETER{ :see line 390 of currents.cpp (from Giri Krishnan) for parameter values
	gmax   = 0.0001 (uS) :max conductance of *one* synapse (so in BREAKPOINT, g can be greater than this if there are multiple incoming connections)
	Cdur   = 0.3  (ms) :transmitter duration (rising phase)
	deadtime=1.0 (ms) : minimum time between release events
	Cmax   = 0.5	(mM)		: max transmitter concentration
	Alpha  = 1.1  (/ms mM):forward (binding) rate (Krishnan currents.cpp line 394)
	Beta   = 0.19 (/ms):backward (dissociation) rate (Krishnan currents.cpp line 394)
	Erev   = 0    (mV) :equilibrium potential
}

ASSIGNED {
	v    (mV)   : postsynaptic voltage
	i    (nA)   : current = g*(v-Erev)
	g    (umho) : conductance
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
	:g = (Ron + Roff)*1(umho)
	g = gmax * (Ron + Roff) :max value is gmax*synon*Rinf
	i = g*(v - Erev)
}

DERIVATIVE release {
	Ron'  = (synon*Rinf - Ron)/Rtau
	Roff' = -Beta*Roff
}

:weight is assumed to be a unitless factor which scales gmax
NET_RECEIVE(weight, r0, t0 (ms), lastspike (ms)) {
	INITIAL{
		r0 = 0
		t0 = 0 (ms)
		lastspike = -10*(Cdur + deadtime)
	}
	:flag is an implicit argument of NET_RECEIVE, normally 0
	if (flag == 0){ :flag==0 implies a spike is received
		:a spike arrived; ignore it if we are already within either a spike state, or deadtime
		if( (t-lastspike)>(Cdur + deadtime) ){
			synon = synon + weight
			r0 = r0*exp(-Beta*(t-t0)) :r0 at start of onset state
			Ron = Ron + r0
			Roff = Roff - r0
			t0 = t
			lastspike = t
			:come again in Cdur with flag = 1
			net_send(Cdur, 1)
		}
	}
	if (flag == 1) {
		: "turn off transmitter"
		: i.e. this synapse enters the offset state
		synon = synon - weight
		: r0 at start of offset state
		r0 = weight*Rinf + (r0-weight*Rinf)*exp(-(t-t0)/Rtau)
		Ron = Ron - r0
		Roff = Roff + r0
		t0 = t
	}
}
