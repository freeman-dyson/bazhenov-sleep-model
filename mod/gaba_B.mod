: $Id: gabab.mod,v 1.9 2004/06/17 16:04:05 billl Exp $

COMMENT
This is a modified version of the mod file found here:
https://senselab.med.yale.edu/modeldb/showmodel.cshtml?model=150538&file=%2fxietal2013%2fgababsyn.mod#tabs-2
Modifications by Christian Fink
-----------------------------------------------------------------------------

	Kinetic model of GABA-B receptors
	=================================

  MODEL OF SECOND-ORDER G-PROTEIN TRANSDUCTION AND FAST K+ OPENING
  WITH COOPERATIVITY OF G-PROTEIN BINDING TO K+ CHANNEL

  PULSE OF TRANSMITTER

  SIMPLE KINETICS WITH NO DESENSITIZATION

	Features:

  	  - peak at 100 ms; time course fit to Tom Otis' PSC
	  - SUMMATION (psc is much stronger with bursts)


	Approximations:

	  - single binding site on receptor	
	  - model of alpha G-protein activation (direct) of K+ channel
	  - G-protein dynamics is second-order; simplified as follows:
		- saturating receptor
		- no desensitization
		- Michaelis-Menten of receptor for G-protein production
		- "resting" G-protein is in excess
		- Quasi-stat of intermediate enzymatic forms
	  - binding on K+ channel is fast


	Kinetic Equations:

	  dR/dt = K1 * T * (1-R-D) - K2 * R

	  dG/dt = K3 * R - K4 * G

	  R : activated receptor
	  T : transmitter
	  G : activated G-protein
	  K1,K2,K3,K4 = kinetic rate cst

  n activated G-protein bind to a K+ channel:

	n G + C <-> O		(Alpha,Beta)

  If the binding is fast, the fraction of open channels is given by:

	O = G^n / ( G^n + KD )

  where KD = Beta / Alpha is the dissociation constant

-----------------------------------------------------------------------------

  Parameters estimated from patch clamp recordings of GABAB PSP's in
  rat hippocampal slices (Otis et al, J. Physiol. 463: 391-407, 1993).

-----------------------------------------------------------------------------

  PULSE MECHANISM

  Kinetic synapse with release mechanism as a pulse.  

  Warning: for this mechanism to be equivalent to the model with diffusion 
  of transmitter, small pulses must be used...

  For a detailed model of GABAB:

  Destexhe, A. and Sejnowski, T.J.  G-protein activation kinetics and
  spill-over of GABA may account for differences between inhibitory responses
  in the hippocampus and thalamus.  Proc. Natl. Acad. Sci. USA  92:
  9515-9519, 1995.

  For a review of models of synaptic currents:

  Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  Kinetic models of 
  synaptic transmission.  In: Methods in Neuronal Modeling (2nd edition; 
  edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1996.

  This simplified model was introduced in:

  Destexhe, A., Bal, T., McCormick, D.A. and Sejnowski, T.J.
  Ionic mechanisms underlying synchronized oscillations and propagating
  waves in a model of ferret thalamic slices. Journal of Neurophysiology
  76: 2049-2070, 1996.  

  See also http://www.cnl.salk.edu/~alain



  Alain Destexhe, Salk Institute and Laval University, 1995

-----------------------------------------------------------------------------
ENDCOMMENT

NEURON {
	POINT_PROCESS GABA_B
	RANGE R, G, g, gmax, Erev
	NONSPECIFIC_CURRENT i
	GLOBAL Cmax, Cdur, deadtime
	GLOBAL K1, K2, K3, K4, KD
}
UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
	(mM) = (milli/liter)
}

PARAMETER {
    gmax = 0.0001 (umho)
	Cmax	= 0.5	(mM)		: max transmitter concentration
	Cdur	= 0.3	(ms)		: transmitter duration (rising phase)
	deadtime=1.0 (ms) : minimum time between release events
:
:	From Kfit with long pulse (5ms 0.5mM)
:   see Krishnan's CellSyn.h (and around line 1480 of 2002 C++ code) for 'Kx' values
	K1	= 0.52	(/ms mM)	: forward binding rate to receptor (Krishnan currents.h line 665)
	K2	= 0.0013 (/ms)		: backward (unbinding) rate of receptor (Krishnan currents.h line 666)
	K3	= 0.098 (/ms)		: rate of G-protein production (Krishnan currents.h line 667)
	K4	= 0.033 (/ms)		: rate of G-protein decay (Krishnan currents.h line 668)
	KD	= 100			: dissociation constant of K+ channel  (Krishnan currents.cpp line 348)
	n	= 4			: nb of binding sites of G-protein on K+ (Krishnan currents.cpp line 348)
	Erev	= -95	(mV)		: reversal potential (E_K) (Krishnan currents.cpp line 346)
}


ASSIGNED {
	v		(mV)		: postsynaptic voltage
	i 		(nA)		: current = g*(v - Erev)
	g 		(umho)		: conductance
	Gn
	R				: fraction of activated receptor
	synon
	Rinf
	Rtau (ms)
	Beta (/ms)
}


STATE {
	Ron Roff
	G				: fraction of activated G-protein
}


INITIAL {
	R = 0
	G = 0
	synon = 0
	Rinf = K1*Cmax/(K1*Cmax + K2)
	Rtau = 1/(K1*Cmax + K2)
	Beta = K2
}

BREAKPOINT {
	SOLVE bindkin METHOD cnexp
	Gn = G*G*G*G : ^n = 4
	g = gmax * Gn / (Gn+KD)
	i = g*(v - Erev)
}


DERIVATIVE bindkin {
	Ron' = synon*K1*Cmax - (K1*Cmax + K2)*Ron
	Roff' = -K2*Roff
	R = Ron + Roff
	G' = K3 * R - K4 * G
}

: following supports both saturation from single input and
: summation from multiple inputs
: Note: automatic initialization of all reference args to 0
: except first

:Fink note: I replaced the NET_RECEIVE block of the original file
:with modifications simliar to those from Section 10.1.7 of The NEURON Book. 
:Differences from 10.1.7 are that a deadtime is included, and presynaptic spikes that occur
:during an 'on' state or deadtime are simply ignored (though for Cdur of 0.3ms and deadtime of 1.0 ms,
:this should never happen)
:also, netcon weight is assumed to be 1.0; changes in synaptic strength should be implemented by changing 
:gmax, since the nonlinearity in the equations results in a nonlinear relationship between 'weight' and 'gmax' 

NET_RECEIVE(weight, r0, t0 (ms), lastspike (ms)) {
	INITIAL{
		r0 = 0
		t0 = 0 (ms)
		lastspike = -10*(Cdur + deadtime) :this statement must be here, and not in other INITIAL block, in order to prevent segmentation fault
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

COMMENT
	:CF: this is the original NET_RECEIVE block from https://senselab.med.yale.edu/modeldb/showmodel.cshtml?model=150538&file=%2fxietal2013%2fgababsyn.mod#tabs-2
	if (flag == 1) { : at end of Cdur pulse so turn off
		r0 = weight*(Rinf + (r0 - Rinf)*exp(-(t - t0)/Rtau))
		t0 = t
		synon = synon - weight
		Ron = Ron-r0
		Roff = Roff+r0
        }else{ : at beginning of Cdur pulse so turn on
		r0 = weight*r0*exp(-Beta*(t - t0)) :CF: the factor of weight here does not make sense; it should only be applied to the exponential rise
		t0 = t
		synon = synon + weight C
		Ron = Ron+r0
		Roff = Roff-r0
		:come again in Cdur
		net_send(Cdur, 1)
        }
ENDCOMMENT
}






