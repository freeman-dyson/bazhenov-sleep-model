TITLE Low threshold calcium current for TC cells
:
:   Ca++ current responsible for low threshold spikes (LTS)
:   RETICULAR THALAMUS
:   Differential equations
:
:   Model of Huguenard & McCormick, J Neurophysiol 68: 1373-1383, 1992.
:   The kinetics is described by standard equations (NOT GHK)
:   using a m2h format, according to the voltage-clamp data
:   (whole cell patch clamp) of Huguenard & Prince, J Neurosci.
:   12: 3804-3817, 1992.
:
:    - Kinetics adapted to fit the T-channel of reticular neuron
:    - Time constant tau_h refitted from experimental data
:    - shift parameter for screening charge
:
:   Model described in detail in:   
:     Destexhe, A., Contreras, D., Steriade, M., Sejnowski, T.J. and
:     Huguenard, J.R.  In vivo, in vitro and computational analysis of
:     dendritic calcium currents in thalamic reticular neurons.
:     Journal of Neuroscience 16: 169-185, 1996.
:   See also:
:     http://www.cnl.salk.edu/~alain
:     http://cns.fmed.ulaval.ca
:
:   Written by Alain Destexhe, Salk Institute, Sept 18, 1992
:   Adapted by Chris Fink, for replicating results from Bazhenov 2002 (6/3/19)

NEURON {
	SUFFIX it_tc
	USEION ca READ cai, cao WRITE ica
	RANGE gcabar, m_inf, tau_m, h_inf, tau_h, shift, qm, qh
}

UNITS {
	(molar) = (1/liter)
	(mV) =	(millivolt)
	(mA) =	(milliamp)
	(mM) =	(millimolar)

	FARADAY = (faraday) (coulomb)
	R = (k-mole) (joule/degC)
}

PARAMETER {
	v		(mV)
:	celsius	= 36	(degC) CF: I am eliminating all references to the temperature in this mod file, because I do not trust NEURON to set the temperature correctly (should be 36 C)
:	eca	= 120	(mV)
	gcabar	= .0022	(mho/cm2) :Bazhenov C++ code line 1495
	cai	= 2.4e-4 (mM)		: adjusted for eca=120 mV
	cao	= 2	(mM) :Bazhenov C++ code line 240
	qm	= 3.55 :Krishnan currents.cpp line 144
	qh 	= 3.0  :Krishnan currents.cpp line 144
}

STATE {
	m h
}

ASSIGNED {
	ica	(mA/cm2)
	carev	(mV)
	m_inf
	tau_m	(ms)
	h_inf
	tau_h	(ms)
	phi_m
	phi_h
}

BREAKPOINT {
	SOLVE castate METHOD cnexp
	:carev = (1e3) * (R*(celsius+273.15))/(2*FARADAY) * log (cao/cai)
	carev = (1e3) * (R*309.15)/(2*FARADAY) * log (cao/cai)
	ica = gcabar * m*m*h * (v-carev)
}

DERIVATIVE castate {
	evaluate_fct(v)

	m' = (m_inf - m) / tau_m
	h' = (h_inf - h) / tau_h
}

UNITSOFF
INITIAL {
:
:   Activation functions and kinetics were obtained from
:   Huguenard & Prince, and were at 23-25 deg.
:   Transformation to 36 deg using Q10
:
	phi_m = 4.57376686268585 :qm ^ ((celsius-24)/10)
	phi_h = 3.7371928188465517 :qh ^ ((celsius-24)/10)

	evaluate_fct(v)
	m = m_inf
	h = h_inf
}

PROCEDURE evaluate_fct(v(mV)) { 
:
:   Time constants were obtained from J. Huguenard
:

	m_inf = 1.0 / ( 1 + exp(-(v+59)/6.2) )
	h_inf = 1.0 / ( 1 + exp((v+83)/4.0) )

	tau_m = (1/(exp(-(v+131.6)/16.7)+exp((v+16.8)/18.2)) + 0.612) / phi_m
	tau_h = (30.8 + (211.4 + exp((v + 115.2)/5))/(1+exp((v + 86)/3.2))) / phi_h
}
UNITSON
