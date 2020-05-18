TITLE kca.mod
 
COMMENT
 Calcium-dependent potassium current for pyramidal cell and interneurons defined in
 Timofeev et. al. 2000 (Cerebral Cortex) and Bazhenov et. al. 2002 (J Neuro) and Chen et. al. 2012 (J. Physiol.)
 This code is adapted from hh.mod distributed with NEURON source code
ENDCOMMENT
 
UNITS {
	(molar) = (1/liter)			: moles do not appear in units
	(mM)	= (millimolar)
        (mA) = (milliamp)
        (mV) = (millivolt)
	(S) = (siemens)
}
 
? interface
NEURON {
        SUFFIX kca
        USEION k READ ek WRITE ik
        USEION ca READ cai
        RANGE gkcabar
        RANGE minf, mtau
}
 
PARAMETER {
        gkcabar = .0003 (S/cm2)	<0,1e9>
        ek = -95 (mV)
}
 
STATE {
        m
}
 
ASSIGNED {
        v (mV)
        :celsius (degC)
		gkca (S/cm2)
        ik (mA/cm2)
        cai (mM)
        minf
        mtau (ms)
}
 
? currents
BREAKPOINT {
        SOLVE states METHOD cnexp
        gkca = gkcabar*m
		ik = 2.952882641412121*gkca*(v - ek) :Krishnan currents.h line 402 prescribes 2.3^((36-23)/10)
}
 
 
INITIAL {
	rates(cai)
	m = minf
}

? states
DERIVATIVE states {  
        rates(cai)
        m' =  (minf-m)/mtau
}
 
? rates
PROCEDURE rates(cai(mM)) {  :Computes rate and other constants at current cai.
                      :Call once from HOC to initialize inf at resting cai
UNITSOFF
                :"m" potassium activation system
        minf = cai/(cai+2) :this is mathematically equivalent to expression given in Chen 2012 (and Krishnan for minf
        mtau = 33.86521313023745/(cai+2) :numerator is 100/2.95288264
UNITSON
}



