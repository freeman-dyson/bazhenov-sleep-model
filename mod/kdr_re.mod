TITLE kdr_re.mod
 
COMMENT
 Delayed rectifying potassium current for RE and TC cells defined in appendix of Bazhenov 1998, and referenced in 
 Bazhenov et. al. 2002 (J Neuro) and Chen et. al. 2012 (J. Physiol.)
 In reality, this code apes that from the C++ code accompanying Bazhenov 2002
 This code is adapted from hh.mod distributed with NEURON source code
ENDCOMMENT
 
UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	(S) = (siemens)
}
 
? interface
NEURON {
        SUFFIX kdr_re
        USEION k READ ek WRITE ik
        RANGE gkbar, gk
        RANGE ninf, ntau
	:THREADSAFE : assigned GLOBALs will be per thread
}
 
PARAMETER {
        gkbar = 0.200 (S/cm2)	<0,1e9>
        ek = -95 (mV)
}
 
STATE {
        n
}
 
ASSIGNED {
        v (mV)
        :celsius (degC)

	gk (S/cm2)
        ik (mA/cm2)
        ninf
	ntau (ms)
}
 
? currents
BREAKPOINT {
        SOLVE states METHOD cnexp
        gk = gkbar*n*n*n*n
		ik = gk*(v - ek) 
}
 
 
INITIAL {
	rates(v)
	n = ninf
}

? states
DERIVATIVE states {  
        rates(v)
        n' =  (ninf-n)/ntau
}
 
? rates
PROCEDURE rates(v(mV)) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
        LOCAL  a, b, sum
UNITSOFF
		:"n" potassium gating 
        a = 0.032*vtrap(-(v+35),5) :to match C++ code, for TC cell replace "35" with "13" (see Vtr and Vtrk in Krishnan's CellSyn.h lines 236-237, currents.h lines 313-314)
        b =  0.5*exp(-(v+40)/40) :to match C++ code, for TC cell replace "40" with "18" (but only in numerator, NOT denominator)
        sum = a + b
		ntau = 1.0/sum
        ninf = a/sum
        
}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2) :derived by Taylor expanding exp to quadratic term, then binomial approximation
        }else{
                vtrap = x/(exp(x/y) - 1)
        }
}
 
UNITSON


