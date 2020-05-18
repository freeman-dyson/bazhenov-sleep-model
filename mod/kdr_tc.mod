TITLE kdr_tc.mod
 
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
        SUFFIX kdr_tc
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
        a = 0.032*vtrap(-(v+13),5) :to match C++ code, for RE cell replace "13" with "35" (see Vtr and Vtrk in Krishnan's CellSyn.h lines 236-237, currents.h lines 313-314)
        b =  0.5*exp(-(v+18)/40) :to match C++ code, for RE cell replace "18" with "40"
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


