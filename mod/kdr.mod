TITLE kdr.mod
 
COMMENT
 Delayed rectifying potassium current for pyramidal cell and interneurons defined in
 Timofeev et. al. 2000 (Cerebral Cortex) and Bazhenov et. al. 2002 (J Neuro) and Chen et. al. 2012 (J. Physiol.)
 This code is adapted from hh.mod distributed with NEURON source code
ENDCOMMENT
 
UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	(S) = (siemens)
}
 
? interface
NEURON {
        SUFFIX kdr
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
        gk = gkbar*n
		ik = 2.952882641412121*gk*(v - ek) :2.95 is q_T from Timofeev 2000; also, C++ code prescribe 2.3^((36-23)/10)
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
        a = 0.02 * vtrap(-(v-25),9)
        b =  0.002 * vtrap(v-25,9) :note the lack of a negative sign in the first argument to vtrap; also no negative sign in front of entire equation, because of reversed order of denominator
        sum = a + b
		ntau = 0.3386521313023745/sum :numerator is 1/2.952882641412121
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


