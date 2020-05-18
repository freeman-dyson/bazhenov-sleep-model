TITLE hva.mod
 
COMMENT
 HVA calcium current for pyramidal cell and interneurons defined in
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
        SUFFIX hva
        USEION ca READ eca WRITE ica :we are treating eca as a fixed parameter, even though cai is allowed to change (we do this using h.ion_style() )
        RANGE gcabar, gca
        RANGE minf, hinf, mtau, htau
		:THREADSAFE : assigned GLOBALs will be per thread
}
 
PARAMETER {
        gcabar = 0.00001 (S/cm2)	<0,1e9>
        eca = 140 (mV)
}
 
STATE {
        m h
}
 
ASSIGNED {
        v (mV)
        :celsius (degC)

		gca (S/cm2)
        ica (mA/cm2)
        minf hinf
		mtau (ms) htau (ms)
}
 
? currents
BREAKPOINT {
        SOLVE states METHOD cnexp
        gca = gcabar*m*m*h
		ica = 2.952882641412121*gca*(v - eca) :2.95 is q_T from Timofeev 2000; also, C++ code prescribe 2.3^((36-23)/10)
}
 
 
INITIAL {
	rates(v)
	m = minf
	h = hinf
}

? states
DERIVATIVE states {  
        rates(v)
        m' =  (minf-m)/mtau
        h' = (hinf-h)/htau
}
 
? rates
PROCEDURE rates(v(mV)) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
        LOCAL  a, b, sum, c, d
UNITSOFF
		:"m" sodium activation system
        a = 0.055 * vtrap(-27-v,3.8)
        b = 0.94 * exp((-75-v) / 17)
        sum = a + b
		mtau = 0.3386521313023745/sum :numerator is 1/2.952882641412121
        minf = a/sum
        :"h" sodium inactivation system
        c = 0.000457*exp((-13-v)/50)
        d = 0.0065/(exp((-v-15)/28) + 1)
        sum = c + d
		htau = 0.3386521313023745/sum :numerator is 1/2.952882641412121
		hinf = c/sum
        
}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2) :derived by Taylor expanding exp to quadratic term, then binomial approximation
        }else{
                vtrap = x/(exp(x/y) - 1)
        }
}
 
UNITSON


