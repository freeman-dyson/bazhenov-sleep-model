TITLE naf.mod
 
COMMENT
 Regular and fast (as opposed to persistent) sodium current for pyramidal cell and interneurons defined in
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
        SUFFIX naf
        USEION na READ ena WRITE ina
        RANGE gnabar, gna
        RANGE minf, hinf, mtau, htau
	:THREADSAFE : assigned GLOBALs will be per thread
}
 
PARAMETER {
        gnabar = 0.0015 (S/cm2)	<0,1e9>
        ena = 50 (mV)
}
 
STATE {
        m h
}
 
ASSIGNED {
        v (mV)
        :celsius (degC)

		gna (S/cm2)
        ina (mA/cm2)
        minf hinf
		mtau (ms) htau (ms)
}
 
? currents
BREAKPOINT {
        SOLVE states METHOD cnexp
        gna = gnabar*m*m*m*h
		ina = 2.952882641412121*gna*(v - ena) :2.95 is q_T from Timofeev 2000; also, C++ code prescribe 2.3^((36-23)/10)
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
        a = 0.182 * vtrap(-(v+25),9)
        b =  0.124 * vtrap(v+25,9) :note the lack of a negative sign in the first argument to vtrap
        sum = a + b
		mtau =  0.3386521313023745/sum :numerator is 1/2.952882641412121
        minf = a/sum
                :"h" sodium inactivation system
        hinf = 1/(1 + exp((v+55)/6.2) )
        c = 0.024 * vtrap(-(v+40),5)
        d = 0.0091 * vtrap(v+65,5) :this is actually what is implemented in Bazhenov 2002 C++ code
        :d = 0.0091 * vtrap(-(v-85),5) :this is what is listed in Timofeev 2000 appendix, as well as in Chen 2012
		htau =  0.3386521313023745/(c+d)
        
}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2) :derived by Taylor expanding exp to quadratic term, then binomial approximation
        }else{
                vtrap = x/(exp(x/y) - 1)
        }
}
 
UNITSON


