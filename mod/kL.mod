TITLE kL.mod
 
COMMENT
 Potassium leak current for pyramidal cell and interneurons defined in
 Timofeev et. al. 2000 (Cerebral Cortex) and Bazhenov et. al. 2002 (J Neuro) and Chen et. al. 2012 (J. Physiol.)
 This code is adapted from hh.mod distributed with NEURON source code
 NOTE: This mod file uses 'krev' instead of 'ek' because in cortical cells, Bazhenov et. al. used a potassium reversal
 potential of -95 mV for the potassium leak current, and -90 mV for the other potassium currents
ENDCOMMENT
 
UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
		(S) = (siemens)
}
 
? interface
NEURON {
        SUFFIX kL
        USEION k WRITE ik
		GLOBAL krev
		RANGE gkL
		:THREADSAFE : assigned GLOBALs will be per thread
}
 
PARAMETER {
        gkL = 0.0000025 (S/cm2)	<0,1e9>
        krev = -95 (mV)
}
 
 
ASSIGNED {
        v (mV)
        ik (mA/cm2)
}
 
? currents
BREAKPOINT {
	ik = gkL*(v - krev) 
}
 