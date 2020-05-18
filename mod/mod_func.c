#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _Ih_reg();
extern void _ampa_reg();
extern void _ampa_D1_reg();
extern void _ampa_D2_reg();
extern void _ampa_NEURON_reg();
extern void _cadecay_reg();
extern void _gaba_A_reg();
extern void _gaba_A_D2_reg();
extern void _gaba_B_reg();
extern void _hva_reg();
extern void _iT_RE_reg();
extern void _iT_TC_reg();
extern void _kL_reg();
extern void _kca_reg();
extern void _kdr_reg();
extern void _kdr_re_reg();
extern void _kdr_tc_reg();
extern void _km_reg();
extern void _naf_reg();
extern void _naf_re_reg();
extern void _naf_tc_reg();
extern void _nap_reg();
extern void _nmda_D1_reg();
extern void _xtra_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," Ih.mod");
fprintf(stderr," ampa.mod");
fprintf(stderr," ampa_D1.mod");
fprintf(stderr," ampa_D2.mod");
fprintf(stderr," ampa_NEURON.mod");
fprintf(stderr," cadecay.mod");
fprintf(stderr," gaba_A.mod");
fprintf(stderr," gaba_A_D2.mod");
fprintf(stderr," gaba_B.mod");
fprintf(stderr," hva.mod");
fprintf(stderr," iT_RE.mod");
fprintf(stderr," iT_TC.mod");
fprintf(stderr," kL.mod");
fprintf(stderr," kca.mod");
fprintf(stderr," kdr.mod");
fprintf(stderr," kdr_re.mod");
fprintf(stderr," kdr_tc.mod");
fprintf(stderr," km.mod");
fprintf(stderr," naf.mod");
fprintf(stderr," naf_re.mod");
fprintf(stderr," naf_tc.mod");
fprintf(stderr," nap.mod");
fprintf(stderr," nmda_D1.mod");
fprintf(stderr," xtra.mod");
fprintf(stderr, "\n");
    }
_Ih_reg();
_ampa_reg();
_ampa_D1_reg();
_ampa_D2_reg();
_ampa_NEURON_reg();
_cadecay_reg();
_gaba_A_reg();
_gaba_A_D2_reg();
_gaba_B_reg();
_hva_reg();
_iT_RE_reg();
_iT_TC_reg();
_kL_reg();
_kca_reg();
_kdr_reg();
_kdr_re_reg();
_kdr_tc_reg();
_km_reg();
_naf_reg();
_naf_re_reg();
_naf_tc_reg();
_nap_reg();
_nmda_D1_reg();
_xtra_reg();
}
