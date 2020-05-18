/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__iar
#define _nrn_initial _nrn_initial__iar
#define nrn_cur _nrn_cur__iar
#define _nrn_current _nrn_current__iar
#define nrn_jacob _nrn_jacob__iar
#define nrn_state _nrn_state__iar
#define _net_receive _net_receive__iar 
#define activation activation__iar 
#define evaluate_fct evaluate_fct__iar 
#define ihkin ihkin__iar 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define ghbar _p[0]
#define fac_gh_TC _p[1]
#define h_inf _p[2]
#define tau_s _p[3]
#define m _p[4]
#define c1 _p[5]
#define o1 _p[6]
#define o2 _p[7]
#define p0 _p[8]
#define p1 _p[9]
#define eh _p[10]
#define Dc1 _p[11]
#define Do1 _p[12]
#define Do2 _p[13]
#define Dp0 _p[14]
#define Dp1 _p[15]
#define cai _p[16]
#define ih _p[17]
#define gh _p[18]
#define alpha _p[19]
#define beta _p[20]
#define k1ca _p[21]
#define k3p _p[22]
#define tadj _p[23]
#define v _p[24]
#define _g _p[25]
#define _ion_eh	*_ppvar[0]._pval
#define _ion_ih	*_ppvar[1]._pval
#define _ion_dihdv	*_ppvar[2]._pval
#define _ion_cai	*_ppvar[3]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static void _hoc_activation(void);
 static void _hoc_evaluate_fct(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_iar", _hoc_setdata,
 "activation_iar", _hoc_activation,
 "evaluate_fct_iar", _hoc_evaluate_fct,
 0, 0
};
 /* declare global and static user variables */
#define Pc Pc_iar
 double Pc = 0.007;
#define cac cac_iar
 double cac = 0.0015;
#define ginc ginc_iar
 double ginc = 2;
#define k4 k4_iar
 double k4 = 0.001;
#define k2 k2_iar
 double k2 = 0.0004;
#define nexp nexp_iar
 double nexp = 1;
#define nca nca_iar
 double nca = 4;
#define taum taum_iar
 double taum = 20;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "cac_iar", "mM",
 "k2_iar", "1/ms",
 "k4_iar", "1/ms",
 "taum_iar", "ms",
 "ghbar_iar", "mho/cm2",
 "fac_gh_TC_iar", "mV",
 "tau_s_iar", "ms",
 0,0
};
 static double c10 = 0;
 static double delta_t = 0.01;
 static double o20 = 0;
 static double o10 = 0;
 static double p10 = 0;
 static double p00 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "cac_iar", &cac_iar,
 "k2_iar", &k2_iar,
 "Pc_iar", &Pc_iar,
 "k4_iar", &k4_iar,
 "nca_iar", &nca_iar,
 "nexp_iar", &nexp_iar,
 "ginc_iar", &ginc_iar,
 "taum_iar", &taum_iar,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(_NrnThread*, _Memb_list*, int);
static void _ode_matsol(_NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[4]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"iar",
 "ghbar_iar",
 "fac_gh_TC_iar",
 0,
 "h_inf_iar",
 "tau_s_iar",
 "m_iar",
 0,
 "c1_iar",
 "o1_iar",
 "o2_iar",
 "p0_iar",
 "p1_iar",
 0,
 0};
 static Symbol* _h_sym;
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 26, _prop);
 	/*initialize range parameters*/
 	ghbar = 1.7e-005;
 	fac_gh_TC = 0;
 	_prop->param = _p;
 	_prop->param_size = 26;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_h_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* eh */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ih */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dihdv */
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[3]._pval = &prop_ion->param[1]; /* cai */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _thread_cleanup(Datum*);
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _Ih_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("h", 1.0);
 	ion_reg("ca", -10000.);
 	_h_sym = hoc_lookup("h_ion");
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 3);
  _extcall_thread = (Datum*)ecalloc(2, sizeof(Datum));
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 26, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "h_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "h_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "h_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 iar C:/Users/finkt/OneDrive - Gonzaga University/Documents/research/bazhenov_sleep_model/network_model/v6_updated/mod/Ih.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "anomalous rectifier channel";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int activation(_threadargsprotocomma_ double, double);
static int evaluate_fct(_threadargsprotocomma_ double, double);
 extern double *_nrn_thread_getelm();
 
#define _MATELM1(_row,_col) *(_nrn_thread_getelm(_so, _row + 1, _col + 1))
 
#define _RHS1(_arg) _rhs[_arg+1]
  
#define _linmat1  1
 static int _spth1 = 1;
 static int _cvspth1 = 0;
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[5], _dlist1[5]; static double *_temp1;
 static int ihkin();
 
static int ihkin (void* _so, double* _rhs, double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt)
 {int _reset=0;
 {
   double b_flux, f_flux, _term; int _i;
 {int _i; double _dt1 = 1.0/dt;
for(_i=2;_i<5;_i++){
  	_RHS1(_i) = -_dt1*(_p[_slist1[_i]] - _p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 evaluate_fct ( _threadargscomma_ v , cai ) ;
   /* ~ c1 <-> o1 ( alpha , beta )*/
 f_flux =  alpha * c1 ;
 b_flux =  beta * o1 ;
 _RHS1( 2) -= (f_flux - b_flux);
 _RHS1( 3) += (f_flux - b_flux);
 
 _term =  alpha ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 3 ,2)  -= _term;
 _term =  beta ;
 _MATELM1( 2 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ p0 <-> p1 ( k1ca , k2 )*/
 f_flux =  k1ca * p0 ;
 b_flux =  k2 * p1 ;
 _RHS1( 4) -= (f_flux - b_flux);
 
 _term =  k1ca ;
 _MATELM1( 4 ,4)  += _term;
 _term =  k2 ;
 _MATELM1( 4 ,1)  -= _term;
 /*REACTION*/
  /* ~ o1 <-> o2 ( k3p , k4 )*/
 f_flux =  k3p * o1 ;
 b_flux =  k4 * o2 ;
 _RHS1( 3) -= (f_flux - b_flux);
 
 _term =  k3p ;
 _MATELM1( 3 ,3)  += _term;
 _term =  k4 ;
 _MATELM1( 3 ,0)  -= _term;
 /*REACTION*/
   /* p0 + p1 = 1.0 */
 _RHS1(1) =  1.0;
 _MATELM1(1, 1) = 1;
 _RHS1(1) -= p1 ;
 _MATELM1(1, 4) = 1;
 _RHS1(1) -= p0 ;
 /*CONSERVATION*/
  /* c1 + o1 + o2 = 1.0 */
 _RHS1(0) =  1.0;
 _MATELM1(0, 0) = 1;
 _RHS1(0) -= o2 ;
 _MATELM1(0, 3) = 1;
 _RHS1(0) -= o1 ;
 _MATELM1(0, 2) = 1;
 _RHS1(0) -= c1 ;
 /*CONSERVATION*/
   } return _reset;
 }
 
static int  evaluate_fct ( _threadargsprotocomma_ double _lv , double _lcai ) {
   h_inf = 1.0 / ( 1.0 + exp ( ( _lv + 75.0 + fac_gh_TC ) / 5.5 ) ) ;
   tau_s = ( taum + 1000.0 / ( exp ( ( _lv + 71.5 + fac_gh_TC ) / 14.2 ) + exp ( - ( _lv + 89.0 + fac_gh_TC ) / 11.6 ) ) ) / tadj ;
   alpha = h_inf / tau_s ;
   beta = ( 1.0 - h_inf ) / tau_s ;
   k1ca = k2 * pow( ( _lcai / cac ) , nca ) ;
   k3p = k4 * pow( ( p1 / Pc ) , nexp ) ;
    return 0; }
 
static void _hoc_evaluate_fct(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 evaluate_fct ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
static int  activation ( _threadargsprotocomma_ double _lv , double _lcai ) {
   double _lcc ;
 evaluate_fct ( _threadargscomma_ _lv , _lcai ) ;
   _lcc = 1.0 / ( 1.0 + pow( ( cac / _lcai ) , nca ) ) ;
   m = 1.0 / ( 1.0 + beta / alpha + pow( ( _lcc / Pc ) , nexp ) ) ;
   m = ( 1.0 + ginc * pow( ( _lcc / Pc ) , nexp ) ) * m ;
    return 0; }
 
static void _hoc_activation(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 activation ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
/*CVODE ode begin*/
 static int _ode_spec1(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
 {int _i; for(_i=0;_i<5;_i++) _p[_dlist1[_i]] = 0.0;}
 evaluate_fct ( _threadargscomma_ v , cai ) ;
 /* ~ c1 <-> o1 ( alpha , beta )*/
 f_flux =  alpha * c1 ;
 b_flux =  beta * o1 ;
 Dc1 -= (f_flux - b_flux);
 Do1 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ p0 <-> p1 ( k1ca , k2 )*/
 f_flux =  k1ca * p0 ;
 b_flux =  k2 * p1 ;
 Dp0 -= (f_flux - b_flux);
 Dp1 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ o1 <-> o2 ( k3p , k4 )*/
 f_flux =  k3p * o1 ;
 b_flux =  k4 * o2 ;
 Do1 -= (f_flux - b_flux);
 Do2 += (f_flux - b_flux);
 
 /*REACTION*/
   /* p0 + p1 = 1.0 */
 /*CONSERVATION*/
  /* c1 + o1 + o2 = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE matsol*/
 static int _ode_matsol1(void* _so, double* _rhs, double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
   b_flux = f_flux = 0.;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<5;_i++){
  	_RHS1(_i) = _dt1*(_p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 evaluate_fct ( _threadargscomma_ v , cai ) ;
 /* ~ c1 <-> o1 ( alpha , beta )*/
 _term =  alpha ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 3 ,2)  -= _term;
 _term =  beta ;
 _MATELM1( 2 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ p0 <-> p1 ( k1ca , k2 )*/
 _term =  k1ca ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 1 ,4)  -= _term;
 _term =  k2 ;
 _MATELM1( 4 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ o1 <-> o2 ( k3p , k4 )*/
 _term =  k3p ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 0 ,3)  -= _term;
 _term =  k4 ;
 _MATELM1( 3 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
   /* p0 + p1 = 1.0 */
 /*CONSERVATION*/
  /* c1 + o1 + o2 = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE end*/
 
static int _ode_count(int _type){ return 5;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  eh = _ion_eh;
  cai = _ion_cai;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 5; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _cvode_sparse_thread(&_thread[_cvspth1]._pvoid, 5, _dlist1, _p, _ode_matsol1, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  eh = _ion_eh;
  cai = _ion_cai;
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_cleanup(Datum* _thread) {
   _nrn_destroy_sparseobj_thread(_thread[_cvspth1]._pvoid);
   _nrn_destroy_sparseobj_thread(_thread[_spth1]._pvoid);
 }
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_h_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_h_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_h_sym, _ppvar, 2, 4);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 1);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  int _i; double _save;{
  c1 = c10;
  o2 = o20;
  o1 = o10;
  p1 = p10;
  p0 = p00;
 {
   tadj = 1.0 ;
   evaluate_fct ( _threadargscomma_ v , cai ) ;
   p1 = 0.0 ;
   o1 = 0.0 ;
   o2 = 0.0 ;
   c1 = 1.0 ;
   p0 = 1.0 ;
   }
 
}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  eh = _ion_eh;
  cai = _ion_cai;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   m = o1 + ginc * o2 ;
   ih = ghbar * m * ( v - eh ) ;
   }
 _current += ih;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  eh = _ion_eh;
  cai = _ion_cai;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dih;
  _dih = ih;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dihdv += (_dih - ih)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ih += ih ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
double _dtsav = dt;
if (secondorder) { dt *= 0.5; }
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  eh = _ion_eh;
  cai = _ion_cai;
 {  sparse_thread(&_thread[_spth1]._pvoid, 5, _slist1, _dlist1, _p, &t, dt, ihkin, _linmat1, _ppvar, _thread, _nt);
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 5; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
 } }}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(o2) - _p;  _dlist1[0] = &(Do2) - _p;
 _slist1[1] = &(p1) - _p;  _dlist1[1] = &(Dp1) - _p;
 _slist1[2] = &(c1) - _p;  _dlist1[2] = &(Dc1) - _p;
 _slist1[3] = &(o1) - _p;  _dlist1[3] = &(Do1) - _p;
 _slist1[4] = &(p0) - _p;  _dlist1[4] = &(Dp0) - _p;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "Ih.mod";
static const char* nmodl_file_text = 
  "TITLE anomalous rectifier channel\n"
  "COMMENT\n"
  ":\n"
  ": Anomalous Rectifier Ih - cation (Na/K) channel in thalamocortical neurons\n"
  ":\n"
  ": Kinetic model of calcium-induced shift in the activation of Ih channels.\n"
  ": Model of Destexhe et al., Biophys J. 65: 1538-1552, 1993, based on the\n"
  ": voltage-clamp data on the calcium dependence of If in heart cells\n"
  ": (Harigawa & Irisawa, J. Physiol. 409: 121, 1989)\n"
  ":\n"
  ": The voltage-dependence is derived from Huguenard & McCormick, \n"
  ": J Neurophysiol. 68: 1373-1383, 1992, based on voltage-clamp data of \n"
  ": McCormick & Pape, J. Physiol. 431: 291, 1990. \n"
  ":\n"
  ": Modified model of the binding of calcium through a calcium-binding (CB)\n"
  ": protein, which in turn acts on Ih channels.  This model was described in\n"
  ": detail in the following reference:\n"
  ":    Destexhe, A., Bal, T., McCormick, D.A. and Sejnowski, T.J.  Ionic \n"
  ":    mechanisms underlying synchronized oscillations and propagating waves\n"
  ":    in a model of ferret thalamic slices. Journal of Neurophysiology 76:\n"
  ":    2049-2070, 1996.\n"
  ": See also http://www.cnl.salk.edu/~alain , http://cns.fmed.ulaval.ca\n"
  ":\n"
  ":   KINETIC MODEL:\n"
  ":\n"
  ":	  Normal voltage-dependent opening of Ih channels:\n"
  ":\n"
  ":		c1 (closed) <-> o1 (open)	; rate cst alpha(V),beta(V)\n"
  ":\n"
  ":	  Ca++ binding on CB protein\n"
  ":\n"
  ":		p0 (inactive) + nca Ca <-> p1 (active)	; rate cst k1,k2\n"
  ":\n"
  ":	  Binding of active CB protein on the open form (nexp binding sites) :\n"
  ":\n"
  ":		o1 (open) + nexp p1 <-> o2 (open)	; rate cst k3,k4\n"
  ":\n"
  ":\n"
  ":   PARAMETERS:\n"
  ":	It is more useful to reformulate the parameters k1,k2 into\n"
  ":	k2 and cac = (k2/k1)^(1/nca) = half activation calcium dependence, \n"
  ":	and idem for k3,k4 into k4 and Pc = (k4/k3)^(1/nexp) = half activation\n"
  ":	of Ih binding (this is like dealing with tau_m and m_inf instead of\n"
  ":	alpha and beta in Hodgkin-Huxley equations)\n"
  ":	- k2:	this rate constant is the inverse of the real time constant of \n"
  ":             	the binding of Ca to the CB protein\n"
  ":	- cac:	the half activation (affinity) of the CB protein;\n"
  ":		around 1 to 10 microM.  \n"
  ":	- k4:	this rate constant is the inverse of the real time constant of \n"
  ":             	the binding of the CB protein to Ih channels\n"
  ":		very low: it basically governs the interspindle period\n"
  ":	- Pc:	the half activation (affinity) of the Ih channels for the\n"
  ":		CB protein;\n"
  ":	- nca:	number of binding sites of calcium on CB protein; usually 4\n"
  ":	- nexp:	number of binding sites on Ih channels\n"
  ":       - ginc: augmentation of conductance associated with the Ca bound state\n"
  ":	  (about 2-3; see Harigawa & Hirisawa, 1989)\n"
  ":\n"
  ":\n"
  ":   IMPORTANT REMARKS:\n"
  ":       - This simple model for the binding of Ca++ on the open channel \n"
  ":	  suffies to account for the shift in the voltage-dependence of Ih\n"
  ":	  activation with calcium (see details in Destexhe et al, 1993).\n"
  ":	- It may be that calcium just binds to the Ih channel, preventing the \n"
  ":	  conformational change between open and closed; in this case one\n"
  ":	  should take into account binding on the closed state, which is \n"
  ":	  neglected here.\n"
  ":\n"
  ":   MODIFICATIONS\n"
  ":	- this file also contains a procedure (\"activation\") to estimate\n"
  ":	  the steady-state activation of the current; callable from outside\n"
  ":	- the time constant now contains a changeable minimal value (taum)\n"
  ":	- shift: new local variable to displace the voltage-dependence\n"
  ":	  (shift>0 -> depolarizing shift)\n"
  ":\n"
  ":\n"
  ": Alain Destexhe, Salk Institute and Laval University, 1995\n"
  ":\n"
  ": modified by Chris Fink to remove dependence on temperature\n"
  ": also replaced 'shift' with 'fac_gh_TC'\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX iar\n"
  "	USEION h READ eh WRITE ih VALENCE 1\n"
  "	USEION ca READ cai\n"
  "    RANGE ghbar, h_inf, tau_s, m, fac_gh_TC\n"
  "	GLOBAL k2, cac, k4, Pc, nca, nexp, ginc, taum\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(molar)	= (1/liter)\n"
  "	(mM)	= (millimolar)\n"
  "	(mA) 	= (milliamp)\n"
  "	(mV) 	= (millivolt)\n"
  "	(msM)	= (ms mM)\n"
  "}\n"
  "\n"
  "\n"
  "PARAMETER {\n"
  "	eh	= -40	(mV) :Bazhenov C++ line 290\n"
  "	:celsius = 36	(degC)\n"
  "	ghbar	= 0.000017 (mho/cm2)    :Bazhenov C++ line 1506\n"
  "	cac	= 0.0015 (mM)		: half-activation of calcium dependence (Bazhenov C++ line 272)\n"
  "	k2	= 0.0004 (1/ms)		: inverse of time constant (Bazhenov C++ line 292)\n"
  "	Pc	= 0.007			: half-activation of CB protein dependence (Bazhenov C++ line 1497)\n"
  "	k4	= 0.001	(1/ms)		: backward binding on Ih (Bazhenov C++ line 274 & 1498)\n"
  "	nca	= 4			: number of binding sites of ca++ (Bazhenov C++ line 293)\n"
  "	nexp	= 1			: number of binding sites on Ih channels (Bazhenov C++ line 293)\n"
  "	ginc	= 2.0			: augmentation of conductance with Ca++ (Bazhenov C++ line 1493)\n"
  "	taum	= 20	(ms)		: min value of tau (Bazhenov C++ line 293)\n"
  "	fac_gh_TC	= 0	(mV)		: shift of Ih voltage-dependence\n"
  "}\n"
  "\n"
  "\n"
  "STATE {\n"
  "	c1	: closed state of channel\n"
  "	o1	: open state\n"
  "	o2	: CB-bound open state\n"
  "	p0	: resting CB\n"
  "	p1	: Ca++-bound CB\n"
  "}\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "	v	(mV)\n"
  "	cai	(mM)\n"
  "	ih	(mA/cm2)\n"
  "    gh	(mho/cm2)\n"
  "	h_inf\n"
  "	tau_s	(ms)\n"
  "	alpha	(1/ms)\n"
  "	beta	(1/ms)\n"
  "	k1ca	(1/ms)\n"
  "	k3p	(1/ms)\n"
  "	m\n"
  "	tadj\n"
  "}\n"
  "\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE ihkin METHOD sparse\n"
  "\n"
  "	m = o1 + ginc * o2\n"
  "\n"
  "	ih = ghbar * m * (v - eh)\n"
  "}\n"
  "\n"
  "KINETIC ihkin {\n"
  ":\n"
  ":  Here k1ca and k3p are recalculated at each call to evaluate_fct\n"
  ":  because Ca or p1 have to be taken at some power and this does\n"
  ":  not work with the KINETIC block.\n"
  ":  So the kinetics is actually equivalent to\n"
  ":	c1 <-> o1\n"
  ":	p0 + nca Cai <-> p1\n"
  ":	o1 + nexp p1 <-> o2\n"
  "\n"
  "	evaluate_fct(v,cai)\n"
  "\n"
  "	~ c1 <-> o1		(alpha,beta)\n"
  "\n"
  "	~ p0 <-> p1		(k1ca,k2)\n"
  "\n"
  "	~ o1 <-> o2		(k3p,k4)\n"
  "\n"
  "	CONSERVE p0 + p1 = 1\n"
  "	CONSERVE c1 + o1 + o2 = 1\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "INITIAL {\n"
  ":\n"
  ":  Experiments of McCormick & Pape were at 36 deg.C\n"
  ":  Q10 is assumed equal to 3\n"
  ":\n"
  "        tadj = 1.0 :3.0 ^ ((celsius-36 (degC) )/10 (degC) ) :Krishnan currents.h line 379\n"
  "\n"
  "	evaluate_fct(v,cai)\n"
  "\n"
  "	: commented-out intializations are what was used in 2002 Bazhenov C++ code (see 6/29/19 journal entry)\n"
  "	p1 = 0 :1/(1 + (cac/cai) ^ nca)\n"
  "	o1 = 0  :1/(1 + beta/alpha + (p1/Pc)^nexp )\n"
  "	o2 = 0 :((p1/Pc)^nexp) * o1\n"
  "	c1 = 1 :1-o1-o2\n"
  "	p0 = 1 :1-p1\n"
  "	\n"
  "}\n"
  "\n"
  "\n"
  "UNITSOFF\n"
  "PROCEDURE evaluate_fct(v (mV), cai (mM)) {\n"
  "	: CF: I replaced 'shift' with 'fac_gh_TC' and *added* fac_gh_TC (rather than subtracting shift), in order to replicate Krishnan's approach in currents.cpp\n"
  "	h_inf = 1 / ( 1 + exp((v+75+fac_gh_TC)/5.5) )\n"
  "\n"
  "	tau_s = (taum + 1000 / ( exp((v+71.5+fac_gh_TC)/14.2) + exp(-(v+89+fac_gh_TC)/11.6) ) ) / tadj\n"
  "\n"
  "	alpha = h_inf / tau_s\n"
  "	beta  = ( 1 - h_inf ) / tau_s\n"
  "\n"
  "	k1ca = k2 * (cai/cac)^nca :Krishnan currents.cpp line 178\n"
  "\n"
  "	k3p = k4 * (p1/Pc)^nexp :Krishnan currents.cpp line 179\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  ":\n"
  ":  procedure for evaluating the activation curve of Ih\n"
  ":\n"
  "PROCEDURE activation(v (mV), cai (mM)) { LOCAL cc\n"
  "\n"
  "	evaluate_fct(v,cai)\n"
  "\n"
  "	cc = 1 / (1 + (cac/cai)^nca ) 		: equil conc of CB-protein (Bazhenov C++ line 281)\n"
  "\n"
  "	m = 1 / ( 1 + beta/alpha + (cc/Pc)^nexp ) :Bazhenov C++ line 282\n"
  "\n"
  "	m = ( 1 + ginc * (cc/Pc)^nexp ) * m :Bazhenov C++ line 283\n"
  "}\n"
  "\n"
  "UNITSON\n"
  "\n"
  ;
#endif
