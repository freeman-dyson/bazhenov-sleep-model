/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
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
 
#define nrn_init _nrn_init__AMPA
#define _nrn_initial _nrn_initial__AMPA
#define nrn_cur _nrn_cur__AMPA
#define _nrn_current _nrn_current__AMPA
#define nrn_jacob _nrn_jacob__AMPA
#define nrn_state _nrn_state__AMPA
#define _net_receive _net_receive__AMPA 
#define release release__AMPA 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define gmax _p[0]
#define Erev _p[1]
#define i _p[2]
#define g _p[3]
#define Ron _p[4]
#define Roff _p[5]
#define synon _p[6]
#define DRon _p[7]
#define DRoff _p[8]
#define _g _p[9]
#define _tsav _p[10]
#define _nd_area  *_ppvar[0]._pval
 
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
 /* external NEURON variables */
 /* declaration of user functions */
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

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(_ho) Object* _ho; { void* create_point_process();
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt();
 static double _hoc_loc_pnt(_vptr) void* _vptr; {double loc_point_process();
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(_vptr) void* _vptr; {double has_loc_point();
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(_vptr)void* _vptr; {
 double get_loc_point_process(); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 0, 0
};
 /* declare global and static user variables */
#define Alpha Alpha_AMPA
 double Alpha = 1.1;
#define Beta Beta_AMPA
 double Beta = 0.19;
#define Cmax Cmax_AMPA
 double Cmax = 0.5;
#define Cdur Cdur_AMPA
 double Cdur = 0.3;
#define Rtau Rtau_AMPA
 double Rtau = 0;
#define Rinf Rinf_AMPA
 double Rinf = 0;
#define deadtime deadtime_AMPA
 double deadtime = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "Cdur_AMPA", "ms",
 "deadtime_AMPA", "ms",
 "Cmax_AMPA", "mM",
 "Alpha_AMPA", "/ms",
 "Beta_AMPA", "/ms",
 "Rtau_AMPA", "ms",
 "gmax", "uS",
 "Erev", "mV",
 "i", "nA",
 "g", "umho",
 0,0
};
 static double Roff0 = 0;
 static double Ron0 = 0;
 static double delta_t = 0.01;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "Cdur_AMPA", &Cdur_AMPA,
 "deadtime_AMPA", &deadtime_AMPA,
 "Cmax_AMPA", &Cmax_AMPA,
 "Alpha_AMPA", &Alpha_AMPA,
 "Beta_AMPA", &Beta_AMPA,
 "Rtau_AMPA", &Rtau_AMPA,
 "Rinf_AMPA", &Rinf_AMPA,
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
 static void _hoc_destroy_pnt(_vptr) void* _vptr; {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(_NrnThread*, _Memb_list*, int);
static void _ode_matsol(_NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"AMPA",
 "gmax",
 "Erev",
 0,
 "i",
 "g",
 0,
 "Ron",
 "Roff",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 11, _prop);
 	/*initialize range parameters*/
 	gmax = 0.0001;
 	Erev = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 11;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 
#define _tqitem &(_ppvar[2]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 static void _net_init(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _ampa_reg() {
	int _vectorized = 0;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 0,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 11, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "netsend");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_init[_mechtype] = _net_init;
 pnt_receive_size[_mechtype] = 4;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 AMPA C:/Users/finkt/OneDrive - Gonzaga University/Documents/research/bazhenov_sleep_model/network_model/v6_updated/mod/ampa.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "ampa.mod";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int release(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   DRon = ( synon * Rinf - Ron ) / Rtau ;
   DRoff = - Beta * Roff ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 DRon = DRon  / (1. - dt*( ( ( ( - 1.0 ) ) ) / Rtau )) ;
 DRoff = DRoff  / (1. - dt*( ( - Beta )*( 1.0 ) )) ;
  return 0;
}
 /*END CVODE*/
 static int release () {_reset=0;
 {
    Ron = Ron + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / Rtau)))*(- ( ( ( ( synon )*( Rinf ) ) ) / Rtau ) / ( ( ( ( - 1.0 ) ) ) / Rtau ) - Ron) ;
    Roff = Roff + (1. - exp(dt*(( - Beta )*( 1.0 ))))*(- ( 0.0 ) / ( ( - Beta )*( 1.0 ) ) - Roff) ;
   }
  return 0;
}
 
static void _net_receive (_pnt, _args, _lflag) Point_process* _pnt; double* _args; double _lflag; 
{    _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( _lflag  == 0.0 ) {
     if ( ( t - _args[3] ) > ( Cdur + deadtime ) ) {
       synon = synon + _args[0] ;
       _args[1] = _args[1] * exp ( - Beta * ( t - _args[2] ) ) ;
         if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Ron;
    double __primary = (Ron + _args[1]) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( ( ( - 1.0 ) ) ) / Rtau ) ) )*( - ( ( ( ( synon )*( Rinf ) ) ) / Rtau ) / ( ( ( ( - 1.0 ) ) ) / Rtau ) - __primary );
    Ron += __primary;
  } else {
 Ron = Ron + _args[1] ;
         }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Roff;
    double __primary = (Roff - _args[1]) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - Beta )*( 1.0 ) ) ) )*( - ( 0.0 ) / ( ( - Beta )*( 1.0 ) ) - __primary );
    Roff += __primary;
  } else {
 Roff = Roff - _args[1] ;
         }
 _args[2] = t ;
       _args[3] = t ;
       net_send ( _tqitem, _args, _pnt, t +  Cdur , 1.0 ) ;
       }
     }
   if ( _lflag  == 1.0 ) {
     synon = synon - _args[0] ;
     _args[1] = _args[0] * Rinf + ( _args[1] - _args[0] * Rinf ) * exp ( - ( t - _args[2] ) / Rtau ) ;
       if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Ron;
    double __primary = (Ron - _args[1]) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( ( ( - 1.0 ) ) ) / Rtau ) ) )*( - ( ( ( ( synon )*( Rinf ) ) ) / Rtau ) / ( ( ( ( - 1.0 ) ) ) / Rtau ) - __primary );
    Ron += __primary;
  } else {
 Ron = Ron - _args[1] ;
       }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Roff;
    double __primary = (Roff + _args[1]) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - Beta )*( 1.0 ) ) ) )*( - ( 0.0 ) / ( ( - Beta )*( 1.0 ) ) - __primary );
    Roff += __primary;
  } else {
 Roff = Roff + _args[1] ;
       }
 _args[2] = t ;
     }
   } }
 
static void _net_init(Point_process* _pnt, double* _args, double _lflag) {
       _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
 _args[1] = 0.0 ;
   _args[2] = 0.0 ;
   _args[3] = - 10.0 * ( Cdur + deadtime ) ;
   }
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 ();
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 ();
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  Roff = Roff0;
  Ron = Ron0;
 {
   synon = 0.0 ;
   Rtau = 1.0 / ( ( Alpha * Cmax ) + Beta ) ;
   Rinf = Cmax * Alpha / ( Cmax * Alpha + Beta ) ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
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
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   g = gmax * ( Ron + Roff ) ;
   i = g * ( v - Erev ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
 _g = _nrn_current(_v + .001);
 	{ _rhs = _nrn_current(_v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
 
}}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
 { error =  release();
 if(error){fprintf(stderr,"at line 62 in file ampa.mod:\n	:g = (Ron + Roff)*1(umho)\n"); nrn_complain(_p); abort_run(error);}
 }}}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(Ron) - _p;  _dlist1[0] = &(DRon) - _p;
 _slist1[1] = &(Roff) - _p;  _dlist1[1] = &(DRoff) - _p;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "ampa.mod";
static const char* nmodl_file_text = 
  "TITLE ampa.mod\n"
  "\n"
  "COMMENT\n"
  "First-order synaptic dynamics originally proposed in \"An efficient method for computing synaptic\n"
  "conductances based on a kinetic model of receptor binding\" (Destexhe et. al., 1994).\n"
  "This is an updated version of a mod file originally by Alain Destexhe, ModelDB #18198.\n"
  "\n"
  "This updated version is based primarily on Section 10.1.7 from The NEURON Book. It is\n"
  "revised (from ampa_NEURON.mod) to include the parameters Cdur and gmax, as well as to include a \"deadtime,\" as in \n"
  "Destexhe's original model. \n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON{\n"
  "	POINT_PROCESS AMPA\n"
  "	NONSPECIFIC_CURRENT i\n"
  "	GLOBAL deadtime, Cdur, Alpha, Beta, Rinf, Rtau :values of global variables are the same within a mechanism, but not across mechanisms (e.g., AMPA's deadtime may have a different value than AMPA_D1's deadtime)\n"
  "	RANGE g, gmax, Erev\n"
  "}\n"
  "\n"
  "UNITS{\n"
  "	(nA) = (nanoamp)\n"
  "	(mV) = (millivolt)\n"
  "	(umho) = (micromho)\n"
  "	(mM) = (milli/liter)\n"
  "	(uS) = (micromho)\n"
  "}\n"
  "\n"
  "PARAMETER{ :see line 390 of currents.cpp (from Giri Krishnan) for parameter values\n"
  "	gmax   = 0.0001 (uS) :max conductance of *one* synapse (so in BREAKPOINT, g can be greater than this if there are multiple incoming connections)\n"
  "	Cdur   = 0.3  (ms) :transmitter duration (rising phase)\n"
  "	deadtime=1.0 (ms) : minimum time between release events\n"
  "	Cmax   = 0.5	(mM)		: max transmitter concentration\n"
  "	Alpha  = 1.1  (/ms mM):forward (binding) rate (Krishnan currents.cpp line 394)\n"
  "	Beta   = 0.19 (/ms):backward (dissociation) rate (Krishnan currents.cpp line 394)\n"
  "	Erev   = 0    (mV) :equilibrium potential\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v    (mV)   : postsynaptic voltage\n"
  "	i    (nA)   : current = g*(v-Erev)\n"
  "	g    (umho) : conductance\n"
  "	Rtau (ms)   : time constant of channel building\n"
  "	Rinf        :fraction of open channels if xmtr is present \"forever\"\n"
  "	synon       :sum of weights of all synapses in the \"onset\" state (where weight is assumed to be a unitless factor which scales gmax)\n"
  "}\n"
  "\n"
  "STATE { Ron Roff }  :initialized to 0 by default\n"
  ": Ron and Roff are the total conductances of all synapses\n"
  ": that are in the \"onset\" (transmitter pulse ON)\n"
  ": and \"offset\" (transmitter pulse OFF) states, respectively\n"
  ":declared without units, so units are specified in BREAKPOINT block\n"
  "\n"
  "INITIAL {\n"
  "	:Ron and Roff default to being initialized to zero\n"
  "	synon = 0\n"
  "	Rtau = 1 / ((Alpha * Cmax) + Beta)\n"
  "	Rinf = Cmax*Alpha / (Cmax*Alpha + Beta)\n"
  "}\n"
  "\n"
  "BREAKPOINT { : would be good to get this in terms of gmax\n"
  "	SOLVE release METHOD cnexp\n"
  "	:g = (Ron + Roff)*1(umho)\n"
  "	g = gmax * (Ron + Roff) :max value is gmax*synon*Rinf\n"
  "	i = g*(v - Erev)\n"
  "}\n"
  "\n"
  "DERIVATIVE release {\n"
  "	Ron'  = (synon*Rinf - Ron)/Rtau\n"
  "	Roff' = -Beta*Roff\n"
  "}\n"
  "\n"
  ":weight is assumed to be a unitless factor which scales gmax\n"
  "NET_RECEIVE(weight, r0, t0 (ms), lastspike (ms)) {\n"
  "	INITIAL{\n"
  "		r0 = 0\n"
  "		t0 = 0 (ms)\n"
  "		lastspike = -10*(Cdur + deadtime)\n"
  "	}\n"
  "	:flag is an implicit argument of NET_RECEIVE, normally 0\n"
  "	if (flag == 0){ :flag==0 implies a spike is received\n"
  "		:a spike arrived; ignore it if we are already within either a spike state, or deadtime\n"
  "		if( (t-lastspike)>(Cdur + deadtime) ){\n"
  "			synon = synon + weight\n"
  "			r0 = r0*exp(-Beta*(t-t0)) :r0 at start of onset state\n"
  "			Ron = Ron + r0\n"
  "			Roff = Roff - r0\n"
  "			t0 = t\n"
  "			lastspike = t\n"
  "			:come again in Cdur with flag = 1\n"
  "			net_send(Cdur, 1)\n"
  "		}\n"
  "	}\n"
  "	if (flag == 1) {\n"
  "		: \"turn off transmitter\"\n"
  "		: i.e. this synapse enters the offset state\n"
  "		synon = synon - weight\n"
  "		: r0 at start of offset state\n"
  "		r0 = weight*Rinf + (r0-weight*Rinf)*exp(-(t-t0)/Rtau)\n"
  "		Ron = Ron - r0\n"
  "		Roff = Roff + r0\n"
  "		t0 = t\n"
  "	}\n"
  "}\n"
  ;
#endif
