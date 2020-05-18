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
 
#define nrn_init _nrn_init__GABA_A_D2
#define _nrn_initial _nrn_initial__GABA_A_D2
#define nrn_cur _nrn_cur__GABA_A_D2
#define _nrn_current _nrn_current__GABA_A_D2
#define nrn_jacob _nrn_jacob__GABA_A_D2
#define nrn_state _nrn_state__GABA_A_D2
#define _net_receive _net_receive__GABA_A_D2 
#define release release__GABA_A_D2 
#define setrand setrand__GABA_A_D2 
 
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
#define psp_weight _p[1]
#define Erev _p[2]
#define gid _p[3]
#define syn_index _p[4]
#define i _p[5]
#define g _p[6]
#define Ron _p[7]
#define Roff _p[8]
#define synon _p[9]
#define DRon _p[10]
#define DRoff _p[11]
#define _g _p[12]
#define _tsav _p[13]
#define _nd_area  *_ppvar[0]._pval
#define ptr	*_ppvar[2]._pval
#define _p_ptr	_ppvar[2]._pval
 
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
 static int hoc_nrnpointerindex =  2;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_gen_nextpsp();
 static double _hoc_max();
 static double _hoc_pick();
 static double _hoc_setrand();
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
 "gen_nextpsp", _hoc_gen_nextpsp,
 "max", _hoc_max,
 "pick", _hoc_pick,
 "setrand", _hoc_setrand,
 0, 0
};
#define gen_nextpsp gen_nextpsp_GABA_A_D2
#define max max_GABA_A_D2
#define pick pick_GABA_A_D2
 extern double gen_nextpsp( double , double , double );
 extern double max( double , double );
 extern double pick( );
 /* declare global and static user variables */
#define Alpha Alpha_GABA_A_D2
 double Alpha = 0.53;
#define Beta Beta_GABA_A_D2
 double Beta = 0.18;
#define Cmax Cmax_GABA_A_D2
 double Cmax = 0.5;
#define Cdur Cdur_GABA_A_D2
 double Cdur = 0.3;
#define Rtau Rtau_GABA_A_D2
 double Rtau = 0;
#define Rinf Rinf_GABA_A_D2
 double Rinf = 0;
#define SS_denom SS_denom_GABA_A_D2
 double SS_denom = 250;
#define Tr Tr_GABA_A_D2
 double Tr = 700;
#define afterspike_time afterspike_time_GABA_A_D2
 double afterspike_time = 70;
#define deadtime deadtime_GABA_A_D2
 double deadtime = 1;
#define mini_fre mini_fre_GABA_A_D2
 double mini_fre = 20;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "Cdur_GABA_A_D2", "ms",
 "deadtime_GABA_A_D2", "ms",
 "Cmax_GABA_A_D2", "mM",
 "Alpha_GABA_A_D2", "/ms",
 "Beta_GABA_A_D2", "/ms",
 "Tr_GABA_A_D2", "ms",
 "afterspike_time_GABA_A_D2", "ms",
 "mini_fre_GABA_A_D2", "ms",
 "SS_denom_GABA_A_D2", "ms",
 "Rtau_GABA_A_D2", "ms",
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
 "Cdur_GABA_A_D2", &Cdur_GABA_A_D2,
 "deadtime_GABA_A_D2", &deadtime_GABA_A_D2,
 "Cmax_GABA_A_D2", &Cmax_GABA_A_D2,
 "Alpha_GABA_A_D2", &Alpha_GABA_A_D2,
 "Beta_GABA_A_D2", &Beta_GABA_A_D2,
 "Tr_GABA_A_D2", &Tr_GABA_A_D2,
 "afterspike_time_GABA_A_D2", &afterspike_time_GABA_A_D2,
 "mini_fre_GABA_A_D2", &mini_fre_GABA_A_D2,
 "SS_denom_GABA_A_D2", &SS_denom_GABA_A_D2,
 "Rtau_GABA_A_D2", &Rtau_GABA_A_D2,
 "Rinf_GABA_A_D2", &Rinf_GABA_A_D2,
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
 
#define _cvode_ieq _ppvar[4]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"GABA_A_D2",
 "gmax",
 "psp_weight",
 "Erev",
 "gid",
 "syn_index",
 0,
 "i",
 "g",
 0,
 "Ron",
 "Roff",
 0,
 "ptr",
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
 	_p = nrn_prop_data_alloc(_mechtype, 14, _prop);
 	/*initialize range parameters*/
 	gmax = 0.0001;
 	psp_weight = 0.1;
 	Erev = -70;
 	gid = 0;
 	syn_index = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 14;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
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
 
#define _tqitem &(_ppvar[3]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 static void _net_init(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _gaba_A_D2_reg() {
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
  hoc_register_prop_size(_mechtype, 14, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "pointer");
  hoc_register_dparam_semantics(_mechtype, 3, "netsend");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_init[_mechtype] = _net_init;
 pnt_receive_size[_mechtype] = 7;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 GABA_A_D2 C:/Users/finkt/OneDrive - Gonzaga University/Documents/research/bazhenov_sleep_model/network_model/v6_updated/mod/gaba_A_D2.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "gaba_A_D2.mod";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int setrand(double, double);
 
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
     if ( ( t - _args[3] ) > ( Cdur + deadtime )  && ( t - _args[4] ) > ( Cdur + deadtime ) ) {
       _args[6] = 1.0 - ( 1.0 - _args[6] * ( 1.0 - 0.07 ) ) * exp ( - ( t - _args[3] ) / Tr ) ;
       synon = synon + _args[6] * _args[0] ;
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
 _args[3] = t ;
       net_send ( _tqitem, _args, _pnt, t +  Cdur , 2.0 ) ;
       }
     }
   if ( _lflag  == 1.0 ) {
     if ( ( t - _args[4] ) <= ( Cdur + deadtime ) ) {
       }
     if ( ( t - _args[3] ) > ( afterspike_time - 0.00001 )  && ( t - _args[3] ) > ( _args[5] - 0.00001 ) ) {
       synon = synon + psp_weight ;
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
 _args[4] = t ;
       net_send ( _tqitem, _args, _pnt, t +  Cdur , 2.0 ) ;
       _args[5] = 0.0 ;
       while ( _args[5] < ( Cdur + deadtime ) ) {
         _args[5] = gen_nextpsp ( _threadargscomma_ _args[3] , mini_fre , SS_denom ) ;
         }
       net_send ( _tqitem, _args, _pnt, t +  max ( _threadargscomma_ afterspike_time - ( t - _args[3] ) , _args[5] ) , 1.0 ) ;
       }
     else {
       net_send ( _tqitem, _args, _pnt, t +  max ( _threadargscomma_ afterspike_time , _args[5] ) - ( t - _args[3] ) , 1.0 ) ;
       }
     }
   if ( _lflag  == 2.0 ) {
     if ( _args[3] > _args[4] ) {
       synon = synon - _args[6] * _args[0] ;
       _args[1] = _args[6] * _args[0] * Rinf + ( _args[1] - _args[6] * _args[0] * Rinf ) * exp ( - ( t - _args[3] ) / Rtau ) ;
       }
     else {
       synon = synon - psp_weight ;
       _args[1] = psp_weight * Rinf + ( _args[1] - psp_weight * Rinf ) * exp ( - ( t - _args[4] ) / Rtau ) ;
       }
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
 net_send ( _tqitem, _args, _pnt, t +  gen_nextpsp ( _threadargscomma_ - 100.0 * Rtau , mini_fre , SS_denom ) , 1.0 ) ;
   _args[1] = 0.0 ;
   _args[2] = 0.0 ;
   _args[3] = - 100.0 * Rtau ;
   _args[4] = - 100.0 * Rtau ;
   _args[5] = 0.0 ;
   _args[6] = 1.0 ;
   }
 
double max (  double _lx , double _ly ) {
   double _lmax;
 if ( _lx > _ly ) {
     _lmax = _lx ;
     }
   else {
     _lmax = _ly ;
     }
   
return _lmax;
 }
 
static double _hoc_max(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r =  max (  *getarg(1) , *getarg(2) );
 return(_r);
}
 
/*VERBATIM*/
#define VOIDCAST void** vp = (void**)(&(_p_ptr))
extern void * nrnran123_newstream(int,int);
extern void nrnran123_deletestream(void *);
extern double nrnran123_dblpick(void *);
 
static int  setrand (  double _lid1 , double _lid2 ) {
   
/*VERBATIM*/
	VOIDCAST;
	if(*vp) {
		nrnran123_deletestream(*vp);
	} 
	*vp = nrnran123_newstream((int) _lid1,(int) _lid2);
  return 0; }
 
static double _hoc_setrand(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r = 1.;
 setrand (  *getarg(1) , *getarg(2) );
 return(_r);
}
 
double pick (  ) {
   double _lpick;
 
/*VERBATIM*/
	VOIDCAST;
	_lpick = nrnran123_dblpick(*vp);
 
return _lpick;
 }
 
static double _hoc_pick(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r =  pick (  );
 return(_r);
}
 
double gen_nextpsp (  double _llastspike , double _lmini_fre , double _lSS_denom ) {
   double _lgen_nextpsp;
 double _lS , _lSS ;
 _lSS = ( 2.0 / ( 1.0 + exp ( - ( t - _llastspike ) / _lmini_fre ) ) - 1.0 ) / _lSS_denom ;
   _lS = pick ( _threadargs_ ) ;
   if ( _lS < 0.000001 ) {
     _lS = 0.000001 ;
     }
   _lgen_nextpsp = - ( log ( _lS ) ) / _lSS ;
   
return _lgen_nextpsp;
 }
 
static double _hoc_gen_nextpsp(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r =  gen_nextpsp (  *getarg(1) , *getarg(2) , *getarg(3) );
 return(_r);
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
 if(error){fprintf(stderr,"at line 72 in file gaba_A_D2.mod:\n	SOLVE release METHOD cnexp\n"); nrn_complain(_p); abort_run(error);}
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
static const char* nmodl_filename = "gaba_A_D2.mod";
static const char* nmodl_file_text = 
  "TITLE gaba_A_D2.mod\n"
  "\n"
  "COMMENT\n"
  "This file modifies ampa_d2.mod to make a gaba_A synapse (with depression and stochastic stimulation, \n"
  "as in the model used in Krishnan 2016 (eLife) (and similar to Bazhenov 2002). See line 708ff in Krishnan's currents.cpp.\n"
  "First-order synaptic dynamics originally proposed in \"An efficient method for computing synaptic\n"
  "conductances based on a kinetic model of receptor binding\" (Destexhe et. al., 1994).\n"
  "This is an updated version of a mod file originally by Alain Destexhe, ModelDB #18198.\n"
  "This updated version is based primarily on Section 10.1.7 from The NEURON Book, \n"
  "revised to include the parameters Cdur and gmax, as well as a \"deadtime\"\n"
  "--adapted by Christian G. Fink, Gonzaga University--\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON{\n"
  "	POINT_PROCESS GABA_A_D2\n"
  "	NONSPECIFIC_CURRENT i\n"
  "	POINTER ptr\n"
  "	GLOBAL Rinf, Rtau, Cdur, Alpha, Beta, mini_fre, SS_denom  :values of global variables are the same within a mechanism, but not across mechanisms (e.g., AMPA's deadtime may have a different value than AMPA_D1's deadtime)\n"
  "	RANGE g, gmax,  Erev, gid, syn_index, psp_weight \n"
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
  "PARAMETER{ :see line 447 of currents.cpp (from Giri Krishnan) for parameter values\n"
  "	gmax   = 0.0001 (uS) :max conductance of *one* synapse (so in BREAKPOINT, g can be greater than this if there are multiple incoming connections)\n"
  "	psp_weight = 0.1 :unitless factor which scales gmax for the stochastic PSP's prescribed in Bazhenov 2002 and Krishnan 2016\n"
  "	Cdur   = 0.3  (ms) :transmitter duration (rising phase)\n"
  "	deadtime=1.0 (ms) : minimum time between release events; this code assumes Cdur+deadtime<100 ms, because PSP's must happen at least 100 ms after last spike, which is assumed not to be during a deadtime\n"
  "	Cmax   = 0.5	(mM)		: max transmitter concentration\n"
  "	Alpha  = 0.53  (/ms mM):forward (binding) rate (Krishnan currents.cpp line 709)\n"
  "	Beta   = 0.18 (/ms):backward (dissociation) rate (Krishnan currents.cpp line 709)\n"
  "	Erev   = -70    (mV) :equilibrium potential\n"
  "	Tr     = 700 (ms) :time constant for short-term depression\n"
  "	afterspike_time= 70 (ms) :EPSP's can only occur a minimum of 'afterspike_time' after most recent presynaptic spike (Krishnan uses 70 for GABA_A_D2)\n"
  "	mini_fre=20  (ms) :parameter for generating epsp's (this is the same variable name as in currents.cpp in Krishnan's code)\n"
  "	SS_denom=250 (ms) :another parameter for generating epsp's (this is in the denominator of 'SS' in Krishnan's currents.cpp)\n"
  "	gid = 0 :need to remember to change this when the synapse is created in NEURON\n"
  "	syn_index = 0 :may change this when the synapse is created in NEURON; this is used by the random number generator to generate different streams for different synapses on the same cell; so if a cell has more than one synapse which uses Random123, this should be changed\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v    (mV)   : postsynaptic voltage\n"
  "	i    (nA)   : current = g*(v-Erev)\n"
  "	g    (umho) : conductance\n"
  "	Rtau (ms)   : time constant of channel building\n"
  "	Rinf        :fraction of open channels if xmtr is present \"forever\"\n"
  "	synon       :sum of weights of all synapses in the \"onset\" state (where weight is assumed to be a unitless factor which scales gmax)\n"
  "	ptr\n"
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
  "	:setrand(gid,syn_index) :this is now set from NEURON (see emails with Michael Hines, starting 7/24/19)\n"
  "}\n"
  "\n"
  "BREAKPOINT { : would be good to get this in terms of gmax\n"
  "	SOLVE release METHOD cnexp\n"
  "	g = gmax * (Ron + Roff) :max value is weight*gmax*synon*Rinf (for spikes; random PSP's carry psp_weight instead of weight)\n"
  "	i = g*(v - Erev)\n"
  "}\n"
  "\n"
  "DERIVATIVE release {\n"
  "	Ron'  = (synon*Rinf - Ron)/Rtau\n"
  "	Roff' = -Beta*Roff\n"
  "}\n"
  "\n"
  ":weight is assumed to be a unitless factor which scales gmax for presynaptic spikes\n"
  ":short-term depression achieved using the factor 'E,' which is proportion \n"
  ":of available presynaptic resources. \n"
  "\n"
  "NET_RECEIVE(weight, r0, toff (ms), lastspike (ms), lastpsp (ms), nextpsp (ms), E) {\n"
  "	INITIAL{\n"
  "		net_send(gen_nextpsp(-100*Rtau,mini_fre,SS_denom),1) :put a possible PSP event in the queue, using a 'lastspike' value of -100*Rtau (very long time ago); for some reason, placing this statement in the mod file's INITIAL block caused a segmentation fault\n"
  "		r0 = 0\n"
  "		toff = 0 (ms) :this value doesn't matter\n"
  "		lastspike = -100*Rtau :initialize to large neg val so that do not get depression on first spike\n"
  "		lastpsp = -100*Rtau :initialize to large neg val so an early spike will trigger neurotransmitter release\n"
  "		nextpsp = 0.0 (ms) :time until the next PSP *might* be generated (as long as the last spike occurred more than 'afterspike_time' ms ago); this value doesn't matter bc. we initiate PSP event in INITIAL block\n"
  "		E  = 1 :synapse starts out at full strength\n"
  "	}\n"
  "	\n"
  "	:flag is an implicit argument of NET_RECEIVE, normally 0\n"
  "	if (flag == 0){ :flag==0 implies a spike is received \n"
  "		:a spike arrived; ignore it if we are already within either a spike or a psp on state, or deadtime\n"
  "		:printf(\"Entered flag=0: t=%8.2f ms \\n\", t)\n"
  "		if((t-lastspike)>(Cdur + deadtime) && (t-lastpsp)>(Cdur+deadtime)){\n"
  "			:for 'E,' see \"Synaptic Currents\" section of Bazhenov 2002 (and line ~500 of Krishnan's currents.cpp)\n"
  "			E = 1 - (1 - E*(1-0.07)) * exp(-(t-lastspike)/Tr) :note that we are assuming the last spike in this stream occurred at t0-Cdur\n"
  "			synon = synon + E*weight :weight is scaled by 'E' to implement synaptic depression\n"
  "			r0 = r0*exp(-Beta*(t-toff)) :r0 at start of onset state\n"
  "			Ron = Ron + r0\n"
  "			Roff = Roff - r0\n"
  "			lastspike = t \n"
  "			:printf(\"New lastspike time: %8.2f ms \\n\",lastspike)\n"
  "			net_send(Cdur, 2) :turn off neurotransmitter in Cdur with flag = 2\n"
  "		}\n"
  "	}\n"
  "	if (flag == 1) { :possible PSP event, as long as it is sufficiently long after spike and previous psp\n"
  "		if( (t-lastpsp)<=(Cdur+deadtime) ){\n"
  "			:printf(\"Flag=1, and (t-lastpsp)<=(Cdur+deadtime) \\n\")\n"
  "		}\n"
  "		if((t-lastspike)>(afterspike_time-0.00001) && (t-lastspike)>(nextpsp-0.00001)) { :\"-0.00001\" to avoid round-off issues; second clause bc. Krishnan requires that last spike or psp occur at least 'nextpsp' before next psp (and by design, lastpsp occurred at least 'nextpsp' ago); third clause prevents two PSP events from overlapping\n"
  "			synon = synon + psp_weight\n"
  "			r0 = r0*exp(-Beta*(t-toff)) \n"
  "			Ron = Ron + r0\n"
  "			Roff = Roff - r0\n"
  "			lastpsp = t\n"
  "			net_send(Cdur, 2) :turn off neurotransmitter\n"
  "			\n"
  "			nextpsp=0 :generate next epsp time (analogous to 'newrelease' in Krishnan code), and make sure it does not come before the end of Cdur+deadtime\n"
  "			while(nextpsp < (Cdur+deadtime) ){\n"
  "				nextpsp = gen_nextpsp(lastspike,mini_fre,SS_denom) \n"
  "			}\n"
  "			:printf(\"Flag==1: New next psp: t=%8.2f, nextpsp=%8.2f, lastspike=%8.2f \\n\",t, nextpsp,lastspike)\n"
  "			net_send(max(afterspike_time-(t-lastspike),nextpsp),1) :Krishnan's code requires that PSP's occur at least 100 ms after most recent spike; this is earliest time that a PSP could be allowed to happen, so send an event to check at that time\n"
  "		} else {\n"
  "			net_send( max(afterspike_time,nextpsp)-(t-lastspike) , 1) :at this point, we know for sure that lastpsp occurred more than 'nextpsp' ago (bc. when the last psp occurred, we scheduled the next psp to occur 'nextpsp' later), so the next psp must occur more than 100 ms after lastspike or 'nextpsp' ms after lastspike (whichever is greater)\n"
  "			:printf(\"Regenerated psp: t=%8.2f, nextpsp=%8.2f, lastspike=%8.2f \\n\",t,nextpsp,lastspike)\n"
  "		}\n"
  "	}\n"
  "	if (flag == 2) {\n"
  "		: \"turn off transmitter,\" i.e. this synapse enters the offset state\n"
  "		:printf(\"Entered flag==2: t=%8.2f ms \\n\",t)\n"
  "		if(lastspike>lastpsp) { :need to consider that spikes carry different weights from epsp's\n"
  "			synon = synon - E*weight :note we need to include 'E' in order to undo the addition to synon in the above block\n"
  "			: r0 at start of offset state\n"
  "			r0 = E*weight*Rinf + (r0-E*weight*Rinf)*exp(-(t-lastspike)/Rtau)\n"
  "		} else {\n"
  "			synon = synon - psp_weight\n"
  "			r0 = psp_weight*Rinf + (r0-psp_weight*Rinf)*exp(-(t-lastpsp)/Rtau)\n"
  "		}\n"
  "		Ron = Ron - r0\n"
  "		Roff = Roff + r0\n"
  "		:printf(\"Flag=2: Ron=%8.2f, Roff=%8.2f \\n\",Ron,Roff)\n"
  "		toff = t :update time of most recent offset\n"
  "	}\n"
  "}\n"
  "\n"
  "FUNCTION max(x(ms),y(ms)) {\n"
  "	if(x>y) {\n"
  "		max = x\n"
  "	} else {\n"
  "		max = y\n"
  "	}\n"
  "}\n"
  "\n"
  ":this code uses Random123, which requires NEURON 7.3 or higher\n"
  ":uses nrnran123.c and nrnran123.h from http://www.neuron.yale.edu/hg/neuron/nrn/file/9d4ab20927bc/src/oc/\n"
  "VERBATIM\n"
  "#define VOIDCAST void** vp = (void**)(&(_p_ptr))\n"
  "extern void * nrnran123_newstream(int,int);\n"
  "extern void nrnran123_deletestream(void *);\n"
  "extern double nrnran123_dblpick(void *);\n"
  "ENDVERBATIM\n"
  "\n"
  "PROCEDURE setrand(id1,id2) {\n"
  "	VERBATIM\n"
  "	VOIDCAST;\n"
  "	if(*vp) {\n"
  "		nrnran123_deletestream(*vp);\n"
  "	} \n"
  "	*vp = nrnran123_newstream((int) _lid1,(int) _lid2);\n"
  "	ENDVERBATIM\n"
  "} \n"
  "\n"
  "FUNCTION pick() {\n"
  "	VERBATIM\n"
  "	VOIDCAST;\n"
  "	_lpick = nrnran123_dblpick(*vp);\n"
  "	ENDVERBATIM\n"
  "}\n"
  "\n"
  "FUNCTION gen_nextpsp(lastspike (ms),mini_fre (ms),SS_denom(ms)) { :adapted from http://www.neuron.yale.edu/hg/neuron/nrn/file/9d4ab20927bc/src/gnu/NegExp.cpp\n"
  "	LOCAL S, SS\n"
  "	SS = (2.0/(1.0+exp(-(t-lastspike)/mini_fre))-1.0)/SS_denom :this is essentially half of a sigmoid function, starting from 0 at t=0 and saturating at 1 as t->\\infty\n"
  "	S = pick()\n"
  "	if(S < 0.000001){ :just following Krishnan here\n"
  "		S = 0.000001\n"
  "	}\n"
  "	gen_nextpsp = -(log(S))/SS :see AMPA_D2 class in Krishnan's currents.cpp\n"
  "}\n"
  ;
#endif
