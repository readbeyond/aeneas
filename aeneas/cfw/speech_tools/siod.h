/* Scheme In One Defun, but in C this time.
 
 *                   COPYRIGHT (c) 1988-1994 BY                             *
 *        PARADIGM ASSOCIATES INCORPORATED, CAMBRIDGE, MASSACHUSETTS.       *
 *        See the source file SLIB.C for more information.                  *

*/

/*===========================================================*/
/*                                                           */
/* Public LISP functions                                     */
/*                                                           */
/*===========================================================*/
#ifndef __SIOD_H__
#define __SIOD_H__

#include "EST_String.h"
#include "EST_string_aux.h"
#include "EST_error.h"
#include "EST_Val.h"
#include "siod_defs.h"

int siod_init(int heap_size=DEFAULT_HEAP_SIZE);
int siod_repl(int interactive);
void siod_print_welcome(EST_String extra_info);
void siod_print_welcome(void);

const char *get_c_string(LISP x);
int get_c_int(LISP x);
double get_c_double(LISP x);
float get_c_float(LISP x);
LISP flocons(double x);
FILE *get_c_file(LISP p,FILE *deflt);
LISP siod_make_typed_cell(long type, void *s);
LISP cintern(const char *name);
LISP rintern(const char *name);
LISP strintern(const char *data);
LISP strcons(long length,const char *data);
LISP cstrcons(const char *data);

void init_subr(const char *name, long type, SUBR_FUNC fcn, const char *doc);
void init_subr_0(const char *name, LISP (*fcn)(void), const char *doc);
void init_subr_1(const char *name, LISP (*fcn)(LISP), const char *doc);
void init_subr_2(const char *name, LISP (*fcn)(LISP,LISP), const char *doc);
void init_subr_3(const char *name, LISP (*fcn)(LISP,LISP,LISP), const char *doc);
void init_subr_4(const char *name, LISP (*fcn)(LISP,LISP,LISP,LISP), const char *doc);
void init_lsubr(const char *name, LISP (*fcn)(LISP), const char *doc);
void init_fsubr(const char *name, LISP (*fcn)(LISP,LISP), const char *doc);
void init_msubr(const char *name, LISP (*fcn)(LISP *,LISP *), const char *doc);
void setdoc(LISP name,LISP doc);

int siod_register_user_type(const char *name);
void set_gc_hooks(long type,
		  int gc_free_once,
		  LISP (*rel)(LISP),
		  LISP (*mark)(LISP),
		  void (*scan)(LISP),
		  void (*free)(LISP),
		  void (*clear)(LISP),
		  long *kind);
void set_eval_hooks(long type,LISP (*fcn)(LISP, LISP *, LISP *));
void set_type_hooks(long type, long (*c_sxhash)(LISP,long), LISP (*equal)(LISP,LISP));
void set_print_hooks(long type,void (*prin1)(LISP, FILE *), void (*print_string)(LISP, char *));
void set_io_hooks(long type, LISP (*fast_print)(LISP,LISP), LISP (*fast_read)(int,LISP));

void set_fatal_exit_hook(void (*fcn)(void));

extern long nointerrupt;
extern LISP current_env;
extern LISP truth;
extern int audsp_mode;
extern int siod_ctrl_c;
extern const char *siod_prog_name;
extern const char *siod_primary_prompt;
extern const char *siod_secondary_prompt;

void siod_reset_prompt(void);

LISP siod_get_lval(const char *name,const char *message);
LISP siod_set_lval(const char *name,LISP val);
LISP siod_assoc_str(const char *key,LISP alist);
LISP siod_member_str(const char *key,LISP list);
LISP siod_regex_member_str(const EST_String &key,LISP list);
EST_Regex &make_regex(const char *r);
LISP siod_member_int(const int key,LISP list);
LISP siod_nth(int nth,LISP list);
LISP siod_last(LISP list);
int siod_llength(LISP list);
int siod_atomic_list(LISP list);
LISP siod_flatten(LISP tree);
int siod_eof(LISP item);
EST_String siod_sprint(LISP exp);
LISP symbol_boundp(LISP x,LISP env);

LISP get_param_lisp(const char *name, LISP params, LISP defval);
int get_param_int(const char *name, LISP params, int defval);
float get_param_float(const char *name, LISP params, float defval);
const char *get_param_str(const char *name, LISP params,const char *defval);
LISP make_param_int(const char *name, int val);
LISP make_param_float(const char *name, float val);
LISP make_param_str(const char *name,const char *val);
LISP make_param_lisp(const char *name,LISP val);
LISP apply_hooks(LISP hook,LISP arg);
LISP apply_hooks_right(LISP hook,LISP args);
LISP apply(LISP func,LISP args);

int parse_url(const EST_String &url,
	      EST_String &protocol, 
	      EST_String &host, 
	      EST_String &port, 
	      EST_String &path);

LISP err(const char *message, LISP x);
LISP err(const char *message, const char *s);
LISP errswitch(void);

void siod_list_to_strlist(LISP l, EST_StrList &a);
LISP siod_strlist_to_list(EST_StrList &a);
void siod_tidy_up();
LISP siod_quit(void);
const char *siod_version(void);

void gc_protect(LISP *location);
void gc_unprotect(LISP *location);
void gc_protect_n(LISP *location,long n);
void gc_protect_sym(LISP *location,const char *st);
LISP user_gc(LISP args);

// Siod internal function that lots of people use
LISP equal(LISP,LISP);
LISP eql(LISP x,LISP y);
LISP reverse(LISP obj);
LISP append(LISP l1, LISP l2);
LISP cons(LISP x,LISP y);
LISP car(LISP x);
LISP cdr(LISP x);
LISP consp(LISP x);
LISP numberp(LISP x);
LISP atomp(LISP x);
LISP assoc(LISP x,LISP alist);
LISP setcar(LISP cell, LISP value);
LISP setcdr(LISP cell, LISP value);
LISP assq(LISP x,LISP alist);
LISP delq(LISP elem,LISP l);
LISP leval(LISP x,LISP env);
LISP symbol_value(LISP x,LISP env);
LISP setvar(LISP var,LISP val,LISP env);
LISP copy_list(LISP x);
LISP quote(LISP item);
LISP read_from_lstring(LISP x);
LISP symbolexplode(LISP name);

LISP fopen_c(const char *name, const char *how);
LISP fclose_l(LISP p);
LISP lprin1f(LISP exp,FILE *f);
void pprint(LISP exp);
LISP lprint(LISP exp);
void pprint_to_fd(FILE *fd,LISP exp);
LISP lread(void);
LISP lreadtk(long j);
LISP lreadf(FILE *f);
#ifdef WIN32
LISP lreadwinsock(void);
#endif
void set_read_hooks(char *all_set,char *end_set,
		    LISP (*fcn1)(int, struct gen_readio *),
		    LISP (*fcn2)(char *,long, int *));
LISP vload(const char *fname,long cflag);
LISP read_from_string(const char *);
long repl_c_string(char *,long want_sigint,long want_init,long want_print);
long repl_from_socket(int fd);
void init_subrs(void);
LISP stringexplode(const char *str);
void fput_st(FILE *f,const char *st);
LISP get_eof_val(void);


#if 0
void print_hs_1(void);
void print_hs_2(void);
void set_repl_hooks(void (*puts_f)(char *),
		    LISP (*read_f)(void),
		    LISP (*eval_f)(LISP),
		    void (*print_f)(LISP));
long repl(struct repl_hooks *);
LISP lerr(LISP message, LISP x);
LISP eq(LISP x,LISP y);
LISP symcons(char *pname,LISP vcell);
LISP symbol_value_p(LISP x,LISP env,int *set);
LISP subrcons(long type, const char *name, SUBR_FUNC f);

void init_storage(int heap_size=DEFAULT_HEAP_SIZE);

LISP gc_status(LISP args);

/* For user defined types in OBJ */

LISP oblistfn(void);
LISP save_forms(LISP fname,LISP forms,LISP how);
LISP intern(LISP x);
void init_trace(void);
LISP siod_fdopen_c(int fd,const char *name,char *how);

LISP probe_file(LISP fname);

LISP fopen_l(LISP name,LISP how);
LISP fopen_l(LISP name,const char *how);

#endif
#define siod_error()  (errjmp_ok ? longjmp(*est_errjmp,1) : exit(-1))

#include "siod_est.h"

#endif
