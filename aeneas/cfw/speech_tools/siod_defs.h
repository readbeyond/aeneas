/* Scheme In One Defun, but in C this time.
 
 *                   COPYRIGHT (c) 1988-1994 BY                             *
 *        PARADIGM ASSOCIATES INCORPORATED, CAMBRIDGE, MASSACHUSETTS.       *
 *        See the source file SLIB.C for more information.                  *

*/

/*************************************************************************/
/*                    Author :  Alan W Black                             */
/*                    Date   :  March 1999                               */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/* Struct and macro definitions for SIOD                                 */
/*                                                                       */
/*=======================================================================*/
#ifndef __EST_SIOD_DEFS_H__
#define __EST_SIOD_DEFS_H__

/* This states the default heap size is effective unset */
/* The size if no heap is specified by a command argument, the */
/* value of the environment variable SIODHEAPSIZE will be used */
/* otherwise ACTUAL_DEFAULT_HEAP_SIZE is used.  This is *not*  */
/* documented because environment variables can cause so many  */
/* problems I'd like to discourage this use unless absolutely  */
/* necessary.                                                  */
#define DEFAULT_HEAP_SIZE -1
#define ACTUAL_DEFAULT_HEAP_SIZE 210000

struct obj
{union {struct {struct obj * car;
		struct obj * cdr;} cons;
	struct {double data;} flonum;
	struct {const char *pname;
		struct obj * vcell;} symbol;
	struct {const char *name;
		struct obj * (*f)(void);} subr0;
  	struct {const char *name;
 		struct obj * (*f)(struct obj *);} subr1;
 	struct {const char *name;
 		struct obj * (*f)(struct obj *, struct obj *);} subr2;
 	struct {const char *name;
 		struct obj * (*f)(struct obj *, struct obj *, struct obj *);
 	      } subr3;
 	struct {const char *name;
 		struct obj * (*f)(struct obj *, struct obj *, 
				  struct obj *, struct obj *);
 	      } subr4;
 	struct {const char *name;
 		struct obj * (*f)(struct obj **, struct obj **);} subrm;
	struct {const char *name;
		struct obj * (*f)(void *,...);} subr;
	struct {struct obj *env;
		struct obj *code;} closure;
	struct {long dim;
		long *data;} long_array;
	struct {long dim;
		double *data;} double_array;
	struct {long dim;
	        char *data;} string;
	struct {long dim;
		struct obj **data;} lisp_array;
	struct {FILE *f;
		char *name;} c_file;
    	struct {EST_Val *v;} val;
    	struct {void *p;} user;
}
 storage_as;
 char *pname;  // This is currently only used by FLONM
 short gc_mark;
 short type;
};

#define CAR(x) ((*x).storage_as.cons.car)
#define CDR(x) ((*x).storage_as.cons.cdr)
#define PNAME(x) ((*x).storage_as.symbol.pname)
#define VCELL(x) ((*x).storage_as.symbol.vcell)
#define SUBR0(x) (*((*x).storage_as.subr0.f))
#define SUBR1(x) (*((*x).storage_as.subr1.f))
#define SUBR2(x) (*((*x).storage_as.subr2.f))
#define SUBR3(x) (*((*x).storage_as.subr3.f))
#define SUBR4(x) (*((*x).storage_as.subr4.f))
#define SUBRM(x) (*((*x).storage_as.subrm.f))
#define SUBRF(x) (*((*x).storage_as.subr.f))
#define FLONM(x) ((*x).storage_as.flonum.data)
#define FLONMPNAME(x) ((*x).pname)
#define USERVAL(x) ((*x).storage_as.user.p)
#define UNTYPEDVAL(x) ((*x).storage_as.user.p)

#define NIL ((struct obj *) 0)
#define EQ(x,y) ((x) == (y))
#define NEQ(x,y) ((x) != (y))
#define NULLP(x) EQ(x,NIL)
#define NNULLP(x) NEQ(x,NIL)

#define TYPE(x) (((x) == NIL) ? 0 : ((*(x)).type))

#define TYPEP(x,y) (TYPE(x) == (y))
#define NTYPEP(x,y) (TYPE(x) != (y))

#define tc_nil    0
#define tc_cons   1
#define tc_flonum 2
#define tc_symbol 3
#define tc_subr_0 4
#define tc_subr_1 5
#define tc_subr_2 6
#define tc_subr_3 7
#define tc_lsubr  8
#define tc_fsubr  9
#define tc_msubr  10
#define tc_closure 11
#define tc_free_cell 12
#define tc_string       13
#define tc_double_array 14
#define tc_long_array   15
#define tc_lisp_array   16
#define tc_c_file       17
#define tc_untyped      18
#define tc_subr_4       19

#define tc_sys_1 31
#define tc_sys_2 32
#define tc_sys_3 33
#define tc_sys_4 34
#define tc_sys_5 35

// older method for adding application specific types
#define tc_application_1 41
#define tc_application_2 42
#define tc_application_3 43
#define tc_application_4 44
#define tc_application_5 45
#define tc_application_6 46
#define tc_application_7 47

// Application specific types may be added using siod_register_user_type()
// Will increment from tc_first_user_type to tc_table_dim
#define tc_first_user_type 50

#define tc_table_dim 100

#define FO_fetch 127
#define FO_store 126
#define FO_list  125
#define FO_listd 124

typedef struct obj* LISP;
typedef LISP (*SUBR_FUNC)(void); 

#define CONSP(x)   TYPEP(x,tc_cons)
#define FLONUMP(x) TYPEP(x,tc_flonum)
#define SYMBOLP(x) TYPEP(x,tc_symbol)
#define STRINGP(x) TYPEP(x,tc_string)

#define NCONSP(x)   NTYPEP(x,tc_cons)
#define NFLONUMP(x) NTYPEP(x,tc_flonum)
#define NSYMBOLP(x) NTYPEP(x,tc_symbol)

// Not for the purists, but I find these more readable than the equivalent
// code inline.

#define CAR1(x) CAR(x)
#define CDR1(x) CDR(x)
#define CAR2(x) CAR(CDR1(x))
#define CDR2(x) CDR(CDR1(x))
#define CAR3(x) CAR(CDR2(x))
#define CDR3(x) CDR(CDR2(x))
#define CAR4(x) CAR(CDR3(x))
#define CDR4(x) CDR(CDR3(x))
#define CAR5(x) CAR(CDR4(x))
#define CDR5(x) CDR(CDR4(x))

#define LISTP(x) (NULLP(x) || CONSP(x))
#define LIST1P(x) (CONSP(x) && NULLP(CDR(x)))
#define LIST2P(x) (CONSP(x) && CONSP(CDR1(x)) && NULLP(CDR2(x)))
#define LIST3P(x) (CONSP(x) && CONSP(CDR1(x)) && CONSP(CDR2(x)) && NULLP(CDR3(x)))
#define LIST4P(x) (CONSP(x) && CONSP(CDR1(x)) && CONSP(CDR2(x)) && CONSP(CDR3(x)) && NULLP(CDR4(x)))
#define LIST5P(x) (CONSP(x) && CONSP(CDR1(x)) && CONSP(CDR2(x)) && CONSP(CDR3(x)) && CONSP(CDR4(x)) &&  NULLP(CDR5(x)))

#define MKPTR(x) (siod_make_ptr((void *)x))

struct gen_readio
{int (*getc_fcn)(char *);
 void (*ungetc_fcn)(int, char *);
 char *cb_argument;};

#define GETC_FCN(x) (*((*x).getc_fcn))((*x).cb_argument)
#define UNGETC_FCN(c,x) (*((*x).ungetc_fcn))(c,(*x).cb_argument)

struct repl_hooks
{void (*repl_puts)(char *);
 LISP (*repl_read)(void);
 LISP (*repl_eval)(LISP);
 void (*repl_print)(LISP);};

/* Macro for defining new class as values public functions */
#define SIOD_REGISTER_CLASS_DCLS(NAME,CLASS)           \
class CLASS *NAME(LISP x);                             \
int NAME##_p(LISP x);                                  \
EST_Val est_val(const class CLASS *v);                 \
LISP siod(const class CLASS *v);

/* Macro for defining new class as siod               */
#define SIOD_REGISTER_CLASS(NAME,CLASS)                \
class CLASS *NAME(LISP x)                              \
{                                                      \
    return NAME(val(x));                               \
}                                                      \
                                                       \
int NAME##_p(LISP x)                                   \
{                                                      \
    if (val_p(x) &&                                    \
        (val_type_##NAME == val(x).type()))            \
	return TRUE;                                   \
    else                                               \
	return FALSE;                                  \
}                                                      \
                                                       \
LISP siod(const class CLASS *v)                        \
{                                                      \
    if (v == 0)                                        \
        return NIL;                                    \
    else                                               \
        return siod(est_val(v));                       \
}                                                      \


/* Macro for defining typedefed something as values public functions */
#define SIOD_REGISTER_TYPE_DCLS(NAME,CLASS)            \
CLASS *NAME(LISP x);                                   \
int NAME##_p(LISP x);                                  \
EST_Val est_val(const CLASS *v);                       \
LISP siod(const CLASS *v);

/* Macro for defining new class as siod               */
#define SIOD_REGISTER_TYPE(NAME,CLASS)                 \
CLASS *NAME(LISP x)                                    \
{                                                      \
    return NAME(val(x));                               \
}                                                      \
                                                       \
int NAME##_p(LISP x)                                   \
{                                                      \
    if (val_p(x) &&                                    \
        (val_type_##NAME == val(x).type()))            \
	return TRUE;                                   \
    else                                               \
	return FALSE;                                  \
}                                                      \
                                                       \
LISP siod(const CLASS *v)                              \
{                                                      \
    if (v == 0)                                        \
        return NIL;                                    \
    else                                               \
        return siod(est_val(v));                       \
}                                                      \


/* Macro for defining function ptr as siod             */
#define SIOD_REGISTER_FUNCPTR(NAME,CLASS)              \
CLASS NAME(LISP x)                                     \
{                                                      \
    return NAME(val(x));                               \
}                                                      \
                                                       \
int NAME##_p(LISP x)                                   \
{                                                      \
    if (val_p(x) &&                                    \
        (val_type_##NAME == val(x).type()))            \
	return TRUE;                                   \
    else                                               \
	return FALSE;                                  \
}                                                      \
                                                       \
LISP siod(const CLASS v)                               \
{                                                      \
    if (v == 0)                                        \
        return NIL;                                    \
    else                                               \
        return siod(est_val(v));                       \
}                                                      \

#endif 
