/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                       Copyright (c) 1996,1997                         */
/*                        All Rights Reserved.                           */
/*                                                                       */
/*  Permission is hereby granted, free of charge, to use and distribute  */
/*  this software and its documentation without restriction, including   */
/*  without limitation the rights to use, copy, modify, merge, publish,  */
/*  distribute, sublicense, and/or sell copies of this work, and to      */
/*  permit persons to whom this work is furnished to do so, subject to   */
/*  the following conditions:                                            */
/*   1. The code must retain the above copyright notice, this list of    */
/*      conditions and the following disclaimer.                         */
/*   2. Any modifications must be clearly marked as such.                */
/*   3. Original authors' names are not deleted.                         */
/*   4. The authors' names are not used to endorse or promote products   */
/*      derived from this software without specific prior written        */
/*      permission.                                                      */
/*                                                                       */
/*  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK        */
/*  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      */
/*  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   */
/*  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE     */
/*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    */
/*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   */
/*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          */
/*  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       */
/*  THIS SOFTWARE.                                                       */
/*                                                                       */
/*************************************************************************/
/*                     Author :  Alan W Black                            */
/*                     Date   :  April 1996                              */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/*             Top level .h file: main public functions                  */
/*=======================================================================*/
#ifndef __FESTIVAL_H__
#define __FESTIVAL_H__

#include <cstdlib>
#include <fstream>

using namespace std;

#include "EST.h"
#include "EST_cutils.h"
#include "siod.h"

#include "Phone.h"

#ifndef streq
#define streq(X,Y) (strcmp(X,Y)==0)
#endif

struct ModuleDescription;

/* An iostream for outputing debug messages, switchable    */
/* to /dev/null or cerr                                    */
extern ostream *cdebug;
#define cwarn cout
extern "C" FILE* stddebug;
extern int ft_server_socket;
extern const char *festival_version;

/* For server/client */
#define FESTIVAL_DEFAULT_PORT 1314
int festival_socket_client(const char *host,int port);
int festival_start_server(int port);

void festival_initialize(int load_init_files,int heap_size);
void festival_init_lang(const EST_String &language);
int festival_eval_command(const EST_String &expr);
int festival_load_file(const EST_String &fname);
int festival_say_file(const EST_String &fname);
int festival_say_text(const EST_String &text);
int festival_text_to_wave(const EST_String &text,EST_Wave &wave);
void festival_repl(int interactive);
void festival_server_mode(void);
void festival_wait_for_spooler(void);
void festival_tidy_up();

/* Never used and conflicts with some external system */
/* typedef void (*FT_Module)(EST_Utterance &utt); */

/* Feature functions */
void festival_def_nff(const EST_String &name,const EST_String &sname, 
		      EST_Item_featfunc func,const char *doc);
typedef EST_Val (*FT_ff_pref_func)(EST_Item *s,const EST_String &name);
void festival_def_ff_pref(const EST_String &pref,const EST_String &sname, 
			  FT_ff_pref_func func, const char *doc);
EST_Val ffeature(EST_Item *s, const EST_String &name);

/* proclaim a new module 
   option Copyright to add to startup banner
   description is a computer readable description of the
   module
 */
void proclaim_module(const EST_String &name,
		     const EST_String &banner_copyright,
		     const ModuleDescription *description = NULL);

void proclaim_module(const EST_String &name,
			    const ModuleDescription *description = NULL);

void init_module_subr(const char *name, LISP (*fcn)(LISP), const ModuleDescription *description);

/* Some basic functions for accessing structures created by */
/* various modelling techniques                             */
EST_Val wagon_predict(EST_Item *s, LISP tree);
LISP wagon_pd(EST_Item *s, LISP tree);
EST_Val lr_predict(EST_Item *s, LISP lr_model);

/* Grammar access functions */
EST_Ngrammar *get_ngram(const EST_String &name,
			const EST_String &filename = EST_String::Empty);
EST_WFST *get_wfst(const EST_String &name,
		   const EST_String &filename = EST_String::Empty);
LISP lisp_wfst_transduce(LISP wfstname, LISP input);

EST_String map_pos(LISP posmap, const EST_String &pos);
LISP map_pos(LISP posmap, LISP pos);

/* On error do a longjmp to appropriate place */
/* This is done as a macro so the compiler can tell its non-returnable */
#define festival_error()  (errjmp_ok ? longjmp(*est_errjmp,1) : festival_tidy_up(),exit(-1))

/* Add new (utterance) module */
void festival_def_utt_module(const char *name,
			     LISP (*fcn)(LISP),
			     const char *docstring);

void utt_cleanup(EST_Utterance &u); // delete all relations
const EST_String utt_iform_string(EST_Utterance &utt);
LISP utt_iform(EST_Utterance &utt);
const EST_String utt_type(EST_Utterance &utt);
void add_item_features(EST_Item *s,LISP features);

extern const char *festival_libdir;
extern const char *festival_datadir;

//  Module specific LISP/etc definitions
void festival_init_modules(void);

// Some general functions 
LISP ft_get_param(const EST_String &pname);

// SIOD user defined types used by festival

#define tc_festival_dummyobject tc_application_1
#define tc_festival_unit tc_application_2
#define tc_festival_unitdatabase tc_application_3
#define tc_festival_unitindex tc_application_4
#define tc_festival_join tc_application_5
#define tc_festival_schememoduledescription tc_application_6
#define tc_festival_unitcatalogue tc_application_7

// used to recognise our types
#define tc_festival_first_type tc_festival_dummyobject
#define tc_festival_last_type tc_festival_schememoduledescription
#define is_festival_type(X) ((X) >= tc_festival_first_type && (X) <= tc_festival_last_type)

class UnitDatabase *get_c_unitdatabase(LISP x);

#define FESTIVAL_HEAP_SIZE 10000000

#endif
