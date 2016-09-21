 /************************************************************************/
 /*                                                                      */
 /*                Centre for Speech Technology Research                 */
 /*                     University of Edinburgh, UK                      */
 /*                       Copyright (c) 1996,1997                        */
 /*                        All Rights Reserved.                          */
 /*                                                                      */
 /*  Permission is hereby granted, free of charge, to use and distribute */
 /*  this software and its documentation without restriction, including  */
 /*  without limitation the rights to use, copy, modify, merge, publish, */
 /*  distribute, sublicense, and/or sell copies of this work, and to     */
 /*  permit persons to whom this work is furnished to do so, subject to  */
 /*  the following conditions:                                           */
 /*   1. The code must retain the above copyright notice, this list of   */
 /*      conditions and the following disclaimer.                        */
 /*   2. Any modifications must be clearly marked as such.               */
 /*   3. Original authors' names are not deleted.                        */
 /*   4. The authors' names are not used to endorse or promote products  */
 /*      derived from this software without specific prior written       */
 /*      permission.                                                     */
 /*                                                                      */
 /*  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK       */
 /*  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING     */
 /*  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT  */
 /*  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE    */
 /*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES   */
 /*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN  */
 /*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,         */
 /*  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF      */
 /*  THIS SOFTWARE.                                                      */
 /*                                                                      */
 /*************************************************************************/
 /*                                                                       */
 /*                 Author: Richard Caley (rjc@cstr.ed.ac.uk)             */
 /*                   Date: Tue Jul 29 1997                               */
 /* --------------------------------------------------------------------  */
 /* Some things which are useful in modules.                              */
 /*                                                                       */
 /*************************************************************************/


#ifndef __MODULE_SUPPORT_H__
#define __MODULE_SUPPORT_H__

#include "EST.h"
#include "festival.h"
#include "ModuleDescription.h"

// To extract arguments passed as a list

void unpack_multiple_args(LISP args, LISP &v1, LISP &v2, LISP &v3, LISP &v4);
void unpack_multiple_args(LISP args, LISP &v1, LISP &v2, LISP &v3, LISP &v4, LISP &v5);

// To extract arguments for a module, modules are called
//	(module-function Utterance StreamName1 StreamName2 StreamName3 ...)

// To tell the unpacking functions what is expected of the stream.
enum RelArgType {
	sat_existing,		// must exist
	sat_new,		// must be new
	sat_replace,		// erase if there, then create
	sat_as_is		// take what we find
};
	

void unpack_relation_arg(EST_Utterance *utt,
		       LISP lrelation_name,
		       EST_String &relation_name, EST_Relation *&relation, RelArgType type);

void unpack_module_args(LISP args, 
			EST_Utterance *&utt);
void unpack_module_args(LISP args, 
			EST_Utterance *&utt, 
			EST_String &relation1_name, EST_Relation *&relation1, RelArgType type1);
void unpack_module_args(LISP args, 
			EST_Utterance *&utt,
			EST_String &relation1_name, EST_Relation *&relation1, RelArgType type1,
			EST_String &relation2_name, EST_Relation *&relation2, RelArgType type2
			);
void unpack_module_args(LISP args, 
			EST_Utterance *&utt,
			EST_String &relation1_name, EST_Relation *&relation1, RelArgType type1,
			EST_String &relation2_name, EST_Relation *&relation2, RelArgType type2,
			EST_String &relation3_name, EST_Relation *&relation3, RelArgType type3
			);
void unpack_module_args(LISP args, 
			EST_Utterance *&utt,
			EST_String &relation1_name, EST_Relation *&relation1, RelArgType type1,
			EST_String &relation2_name, EST_Relation *&relation2, RelArgType type2,
			EST_String &relation3_name, EST_Relation *&relation3, RelArgType type3,
			EST_String &relation4_name, EST_Relation *&relation4, RelArgType type4
			);
void unpack_module_args(LISP args, 
			EST_Utterance *&utt,
			EST_String &relation1_name, EST_Relation *&relation1, RelArgType type1,
			EST_String &relation2_name, EST_Relation *&relation2, RelArgType type2,
			EST_String &relation3_name, EST_Relation *&relation3, RelArgType type3,
			EST_String &relation4_name, EST_Relation *&relation4, RelArgType type4,
			EST_String &relation5_name, EST_Relation *&relation5, RelArgType type5
			);

LISP	      lisp_parameter_get(const EST_String parameter);
int            int_parameter_get(const EST_String parameter, int def=0);
float        float_parameter_get(const EST_String parameter, float def=0.0);
bool          bool_parameter_get(const EST_String parameter);
EST_String  string_parameter_get(const EST_String parameter, EST_String def="");

#endif
