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
 /* --------------------------------------------------------------------  */
 /* Auxiliary functions related to features.                              */
 /*                                                                       */
 /*************************************************************************/

#include "EST_features_aux.h"
#include "EST_Features.h"
#include "EST_String.h"
#include "EST_error.h"

#include "EST_get_function_template.h"

defineGetFunction(EST_Features, val, EST_Val, getVal)
defineGetFunction(EST_Features, val, EST_String, getString)
defineGetFunction(EST_Features, val, float, getFloat)
defineGetFunction(EST_Features, val, int, getInteger)
VAL_REGISTER_FUNCPTR(pointer, void *)


void value_sort(EST_Features &f, const EST_String &field)
{
    int work_to_do = 1;
    (void)field;

    EST_Features::RwEntries p;
    EST_Features::RwEntries n;
    EST_Features::Entry e;

    float p_score, n_score;

    while(work_to_do)
    {
	work_to_do = 0;
	for (p.begin(f); p; ++p)
	{
	    n = p;
	    n++;
	    if (n == 0)
		break;

	    if (p->v.type() != val_type_feats)
	    {
		cerr << "Not a features in val\n";
		break;
	    }
	    p_score = feats(p->v)->F(field, 1.0);
	    n_score = feats(n->v)->F(field, 1.0);
	    if (n_score > p_score)
	    {
		cout << "swapping\n";
		e = *p;
		*p = *n;
		*n = e;
		work_to_do = 1;
	    }
	}
    }
}





