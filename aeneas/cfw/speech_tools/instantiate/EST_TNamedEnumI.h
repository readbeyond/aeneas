
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


#ifndef __EST_TNamedEnum_I_H__
#define __EST_TNamedEnum_I_H__

/** Instantiate rules for named enum template.
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_TNamedEnumI.h,v 1.2 2001/04/04 13:11:27 awb Exp $
  */

// Instantiation Macros

#define Instantiate_TValuedEnumI_T(ENUM, VAL, INFO, TAG) \
	template class EST_TValuedEnumI< ENUM, VAL, INFO >; \


#define Instantiate_TValuedEnum_T(ENUM, VAL, TAG) \
	Instantiate_TValuedEnumI_T(ENUM, VAL, NO_INFO, TAG)

#define Instantiate_TNamedEnumI_T(ENUM, INFO, TAG) \
	template class EST_TNamedEnumI< ENUM, INFO >; \
	Instantiate_TValuedEnumI_T(ENUM, const char *, INFO, TAG)

#define Instantiate_TNamedEnum_T(ENUM, TAG) \
	Instantiate_TValuedEnumI_T(ENUM, const char *, NO_INFO, TAG) \
	template class EST_TNamedEnum< ENUM >; \
	template class EST_TNamedEnumI< ENUM, NO_INFO >;


#define Instantiate_TValuedEnumI(ENUM, VAL, INFO) \
	  Instantiate_TValuedEnumI_T(ENUM, VAL, INFO, ENUM ## VAL ## INFO )

#define Instantiate_TValuedEnum(ENUM, VAL) \
	template class EST_TValuedEnum< ENUM, VAL >; \
	Instantiate_TValuedEnum_T(ENUM, VAL, ENUM ## VAL)


#define Instantiate_TNamedEnumI(ENUM, INFO) \
	Instantiate_TNamedEnumI_T(ENUM, INFO, ENUM ## INFO)

#define Instantiate_TNamedEnum(ENUM) \
	Instantiate_TNamedEnum_T(ENUM, ENUM) \

// declaration macros. NULL at the moment.

#define Declare_TValuedEnumI_T(ENUM, VAL, INFO, TAG) \
	  /* EMPTY */

#define Declare_TValuedEnum_T(ENUM, VAL, TAG) \
	Declare_TValuedEnumI_T(ENUM, VAL, NO_INFO, TAG)

#define Declare_TNamedEnumI_T(ENUM, INFO, TAG) \
	Declare_TValuedEnumI_T(ENUM, const char *, INFO, TAG)

#define Declare_TNamedEnum_T(ENUM, TAG) \
	Declare_TNamedEnumI_T(ENUM, NO_INFO, TAG)


#define Declare_TValuedEnumI(ENUM, VAL, INFO) \
	  Declare_TValuedEnumI_T(ENUM, VAL, INFO, ENUM ## VAL ## INFO )

#define Declare_TValuedEnum(ENUM, VAL) \
	Declare_TValuedEnum_T(ENUM, VAL, ENUM ## VAL) 

#define Declare_TNamedEnumI(ENUM, INFO) \
	Declare_TNamedEnumI_T(ENUM, INFO, ENUM ## INFO)

#define Declare_TNamedEnum(ENUM) \
	Declare_TNamedEnum_T(ENUM, ENUM)

// Actual table declaration macros

#define Create_TValuedEnumDefinition(ENUM, VAL, INFO, TAG) \
     static EST_TValuedEnumDefinition< ENUM, VAL, INFO> TAG ## _names[] 

#define Create_TNamedEnumDefinition(ENUM, INFO, TAG) \
	Create_TValuedEnumDefinition(ENUM, const char *, INFO, TAG)

#define Start_TValuedEnumI_T(ENUM, VAL, INFO, NAME, TAG) \
	  Create_TValuedEnumDefinition(ENUM, VAL, INFO, TAG) = {
#define End_TValuedEnumI_T(ENUM, VAL, INFO, NAME, TAG) \
	  }; \
        EST_TValuedEnumI< ENUM, VAL, INFO > NAME (TAG ## _names);

#define Start_TNamedEnumI_T(ENUM, INFO, NAME, TAG) \
	Create_TValuedEnumDefinition(ENUM, const char *, INFO, TAG) = {
#define End_TNamedEnumI_T(ENUM, INFO, NAME, TAG) \
	}; \
        EST_TNamedEnumI< ENUM, INFO > NAME (TAG ## _names);

#define Start_TValuedEnumI(ENUM, VAL, INFO, NAME) \
	Start_TValuedEnumI_T(ENUM, VAL, INFO, NAME, NAME)
#define End_TValuedEnumI(ENUM, VAL, INFO, NAME) \
	End_TValuedEnumI_T(ENUM, VAL, INFO, NAME, NAME)

#define Start_TNamedEnumI(ENUM, INFO, NAME) \
	Start_TNamedEnumI_T(ENUM, INFO, NAME, NAME)
#define End_TNamedEnumI(ENUM, INFO, NAME) \
	End_TNamedEnumI_T(ENUM, INFO, NAME, NAME)

#define Start_TValuedEnum_T(ENUM, VAL, NAME, TAG) \
	  Create_TValuedEnumDefinition(ENUM, VAL, NO_INFO, TAG) = {
#define End_TValuedEnum_T(ENUM, VAL, NAME, TAG) \
	  }; \
        EST_TValuedEnum< ENUM, VAL > NAME (TAG ## _names);

#define Start_TNamedEnum_T(ENUM, NAME, TAG) \
	   Create_TValuedEnumDefinition(ENUM, const char *, NO_INFO, TAG) = {
#define End_TNamedEnum_T(ENUM, NAME, TAG) \
	   }; \
        EST_TNamedEnum< ENUM > NAME (TAG ## _names);

#define Start_TValuedEnum(ENUM, VAL, NAME) \
	Start_TValuedEnum_T(ENUM, VAL, NAME, NAME)
#define End_TValuedEnum(ENUM, VAL, NAME) \
	End_TValuedEnum_T(ENUM, VAL, NAME, NAME)

#define Start_TNamedEnum(ENUM, NAME) \
	Start_TNamedEnum_T(ENUM, NAME, NAME)
#define End_TNamedEnum(ENUM, NAME) \
	End_TNamedEnum_T(ENUM, NAME, NAME)
#endif

