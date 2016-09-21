/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                         Copyright (c) 1999                            */
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
/*                    Author :  Alan W Black                             */
/*                    Date   :  March 1999                               */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/* Macros definitions for defining anything as vals                      */
/*                                                                       */
/*=======================================================================*/
#ifndef __EST_VAL_DEFS_H__
#define __EST_VAL_DEFS_H__

/* Macro for defining new class as values public functions */
#define VAL_REGISTER_CLASS_DCLS(NAME,CLASS)            \
extern val_type val_type_##NAME;                       \
class CLASS *NAME(const EST_Val &v);                   \
EST_Val est_val(const class CLASS *v);

/* For things that aren't classes (typed def something else) */
#define VAL_REGISTER_TYPE_DCLS(NAME,CLASS)             \
extern val_type val_type_##NAME;                       \
CLASS *NAME(const EST_Val &v);                         \
EST_Val est_val(const CLASS *v);

#define VAL_REGISTER_FUNCPTR_DCLS(NAME,TYPE)           \
extern val_type val_type_##NAME;                       \
TYPE NAME(const EST_Val &v);                           \
EST_Val est_val(const TYPE v);


/* Macro for defining new class as values             */
#define VAL_REGISTER_CLASS(NAME,CLASS)                 \
val_type val_type_##NAME=#NAME;                        \
class CLASS *NAME(const EST_Val &v)                    \
{                                                      \
    if (v.type() == val_type_##NAME)                   \
	return (class CLASS *)v.internal_ptr();        \
    else                                               \
	EST_error("val not of type val_type_"#NAME);   \
    return NULL;                                       \
}                                                      \
                                                       \
static void val_delete_##NAME(void *v)                 \
{                                                      \
    delete (class CLASS *)v;                           \
}                                                      \
                                                       \
EST_Val est_val(const class CLASS *v)                  \
{                                                      \
    return EST_Val(val_type_##NAME,                    \
		   (void *)v,val_delete_##NAME);       \
}                                                      \

/* Macro for defining new typedef'd things as vals       */
/* You don't need CLASS and TYPE but it often convenient */
#define VAL_REGISTER_TYPE(NAME,CLASS)                  \
val_type val_type_##NAME=#NAME;                        \
CLASS *NAME(const EST_Val &v)                          \
{                                                      \
    if (v.type() == val_type_##NAME)                   \
	return (CLASS *)v.internal_ptr();              \
    else                                               \
	EST_error("val not of type val_type_"#NAME);   \
    return NULL;                                       \
}                                                      \
                                                       \
static void val_delete_##NAME(void *v)                 \
{                                                      \
    delete (CLASS *)v;                                 \
}                                                      \
                                                       \
EST_Val est_val(const CLASS *v)                        \
{                                                      \
    return EST_Val(val_type_##NAME,                    \
		   (void *)v,val_delete_##NAME);       \
}                                                      \

/* Macro for defining new typedef'd things as vals that don't get deleted */
/* You don't need CLASS and TYPE but it often convenient */
#define VAL_REGISTER_TYPE_NODEL(NAME,CLASS)            \
val_type val_type_##NAME=#NAME;                        \
CLASS *NAME(const EST_Val &v)                          \
{                                                      \
    if (v.type() == val_type_##NAME)                   \
	return (CLASS *)v.internal_ptr();              \
    else                                               \
	EST_error("val not of type val_type_"#NAME);   \
    return NULL;                                       \
}                                                      \
                                                       \
static void val_delete_##NAME(void *v)                 \
{                                                      \
    (void)v;                                           \
}                                                      \
                                                       \
EST_Val est_val(const CLASS *v)                        \
{                                                      \
    return EST_Val(val_type_##NAME,                    \
		   (void *)v,val_delete_##NAME);       \
}                                                      \

/* Macro for defining new class as values             */
#define VAL_REGISTER_CLASS_NODEL(NAME,CLASS)           \
val_type val_type_##NAME=#NAME;                        \
class CLASS *NAME(const EST_Val &v)                    \
{                                                      \
    if (v.type() == val_type_##NAME)                   \
	return (class CLASS *)v.internal_ptr();        \
    else                                               \
	EST_error("val not of type val_type_"#NAME);   \
    return NULL;                                       \
}                                                      \
                                                       \
static void val_delete_##NAME(void *v)                 \
{                                                      \
    (void)v;                                           \
}                                                      \
                                                       \
EST_Val est_val(const class CLASS *v)                  \
{                                                      \
    return EST_Val(val_type_##NAME,                    \
		   (void *)v,val_delete_##NAME);       \
}                                                      \

/* Macro for defining function pointers as values      */
#define VAL_REGISTER_FUNCPTR(NAME,CLASS)               \
val_type val_type_##NAME=#NAME;                        \
CLASS NAME(const EST_Val &v)                           \
{                                                      \
    if (v.type() == val_type_##NAME)                   \
	return (CLASS)v.internal_ptr();                \
    else                                               \
	EST_error("val not of type val_type_"#NAME);   \
    return NULL;                                       \
}                                                      \
                                                       \
static void val_delete_##NAME(void *v)                 \
{                                                      \
    (void)v;                                           \
}                                                      \
                                                       \
EST_Val est_val(const CLASS v)                         \
{                                                      \
    return EST_Val(val_type_##NAME,                    \
		   (void *)v,val_delete_##NAME);       \
}                                                      \



#endif 
