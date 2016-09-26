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

#ifndef __MODULEDESCRIPTION_H__
#define __MODULEDESCRIPTION_H__

#include <cstdio>
#include <iostream>

using namespace std;

#include "EST_String.h"

/** Machine readable descriptions of modules. Useful for help messages
  * and for verifying that a  set of modules should work together.
  *
  * This is a struct rather than a class so that it can be initialised
  * in the source of the module. 
  * @author Richard Caley <rjc@cstr,ed,ac,uk>
  * @version $Id: ModuleDescription.h,v 1.4 2014/11/25 14:32:04 robert Exp $
  */

struct ModuleDescription {

  /**@name limits */
//@{
/// Number of lines of descriptive text.
#define MD_MAX_DESCRIPTION_LINES (10)
/// Space for input streams.
#define MD_MAX_INPUT_STREAMS (5)
/// Space for optional streams.
#define MD_MAX_OPTIONAL_STREAMS (5)
/// Space for output streams.
#define MD_MAX_OUTPUT_STREAMS (5)
/// Space for parameters.
#define MD_MAX_PARAMETERS (10)
//@}

/**@name Parameter types
  * Use these for types to avoid typoes and to allow for a cleverer system
  * at a later date.
  */
//@{
/// t or nil
#define mpt_bool	"BOOL"
/// Positive integer
#define mpt_natnum	"NATNUM"
/// Integer
#define mpt_int		"INT"
/// Floating point number
#define mpt_float	"FLOAT"
/// Any string
#define mpt_string	"STRING"
/// A UnitDatabase
#define mpt_unitdatabase "UNITDATABASE"
/// Anything
#define mpt_other	"OTHER"
//@}

  /// name of module
  const char * name;		
  /// version number of module
  float version;		
  /// where it comes from
  const char * organisation;	
  /// person(s) responsible
  const char * author;		

  /// general description
  const char * description[MD_MAX_DESCRIPTION_LINES];

  /// streams affected.
  struct stream_parameter {
    /// default stream name
    const char * name;		
    /// what itis used for
    const char * description;	
  };

  /// Streams which must have values when the module is called.
  struct stream_parameter input_streams[MD_MAX_INPUT_STREAMS];
  /// Streams which may or not be defined.
  struct stream_parameter optional_streams[MD_MAX_OPTIONAL_STREAMS];
  /// Streams which will be defined after the module has run.
  struct stream_parameter output_streams[MD_MAX_OUTPUT_STREAMS];

  /// Record for a parameter.
  struct parameter {
    /// Name of parameter
    const char * name;
    /// Type of value.
    const char * type;
    /// Default value assumed.
    const char * default_val;
    /// Human readable description of effect.
    const char * description;
  };
  /// Parameters which effect the module.
  struct parameter parameters[MD_MAX_PARAMETERS];

  /// Create human readable string from description.
  static EST_String to_string(const ModuleDescription &desc);
  /// Create a module description, initialising it properly.
  static struct ModuleDescription *create();
  /// Print the description to the strream.
  static ostream &print(ostream &s, const ModuleDescription &desc);

  static int print(FILE *s, const ModuleDescription &desc);

};

/// Output operator for descriptions.
ostream &operator << (ostream &stream, const ModuleDescription &desc);

//VAL_REGISTER_CLASS_DCLS(moddesc,ModuleDescription) // clang/llvm complains about this (Rob)
//SIOD_REGISTER_CLASS_DCLS(moddesc,ModuleDescription)
VAL_REGISTER_TYPE_DCLS(moddesc,ModuleDescription)
SIOD_REGISTER_TYPE_DCLS(moddesc,ModuleDescription)

#endif
