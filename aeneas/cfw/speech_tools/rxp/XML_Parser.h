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


#ifndef __XML_PARSER_H__
#define __XML_PARSER_H__

#if !defined(CHAR_SIZE)
#    define CHAR_SIZE 8
#endif

#if (CHAR_SIZE!=8)
#   error EST can only handle 8 bit characters
#endif

#include "EST_String.h"
#include "EST_Regex.h"
#include "EST_TKVL.h"
#include "EST_THash.h"
#include "EST_TDeque.h"
#include "EST_TList.h"
#include "rxp/rxp.h"

// We only use types and functions from rxp.h, so we can throw away
// some of the macros which cause problems.

#undef get


/**@name XML Parser
  * Recursive descent parsing skeleton with hooks for processing.
  * A C++ wrapper around the rxp parser.
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: XML_Parser.h,v 1.3 2004/05/04 00:00:17 awb Exp $
  */
//@{

class XML_Parser;
class XML_Parser_Class;

/// Nice name for list of attribute-value pairs.
typedef EST_TStringHash<EST_String> XML_Attribute_List;

/** A Class of parsers, All parsers share callbacks and a
  * list of known public IDs.
  */
class XML_Parser_Class {

private:

  /** Map PUBLIC and SYSTEM IDs to places on the local system.
    */
  EST_TKVL<EST_Regex, EST_String> known_ids;

protected:
  /** Do any necessary remappings and open a stream which reads the given
    * entity.
    */
  static InputSource open_entity(Entity ent, void *arg);
  

  /**@name The callbacks.
    * 
    * These methods can be overridden in a subclass to create a class
    * of parsers to do whatever you want.
    */
  //@{

  /** Called when starting a document.
    */
  virtual void document_open(XML_Parser_Class &c,
			XML_Parser &p,
			void *data);

  /** Called at the end of a document.
    */
  virtual void document_close(XML_Parser_Class &c,
			 XML_Parser &p,
			 void *data);
  
  /** Called when an element starts.
    */
  virtual void element_open(XML_Parser_Class &c,
		       XML_Parser &p,
		       void *data,
		       const char *name,
		       XML_Attribute_List &attributes);

  /** Called when an element ends.
    */
  virtual void element_close(XML_Parser_Class &c,
			XML_Parser &p,
			void *data,
			const char *name);

  /** Called for empty elements.
    *
    * Defaults to element_open(...) followed by element_closed(...).
    */
  virtual void element(XML_Parser_Class &c,
		  XML_Parser &p,
		  void *data,
		  const char *name,
		  XML_Attribute_List &attributes);

  /** Called for parsed character data sequences.
    */
  virtual void pcdata(XML_Parser_Class &c,
		 XML_Parser &p,
		 void *data,
		 const char *chars);
  /** Called for unparsed character data sequences.
    */
  virtual void cdata(XML_Parser_Class &c,
		XML_Parser &p,
		void *data,
		const char *chars);

  /** Called for processing directives.
    */
  virtual void processing(XML_Parser_Class &c,
		     XML_Parser &p,
		     void *data,
		     const char *instruction);

  /** Called when there is an error in parsing.
    */
  virtual void error(XML_Parser_Class &c,
		XML_Parser &p,
		void *data);
  //@}

  /** This can be called from any of the callbacks to present "message"
    * as an error through the error callback, thus getting filename and
    * line information into the message.
    */
  void error(XML_Parser_Class &c,
	     XML_Parser &p,
	     void *data,
	     EST_String message);

  /// Get the error message for the last error.
  const char *get_error(XML_Parser &p);

public:

  /** Create an object representing the class of parsers. 
    */
  XML_Parser_Class();

  virtual ~XML_Parser_Class() { }

  /** Add a mapping from entity ID (SYSTEM or PUBLIC) to filename.
    * 
    * The string can contain escapes like \2 which are replaced by
    * the text matching the Nth bracketed part of the regular expression.
    */
  void register_id(EST_Regex id_pattern, EST_String directory);

  /** Fill in the list with the known entity ID mappings.
    */

  void registered_ids(EST_TList<EST_String> &list);

  /**@name Creating a parser
    * 
    * Each of these methods creates a one-shot parser which will run over the
    * indicated text.
    */
  //@{

  /// Create a parser for the RXP InputSource.
  XML_Parser *make_parser(InputSource source, void *data);

  /// Create a parser for the RXP InputSource.
  XML_Parser *make_parser(InputSource source, Entity initial_entity, void *data);

  /// Create a parser for a stdio input stream.
  XML_Parser *make_parser(FILE *input, void *data);

  /** Create a parser for a stdio input stream, giving  a description for
    * use in errors.
    */
  XML_Parser *make_parser(FILE *input, const EST_String desc, void *data);

  // Create a parser for the named file.
  XML_Parser *make_parser(const EST_String filename, void *data);

  //@}

  /** Utility which tries to open an entity called ID at places
    * specified in the mapping of this parser class.
    */

  InputSource try_and_open(Entity ent);
  
  /** XML_Parser defines the behaviour of an individual one-shot
    * parser.
    */
  friend class XML_Parser;
};

/** An actual parser. Each such instance parses just one stream which is
  * given when the parser is created. 
  *
  * The behaviour of the parser is given by the class to which it belongs.
  */

class XML_Parser {

private:
  /// Last error message from the parser.
  EST_String p_error_message;

  /// Set true when context is being remembered.
  bool p_track_context;

  /// Set true when contents is being remembered. (not yet implemented)
  bool p_track_contents;

protected:
  /** The class to which this parser belongs. Defines the behaviour of
    * the parser.
    */
  XML_Parser_Class *pclass;

  /// The piece of markup being processed.
  XBit current_bit;

  /// Where we are reading from.
  InputSource source;

  /** The entity we started from. May need to be freed at the end of the
    * parse.
    */
  Entity initial_entity;

  /// Arbitrary data which can be used by callbacks.
  void *data;

  /// The RXP parser object.
  Parser p;

  /// If context is being tracked, this is a stack of element names.
  EST_TDeque<EST_String> p_context;


  /// Creator used by XML_Parser_Class::make_parser()
  XML_Parser(XML_Parser_Class &parent, 
	     InputSource source, 
	     Entity initial_entity,
	     void *data);

  /// Open. Asks the parser class to do the work.
  InputSource open(Entity ent);

  /// Get the error message for the last error.
  const char *get_error();

public:

  /// Destructor, may close input if required.
  ~XML_Parser();

  /** Request that parser keep track of the currently open elements.
    * 
    * These are recorded on a atsck. Use context() to access the information.
    */
  void track_context(bool flag);
  /** Keep track of the content of open elements.
    *
    * Not yet implemented.
    */
  void track_contents(bool flag);

  /** Get the name of the nth enclosing element.
    * 
    * context(0) is the element we are directly inside.
    */
  EST_String context(int n);

  /// Run the parser. 
  void go();

  friend class XML_Parser_Class;
};

//@}

#endif

