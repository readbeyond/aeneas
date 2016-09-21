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


#ifndef __EST_SERVICETABLE_H__
#define __EST_SERVICETABLE_H__

#include "EST_String.h"
#include "EST_rw_status.h"
#include "EST_TKVL.h"

/** A global table of known services. Used for fringe and festival
  * servers.
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_ServiceTable.h,v 1.2 2001/04/04 13:11:27 awb Exp $
  */

class EST_ServiceTable {


public:
  /** A service record.  
    *
    * This is returned by service lookup operations, it contains
    * enough information to contact the server and authenticate yourself.
    */

  class Entry
  {
  public:
    /// Name of the server.
    EST_String name;
    /// Type of server (eg `fringe')
    EST_String type;
    /// Human readable hostname.
    EST_String hostname;
    /// Numeric IP address (###.###.###.###)
    EST_String address;
    /// A random string to send as authorisation.
    EST_String cookie;
    /// TCP port number.
    int port;

    /// Create an empty entry.
    Entry();

    /// A suitable human readable name for the entry.
    operator EST_String() const;
    
    /// All entries are taken to be different.
    friend bool operator == (const Entry &a, const Entry &b);

    /// Print in human readable form.
    friend ostream & operator << (ostream &s, const Entry &a);
  };

private:

  class EntryTable 
	{
	public:
        EST_TKVL<EST_String, Entry> t; 
        typedef EST_TKVL<EST_String, Entry>::Entries Entries;
        typedef EST_TKVL<EST_String, Entry>::KeyEntries KeyEntries;
	};

  /// Table of available Fringe servers.
  static EntryTable entries;

  static bool random_init;
  static void init_random(void);

public:

  /**@name Finding What Services Are Available.
    *
    * Servers maintain a per-user file which lists the  processes
    * which are running in server mode by name. These functions read
    * that table.  */
  //@{

  /// Read the users default table. <filename>~/.estServices</filename>
  static void read_table(void);
  /// Read a specific table.
  static void read_table(EST_String socketsFileName);
  /// Write the users default table. <filename>~/.estServices</filename>
  static void write_table(void);
  /// Write a specific table.
  static void write_table(EST_String socketsFileName);
  /// List the table to given stream
  static void list(ostream &s, const EST_String type);
  /// Return a list of server names.
  static void names(EST_TList<EST_String> &names, const EST_String type="");

  //@}

  /** Return the entry for the server with the given name and type.
    * If no such entry is found a dummy entry with a port of 0 is returned.
    */
  static const Entry &lookup(const EST_String name, 
			     const EST_String type);

  /** Create an entry for a server of the given name and type which is
    * listening on the given socket.
    */
  static const Entry &create(const EST_String name, 
			     const EST_String type,
			     int socket);

};

#endif

