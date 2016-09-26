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


#ifndef __EST_SERVER_H__
#define __EST_SERVER_H__

#include "EST_Features.h"
#include "EST_ServiceTable.h"

/** Client-server interface. 
  *
  * An EST_Server object represents a server, it can be at either
  * end of a network connection. That is, a server process has an
  * EST_Server object representing it's wait-dispatch-answer loop,
  * while a client process has an EST_Server object which represents
  * the server process.
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_Server.h,v 1.4 2004/05/04 00:00:16 awb Exp $
  */

class EST_Server {

public:
  
  /// What type of server is this.
  enum Mode {
    /// Bizarre state
    sm_unknown = 0,
    /// Client end of the connection.
    sm_client = 1,
    /// Answer one client at a time.
    sm_sequential = 2,
    /// Answer requests from several clients, as requests arrive.
    sm_interleaved =3,
    /// For off a process for each client.
    sm_fork = 4,
    /// Multi-threaded (not implemented)
    sm_threded = 5
  };

  typedef EST_Features Args;
  typedef EST_Features Result;

  class RequestHandler
  {
  public:
    EST_Server *server;
    EST_String requestString;
    EST_String package;
    EST_String operation;
    Args args;
    Result res;

    RequestHandler();
    virtual ~RequestHandler();
    virtual EST_String process(void)=0;
  };
  
  class ResultHandler
  {
  public:
    EST_Server *server;
    EST_String resString;
    Result res;

    ResultHandler();
    virtual ~ResultHandler();
    virtual void process(void)=0;
  };

  class BufferedSocket
  {
  public:
    int s;
    int  bpos;
    int  blen;
    char *buffer;

    BufferedSocket(int socket);
    ~BufferedSocket();
    void ensure(int n);
    int read_data(void);
  };

private:
  /// Then server we are connected to.
  EST_ServiceTable::Entry p_entry; 
  void *p_serv_addr;
  int p_socket;
  BufferedSocket *p_buffered_socket;
  ostream *p_trace;
  Mode p_mode;

  void zero(void);
  void init(ostream *trace);

  void initClient(const EST_ServiceTable::Entry &e, ostream *trace);
  void initClient(EST_String name, EST_String type, ostream *trace);
  void initClient(EST_String hostname, int port, ostream *trace);

  void initServer(Mode mode, EST_String name, EST_String type, ostream *trace);

protected:
  void write(BufferedSocket &s, const EST_String string, const EST_String term = "");

  EST_String read_data(BufferedSocket &s, const EST_String end, int &eof);

  bool check_cookie(BufferedSocket &socket);

  bool process_command(BufferedSocket &socket, EST_String command, RequestHandler &handler);

  void handle_client(BufferedSocket &socket, RequestHandler &handler);

  void return_error(BufferedSocket &socket, EST_String err);

  void return_value(BufferedSocket &socket, Result &res, bool last);

  void run_sequential(RequestHandler &handler);

public:
  
  /**@name Client end constructors.
    */
  //@{
  /// Create a server connection by name, defaulting to "fringe", the default server name.
  EST_Server(EST_String name, EST_String type);

  EST_Server(EST_String name, EST_String type, ostream *trace);

  /// Create a server connection by explicitly saying where to connect to.
  EST_Server(EST_String hostname, int port);
  EST_Server(EST_String hostname, int port, ostream *trace);
  //@}

  /**@name Server end constructors
    */
  //@{
  EST_Server(Mode mode, EST_String name, EST_String type);
  EST_Server(Mode mode, EST_String name, EST_String type, ostream *trace);
  //@}
  
  /// Destroy the connection.
  virtual ~EST_Server();

  /**@name information about the server.
    */
  //@{
  /// Name of server.
  const EST_String name(void) const;
  /// Type of server.
  const EST_String type(void) const;
  /// Domain name of the server.
  const EST_String hostname(void) const;
  /// Dotted numeric IP address
  const EST_String address(void) const;
  /// Domain name or IP number
  const EST_String servername(void) const;
  /// Port number
  int port(void) const;
  //@}

  /**@name connection management
    */
  //@{
  /// Connect to the server.
  EST_connect_status connect(void);
  /// Are we connected at the moment?
  bool connected(void);
  /// Disconnect.
  EST_connect_status disconnect(void); 
  //@}

  virtual bool parse_command(const EST_String command,
			     EST_String &package,
			     EST_String &operation,
			     Args &arguments);

  virtual EST_String build_command(const EST_String package,
				   const EST_String operation,
				   const Args &arguments);

  virtual bool parse_result(const EST_String resultString,
			     Result &res);

  virtual EST_String build_result(const Result &res);
				   

  bool execute(const EST_String package,
	       const EST_String operation,
	       const Args &arguments,
	       ResultHandler &handler);

  bool execute(const EST_String command,
	       ResultHandler &handler);

  void run(RequestHandler &handler);

};

#endif

