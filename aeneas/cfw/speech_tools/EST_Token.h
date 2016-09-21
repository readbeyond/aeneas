/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                         Copyright (c) 1996                            */
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
/*                    Token/Tokenizer class                              */
/*                                                                       */
/*=======================================================================*/

#ifndef __EST_TOKEN_H__
#define __EST_TOKEN_H__

#include <cstdio>

using namespace std;

#include "EST_String.h"
#include "EST_common.h"

// I can never really remember this so we'll define it here
/// The default whitespace characters
extern const EST_String EST_Token_Default_WhiteSpaceChars;
///
extern const EST_String EST_Token_Default_SingleCharSymbols;
///
extern const EST_String EST_Token_Default_PunctuationSymbols;
///
extern const EST_String EST_Token_Default_PrePunctuationSymbols;

/** This class is similar to \Ref{EST_String} but also maintains 
    the original punctuation and whitespace found around the 
    token.  

    \Ref{EST_Token}'s primary use is with \Ref{EST_TokenStream} class 
    which allows easy tokenizing of ascii files.  

    A token consists of four parts, any of which may be empty: a
    name, the actual token, preceding whitespace, preceding
    punctuation, the name and succeeding punctuation.  

    @author Alan W Black (awb@cstr.ed.ac.uk): April 1996
*/
class EST_Token {
  private:
    EST_String space;
    EST_String prepunc;
    EST_String pname;
    EST_String punc;
    int linenum;
    int linepos;
    int p_filepos;
    int p_quoted;

  public:
    ///
    EST_Token() {init();}
    ///
    EST_Token(const EST_String p) {init(); pname = p; }
    ///
    void init() {p_quoted=linenum=linepos=p_filepos=0;}
    
    /**@name Basic access to fields */
    //@{
    /// set token from a string
    void set_token(const EST_String &p) { pname = p; }
    ///
    void set_token(const char *p) { pname = p; }
    /// set whitespace of token.
    void set_whitespace(const EST_String &p) { space = p; }
    ///
    void set_whitespace(const char *p) { space = p; }
    /// set (post) punctuation of token.
    void set_punctuation(const EST_String &p) { punc = p; }
    /// 
    void set_punctuation(const char *p) { punc = p; }
    /// set prepunction
    void set_prepunctuation(const EST_String &p) { prepunc = p; }
    ///
    void set_prepunctuation(const char *p) { prepunc = p; }
    ///
    const EST_String &whitespace() { return space; }
    ///
    const EST_String &punctuation() { return punc; }
    ///
    const EST_String &prepunctuation() { return prepunc; }

    /**@name Access token as a string */
    //@{
    const EST_String &string() const { return String(); }
    /// Access token as a string
    const EST_String &S() const { return S(); }
    /// Access token as a string
    const EST_String &String() const { return pname; }
    /// For automatic coercion to \Ref{EST_String}
    operator EST_String() const { return String(); }
    //@}

    /**@name Access token as a int */
    //@{
    int Int(bool &valid) const { return String().Int(valid); }
    int Int() const { return String().Int(); }
    int I(bool &valid) const { return Int(valid); }
    int I() const { return Int(); }
    operator int() const { return Int(); }
    //@}

    /**@name Access token as a long */
    //@{
    long Long(bool &valid) const { return String().Long(valid); }
    long Long() const { return String().Long(); }
    long L(bool &valid) const { return Long(valid); }
    long L() const { return Long(); }
    operator long() const { return Long(); }
    //@}

    /**@name Access token as a float */
    //@{
    float Float(bool &valid) const { return String().Float(valid); }
    float Float() const { return String().Float(); }
    float F(bool &valid) const { return Float(valid); }
    float F() const { return Float(); }
    operator float() const { return Float(); }
    //@}

    /**@name Access token as a double */
    //@{
    double Double(bool &valid) const { return String().Double(valid); }
    double Double() const { return String().Double(); }
    double D(bool &valid) const { return Double(valid); }
    double D() const { return Double(); }
    operator double() const { return Double(); }
    //@}

    //@}
    //@{
    /// Note that this token was quoted (or not)
    void set_quoted(int q) { p_quoted = q; }
    /// TRUE is token was quoted
    int quoted() const { return p_quoted; }
    //@}
    ///
    void set_row(int r) { linenum = r; }
    ///
    void set_col(int c) { linepos = c; }
    /// Set file position in original \Ref{EST_TokenStream}
    void set_filepos(int c) { p_filepos = c; }
    /// Return lower case version of token name
    EST_String lstring() { return downcase(pname); }
    /// Return upper case version of token name
    EST_String ustring() { return upcase(pname); }
    /// Line number in original \Ref{EST_TokenStream}.
    int row(void) const { return linenum; }
    /// Line position in original \Ref{EST_TokenStream}.
    int col(void) const { return linepos; }
    /// file position in original \Ref{EST_TokenStream}.
    int filepos(void) const { return p_filepos; }

    /// A string describing current position, suitable for error messages
    const EST_String pos_description() const;

    ///
    friend ostream& operator << (ostream& s, const EST_Token &p);
    
    ///
    EST_Token & operator = (const EST_Token &a);
    ///
    EST_Token & operator = (const EST_String &a);
    ///
    int operator == (const EST_String &a) { return (pname == a); }
    ///
    int operator != (const EST_String &a) { return (pname != a); }
    ///
    int operator == (const char *a) { return (strcmp(pname,a)==0); }
    ///
    int operator != (const char *a) { return (strcmp(pname,a)!=0); }
};

enum EST_tokenstream_type {tst_none, tst_file, tst_pipe, tst_string, tst_istream}; 

/** A class that allows the reading of \Ref{EST_Token}s from a file
    stream, pipe or string.  It automatically tokenizes a file based on
    user definable whitespace and punctuation.

    The definitions of whitespace and punctuation are user definable.
    Also support for single character symbols is included.  Single
    character symbols {\em always} are treated as individual tokens
    irrespective of their white space context.  Also a quote
    mode can be used to read uqoted tokens.

    The setting of whitespace, pre and post punctuation, single character
    symbols and quote mode must be down (immediately) after opening
    the stream.

    There is no unget but peek provides look ahead of one token.
    
    Note there is an interesting issue about what to do about
    the last whitespace in the file.  Should it be ignored or should
    it be attached to a token with a name string of length zero.
    In unquoted mode the eof() will return TRUE if the next token name
    is empty (the mythical last token).  In quoted mode the last must
    be returned so eof will not be raised.

    @author Alan W Black (awb@cstr.ed.ac.uk): April 1996
*/
class EST_TokenStream{
 private:
    EST_tokenstream_type type;
    EST_String WhiteSpaceChars;
    EST_String SingleCharSymbols;
    EST_String PunctuationSymbols;
    EST_String PrePunctuationSymbols;
    EST_String Origin;
    FILE *fp;
    istream *is;
    int fd;
    char *buffer;
    int buffer_length;
    int pos;
    int linepos;
    int p_filepos;
    int getch(void);
    EST_TokenStream &getch(char &C);
    int peeked_charp;
    int peeked_char;       // ungot character 
    int peekch(void);
    int peeked_tokp;
    int eof_flag;
    int quotes;
    char quote;
    char escape;
    EST_Token current_tok;
    void default_values(void);
    /* local buffers to save reallocating */
    int tok_wspacelen;
    char *tok_wspace;
    int tok_stufflen;
    char *tok_stuff;
    int tok_prepuncslen;
    char *tok_prepuncs;
    int close_at_end;

    /* character class map */
    char p_table[256];
    bool p_table_wrong;

    /** This function is deliberately private so that you'll get a compilation
        error if you assign a token stream or pass it as an (non-reference)
        argument.  The problem with copying is that you need to copy the
        filedescriptiors too (which can't be done for pipes).  You probably
        don't really want a copy anyway and meant to pass it as a reference.
        If you really need this (some sort of clever look ahead) I am not
        sure what he consequences really are (or how portable they are).
        Pass the \Ref{EST_TokenStream} by reference instead.
    */
    EST_TokenStream(EST_TokenStream &s);

    void build_table();

    inline int getch_internal();
    inline int peekch_internal();
    inline int getpeeked_internal();
  public:
    ///
    EST_TokenStream();
    /// will close file if appropriate for type
    ~EST_TokenStream();
    //@{
    /// open a \Ref{EST_TokenStream} for a file.
    int open(const EST_String &filename);
    /// open a \Ref{EST_TokenStream} for an already opened file
    int open(FILE *ofp, int close_when_finished);
    /// open a \Ref{EST_TokenStream} for an already open istream
    int open(istream &newis);
    /// open a \Ref{EST_TokenStream} for string rather than a file
    int open_string(const EST_String &newbuffer);
    /// Close stream.
    void close(void);
    //@}
    /**@name stream access functions */
    //@{
    /// get next token in stream
    EST_TokenStream &get(EST_Token &t);
    /// get next token in stream
    EST_Token &get();
    /**@name  get the next token which must be the argument. */
    //@{
    EST_Token &must_get(EST_String expected, bool *ok);
    EST_Token &must_get(EST_String expected, bool &ok) 
	{ return must_get(expected, &ok); }
    EST_Token &must_get(EST_String expected) 
	{ return must_get(expected, (bool *)NULL); }
    //@}
    /// get up to {\tt s} in stream as a single token.
    EST_Token get_upto(const EST_String &s);
    /// get up to {\tt s} in end of line as a single token.
    EST_Token get_upto_eoln(void);
    /// peek at next token
    EST_Token &peek(void)
    {	if (!peeked_tokp) get();
	peeked_tokp = TRUE; return current_tok; }
    /// Reading binary data, (don't use peek() immediately beforehand)
    int fread(void *buff,int size,int nitems) EST_WARN_UNUSED_RESULT;
    //@}
    /**@name stream initialization functions */
    //@{
    /// set which characters are to be treated as whitespace
    void set_WhiteSpaceChars(const EST_String &ws) 
        { WhiteSpaceChars = ws; p_table_wrong=1;}
    /// set which characters are to be treated as single character symbols
    void set_SingleCharSymbols(const EST_String &sc) 
        { SingleCharSymbols = sc; p_table_wrong=1;}
    /// set which characters are to be treated as (post) punctuation
    void set_PunctuationSymbols(const EST_String &ps) 
        { PunctuationSymbols = ps; p_table_wrong=1;}
    /// set which characters are to be treated as (post) punctuation
    void set_PrePunctuationSymbols(const EST_String &ps) 
        { PrePunctuationSymbols = ps; p_table_wrong=1;}
    /// set characters to be used as quotes and escape, and set quote mode
    void set_quotes(char q, char e) { quotes = TRUE; quote = q; escape = e; p_table_wrong=1;}
    /// query quote mode
    int quoted_mode(void) { return quotes; }
    //@}
    /**@name miscellaneous */
    //@{
    /// returns line number of \Ref{EST_TokenStream}
    int linenum(void) const {return linepos;}
    /// end of file
    int eof()
       { return (eof_flag || ((!quotes) && (peek() == ""))); }
    /// end of line
    int eoln();
    /// current file position in \Ref{EST_TokenStream}
    int filepos(void) const { return (type == tst_string) ? pos : p_filepos; }
    /// tell, synonym for filepos
    int tell(void) const { return filepos(); }
    /// seek, reposition file pointer
    int seek(int position);
    int seek_end();
    /// Reset to start of file/string 
    int restart(void);
    /// A string describing current position, suitable for error messages
    const EST_String pos_description();
    /// The originating filename (if there is one)
    const EST_String filename() const { return Origin; }
    /// For the people who *need* the actual description (if possible)
    FILE *filedescriptor() { return (type == tst_file) ? fp : 0; }
    ///
    EST_TokenStream & operator >>(EST_Token &p);
    ///
    EST_TokenStream & operator >>(EST_String &p);
    ///
    friend ostream& operator <<(ostream& s, EST_TokenStream &p);
    //@}
};

/** Quote a string with given quotes and escape character
*/
EST_String quote_string(const EST_String &s,
			const EST_String &quote = "\"", 
			const EST_String &escape = "\\", 
			int force=0);

#endif // __EST_TOKEN_H__
