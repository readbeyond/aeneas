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
/*                         Author :  Alan W Black                        */
/*                         Date   :  April 1996                          */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/* A Tokenize class, both for Tokens (Strings plus alpha)                */
/* EST_TokenStream for strings, FILE *, files, pipes etc                 */
/*                                                                       */
/*=======================================================================*/
#include <cstdio>
#include <iostream>
#include "EST_unix.h"
#include <cstdlib>
#include <climits>
#include <cstring>
#include "EST_math.h"
#include "EST_Token.h"
#include "EST_string_aux.h"
#include "EST_cutils.h"
#include "EST_error.h"

const EST_String EST_Token_Default_WhiteSpaceChars = " \t\n\r";
const EST_String EST_Token_Default_SingleCharSymbols = "(){}[]";
const EST_String EST_Token_Default_PrePunctuationSymbols = "\"'`({[";
const EST_String EST_Token_Default_PunctuationSymbols = "\"'`.,:;!?]})";
const EST_String Token_Origin_FD = "existing file descriptor";
const EST_String Token_Origin_Stream = "existing istream";
const EST_String Token_Origin_String = "existing string";

static EST_Regex RXanywhitespace("[ \t\n\r]");

static inline char *check_extend_str_in(char *str, int pos, int *max)
{
    // Check we are not at the end of the string, if so get some more
    // and copy the old one into the new one
    char *newstuff;
    
    if (pos >= *max)
    {
        if (pos > *max)
            *max = 2 * pos;
        else
            *max *= 2;
	newstuff = new char[*max];
	strncpy(newstuff,str,pos);
	delete [] str;
	return newstuff;
    }
    else 
	return str;
}

#define check_extend_str(STR, POS, MAX) \
	(((POS)>= *(MAX))?check_extend_str_in((STR),(POS),(MAX)):(STR))

ostream& operator<<(ostream& s, const EST_Token &p)
{
    s << "[TOKEN " << p.pname << "]";
    return s;
}


EST_Token &EST_Token::operator = (const EST_Token &a)
{
    linenum = a.linenum;
    linepos = a.linepos;
    p_filepos = a.p_filepos;
    p_quoted = a.p_quoted;
    space = a.space;
    prepunc = a.prepunc;
    pname = a.pname;
    punc = a.punc;
    return *this;
}

const EST_String EST_Token::pos_description() const
{
    return "line "+itoString(linenum)+" char "+itoString(linepos);
}

EST_Token &EST_Token::operator = (const EST_String &a)
{
    pname = a;
    return *this;
}

EST_TokenStream::EST_TokenStream()
{
    tok_wspacelen = 64;  // will grow if necessary
    tok_wspace = new char[tok_wspacelen];
    tok_stufflen = 512;  // will grow if necessary
    tok_stuff = new char[tok_stufflen];
    tok_prepuncslen = 32;  // will grow if necessary
    tok_prepuncs = new char[tok_prepuncslen];

    default_values();
}

EST_TokenStream::EST_TokenStream(EST_TokenStream &s)
{
    (void)s;

    cerr << "TokenStream: warning passing TokenStream not as reference" 
	<< endl;

    // You *really* shouldn't use this AT ALL unless you
    // fully understand its consequences, you'll be copying open
    // files and moving file pointers all over the place
    // basically *DON'T* do this, pass the stream by reference

    // Now there may be occasions when you do want to do this for example
    // when you need to do far look ahead or check point as you read
    // but they are obscure and I'm not sure how to do that for all
    // the file forms supported by the TokenStream.  If you do
    // I can write a clone function that might do it.

}

void EST_TokenStream::default_values()
{
    type = tst_none;
    peeked_tokp = FALSE;
    peeked_charp = FALSE;
    eof_flag = FALSE;
    quotes = FALSE;
    p_filepos = 0;
    linepos = 1;  
    WhiteSpaceChars = EST_Token_Default_WhiteSpaceChars;
    SingleCharSymbols = EST_String::Empty;
    PrePunctuationSymbols = EST_String::Empty;
    PunctuationSymbols = EST_String::Empty;
    build_table();
    close_at_end=TRUE;
}

EST_TokenStream::~EST_TokenStream()
{
    if (type != tst_none) 
	close();
    delete [] tok_wspace;
    delete [] tok_stuff;
    delete [] tok_prepuncs;
    
}

ostream& operator<<(ostream& s, EST_TokenStream &p)
{
    s << "[TOKENSTREAM ";
    switch (p.type)
    {
      case tst_none: 
	cerr << "UNSET"; break;
      case tst_file:
	cerr << "FILE"; break;
      case tst_pipe:
	cerr << "PIPE";	break;
      case tst_istream:
	cerr << "ISTREAM"; break;
      case tst_string:
	cerr << "STRING"; break;
      default:
	cerr << "UNKNOWN" << endl;
    }
    s << "]";
    
    return s;
}

int EST_TokenStream::open(const EST_String &filename)
{
    if (type != tst_none)
	close();
    default_values();
    fp = fopen(filename,"rb");
    if (fp == NULL)
    {
	cerr << "Cannot open file " << filename << " as tokenstream" 
	    << endl;
	return -1;
    }
    Origin = filename;
    type = tst_file;

    return 0;
}

int EST_TokenStream::open(FILE *ofp, int close_when_finished)
{
    // absorb already open stream
    if (type != tst_none)
	close();
    default_values();
    fp = ofp;
    if (fp == NULL)
    {
	cerr << "Cannot absorb NULL filestream as tokenstream" << endl;
	return -1;
    }
    Origin = Token_Origin_FD;
    type = tst_file;
    
    close_at_end = close_when_finished;
    
    return 0;
}

int EST_TokenStream::open(istream &newis)
{
    // absorb already open istream 
    if (type != tst_none)
	close();
    default_values();
    is = &newis;
    Origin = Token_Origin_Stream;
    type = tst_istream;

    return 0;
}

int EST_TokenStream::open_string(const EST_String &newbuffer)
{
    // Make a tokenstream from an internal existing string/buffer
    const char *buf;
    if (type != tst_none)
	close();
    default_values();
    buf = (const char *)newbuffer;
    buffer_length = newbuffer.length();
    buffer = new char[buffer_length+1];
    memmove(buffer,buf,buffer_length+1);
    pos = 0;
    Origin = Token_Origin_String;
    type = tst_string;

    return 0;
}

int EST_TokenStream::seek_end()
{
    // This isn't actually useful but people expect it 
    peeked_charp = FALSE;
    peeked_tokp = FALSE;

    switch (type)
    {
      case tst_none: 
	cerr << "EST_TokenStream unset" << endl;
	return -1;
	break;
      case tst_file:
	fseek(fp,0,SEEK_END);
	p_filepos = ftell(fp);
	return p_filepos;
      case tst_pipe:
	cerr << "EST_TokenStream seek on pipe not supported" << endl;
	return -1;
	break;
      case tst_istream:
    is->seekg(0,is->end);
    p_filepos = is->tellg();
	return p_filepos;
	break;
      case tst_string:
	pos = buffer_length;
	return pos;
      default:
	cerr << "EST_TokenStream: unknown type" << endl;
	return -1;
    }

    return -1;  // can't get here 
}

int EST_TokenStream::seek(int position)
{
    peeked_charp = FALSE;
    peeked_tokp = FALSE;

    switch (type)
    {
      case tst_none: 
	cerr << "EST_TokenStream unset" << endl;
	return -1;
	break;
      case tst_file:
	p_filepos = position;
	return fseek(fp,position,SEEK_SET);
      case tst_pipe:
	cerr << "EST_TokenStream seek on pipe not supported" << endl;
	return -1;
	break;
      case tst_istream:
    p_filepos = position;
    is->seekg(position, is->beg);
	return 0;
	break;
      case tst_string:
	if (position >= pos)
	{
	    pos = position;
	    return -1;
	}
	else
	{
	    pos = position;
	    return 0;
	}
	break;
      default:
	cerr << "EST_TokenStream: unknown type" << endl;
	return -1;
    }

    return -1;  // can't get here 

}

static int stdio_fread(void *buff,int size,int nitems,FILE *fp)
{
    // So it can find the stdio one rather than the TokenStream one
    return fread(buff,size,nitems,fp);
}

int EST_TokenStream::fread(void *buff, int size, int nitems)
{
    // switching into binary mode for current position
    int items_read;

    // so we can continue to read afterwards
    if (peeked_tokp)
    {
	cerr << "ERROR " << pos_description() 
	    << " peeked into binary data" << endl;
	return 0;
    }

    peeked_charp = FALSE;
    peeked_tokp = FALSE;

    switch (type)
    {
      case tst_none: 
	cerr << "EST_TokenStream unset" << endl;
	return 0;
	break;
      case tst_file:
	items_read = stdio_fread(buff,(size_t)size,(size_t)nitems,fp);
	p_filepos += items_read*size;
	return items_read;
      case tst_pipe:
	cerr << "EST_TokenStream fread pipe not yet supported" << endl;
	return 0;
	break;
      case tst_istream:
    is->read((char*)buff, (size_t) size*nitems);
	return is->gcount()/size;
    break;
      case tst_string:
	if ((buffer_length-pos)/size < nitems)
	    items_read = (buffer_length-pos)/size;
	else
	    items_read = nitems;
	memcpy(buff,&buffer[pos],items_read*size);
	pos += items_read*size;
	return items_read;
      default:
	cerr << "EST_TokenStream: unknown type" << endl;
	return EOF;
    }

    return 0;  // can't get here 

}
    
void EST_TokenStream::close(void)
{
    // close any files (if they were used)
    
    switch (type)
    {
      case tst_none: 
	break;
      case tst_file:
	if (close_at_end)
	  fclose(fp);
      case tst_pipe:
	// close(fd);
	break;
      case tst_istream:
	break;
      case tst_string:
	delete [] buffer;
	buffer = 0;
	break;
      default:
	cerr << "EST_TokenStream: unknown type" << endl;
	break;
    }

    type = tst_none;
    peeked_charp = FALSE;
    peeked_tokp = FALSE;

}

int EST_TokenStream::restart(void)
{
    // For paul, the only person I know who uses this
    
    switch (type)
    {
      case tst_none: 
	break;
      case tst_file:
        fp = freopen(Origin,"rb",fp);
	p_filepos = 0;
	break;
      case tst_pipe:
	cerr << "EST_TokenStream: can't rewind pipe" << endl;
	return -1;
	break;
      case tst_istream:
	cerr << "EST_TokenStream: can't rewind istream" << endl;
	break;
      case tst_string:
	pos = 0;
	break;
      default:
	cerr << "EST_TokenStream: unknown type" << endl;
	break;
    }

    linepos = 1;
    peeked_charp = FALSE;
    peeked_tokp = FALSE;
    eof_flag = FALSE;

    return 0;
}
	
EST_TokenStream & EST_TokenStream::operator >>(EST_Token &p)
{
    return get(p);
}
 
EST_TokenStream & EST_TokenStream::operator >>(EST_String &p)
{
    EST_Token t;

    get(t);
    p = t.string();
    return *this;
}

EST_TokenStream &EST_TokenStream::get(EST_Token &tok)
{
    tok = get();
    return *this;
}

EST_Token EST_TokenStream::get_upto(const EST_String &s)
{
    // Returns a concatenated token form here to next symbol that matches s
    // including s (though not adding s on the result)
    // Not really for the purist but lots of times very handy
    // Note this is not very efficient
    EST_String result;
    EST_Token t;

    for (result=EST_String::Empty; (t=get()) != s; )
    {
	result += t.whitespace() + t.prepunctuation() +
	    t.string() + t.punctuation();
	if (eof())
	{
	    cerr << "EST_TokenStream: end of file when looking for \"" <<
		s << "\"" << endl;
	    break;
	}
    }

    return EST_Token(result);
}

EST_Token EST_TokenStream::get_upto_eoln(void)
{
    // Swallow the lot up to end of line 
    // assumes \n is a whitespace character

    EST_String result(EST_String::Empty);

    while (!eoln())
    {
	EST_Token &t=get();
	result += t.whitespace() + t.prepunctuation();

	if (quotes)
	    result += quote_string(t.string());
	else
	    result += t.string();

	result += t.punctuation();

	if (eof())
	{
//	    cerr << "EST_TokenStream: end of file when looking for end of line"
//		<< endl;
	    break;
	}
    }
    // So that the next call works I have to step over the eoln condition
    // That involves removing the whitespace upto and including the next 
    // \n in the peek token.

    char *w = wstrdup(peek().whitespace());
    int i;
    for (i=0; w[i] != 0; i++)
	if (w[i] == '\n')   // maybe not portable 
	    peek().set_whitespace(&w[i+1]);

    wfree(w);

    static EST_Token result_t;

    result_t.set_token(result);

    return result_t;
}

EST_Token &EST_TokenStream::must_get(EST_String expected, bool *ok)
{
    EST_Token &tok = get();

    if (tok != expected)
    {
        if (ok != NULL)
        {
            *ok=FALSE;
            return tok;
        }
        else
            EST_error("Expected '%s' got '%s' at %s", 
                      (const char *)expected, 
                      (const char *)(EST_String)tok,
                      (const char *)pos_description());
    }

    if (ok != NULL)
        *ok=TRUE;
    return tok;
}

void EST_TokenStream::build_table()
{
    int i;
    const char *p;
    unsigned char c;

    for (i=0; i<256; ++i)
	p_table[i]=0;

    for (p=WhiteSpaceChars; *p; ++p)
	if (p_table[c=(unsigned char)*p])
	    EST_warning("Character '%c' has two classes, '%c' and '%c'", 
			*p, c, ' ');
	else
	    p_table[c] = ' ';

    for (p=SingleCharSymbols; *p; ++p)
	if (p_table[c=(unsigned char)*p])
	    EST_warning("Character '%c' has two classes, '%c' and '%c'", 
			*p, p_table[c], '!');
	else
	    p_table[c] = '@';

    for (p=PunctuationSymbols; *p; ++p)
	if (p_table[c=(unsigned char)*p] == '@')
	    continue;
	else if (p_table[c])
	    EST_warning("Character '%c' has two classes, '%c' and '%c'", 
			*p, p_table[c], '.');
	else
	    p_table[c] = '.';

    for(p=PrePunctuationSymbols; *p; ++p)
	if (p_table[c=(unsigned char)*p] == '@')
	    continue;
	else if (p_table[c] == '.')
	    p_table[c] = '"';
	else if (p_table[c])
	    EST_warning("Character '%c' has two classes, '%c' and '%c'", 
			*p, p_table[c], '$');
	else
	    p_table[c] = '$';

    p_table_wrong=0;
}

inline int EST_TokenStream::getpeeked_internal(void)
{
  peeked_charp = FALSE;
  return peeked_char;
}

inline
int EST_TokenStream::getch_internal()
{
    // Return next character in stream
    if (EST_TokenStream::peeked_charp)
    {
      return getpeeked_internal();
    }
    
    switch (type)
    {
      case tst_none: 
	cerr << "EST_TokenStream unset" << endl;
	return EOF;
	break;
      case tst_file:
	p_filepos++;
	{
	    char lc;
	    if (stdio_fread(&lc,1,1,fp) == 0)
		return EOF;
	    else
		return (int)lc;
	}
/*	return getc(fp); */
      case tst_pipe:
	cerr << "EST_TokenStream pipe not yet supported" << endl;
	return EOF;
	break;
      case tst_istream:
	p_filepos++;
	return is->get();
      case tst_string:
	if (pos < buffer_length)
	{
	    p_filepos++;
	    return buffer[pos++];
	}
	else
	    return EOF;
      default:
	cerr << "EST_TokenStream: unknown type" << endl;
	return EOF;
    }

    return EOF;  // can't get here 
}

int EST_TokenStream::getch(void)
{
  return getch_internal();
}

inline int EST_TokenStream::peekch_internal()
{
    // Return next character in stream (without reading it)

    if (!peeked_charp)
	peeked_char = getch_internal();
    peeked_charp = TRUE;
    return peeked_char;
}


int EST_TokenStream::peekch(void)
{
  return peekch_internal();
  
}

#define CLASS(C,CL) (p_table[(unsigned char)(C)]==(CL))

#define CLASS2(C,CL1,CL2) (p_table[(unsigned char)(C)]==(CL1)||p_table[(unsigned char)(C)]==(CL2))

EST_Token &EST_TokenStream::get(void)
{
    if (peeked_tokp)
    {
	peeked_tokp = FALSE;
	return current_tok;
    }

    if (p_table_wrong)
      build_table();

    char *word;
    int c,i,j;

    for (i=0; (CLASS(c=getch_internal(),' ') && 
	       ( c != EOF )); i++)
    {
	if (c == '\n') linepos++;
	tok_wspace = check_extend_str(tok_wspace,i,&tok_wspacelen);
	tok_wspace[i] = c;
    }
    tok_wspace[i] = '\0';

    current_tok.init();

    if (c != EOF)
    {   
	current_tok.set_filepos(p_filepos-1);

	if ((quotes) &&  // quoted strings (with escapes) are allowed
	    (c == quote))
	{
	    for (i=0; 
		 ((c = getch_internal()) != EOF)
		 ;)
	    {
		if (c == quote)
		    break;
		tok_stuff = check_extend_str(tok_stuff,i,&tok_stufflen);
		if (c == escape)
		    c = getch_internal();
		tok_stuff[i++] = c;
	    }
	    current_tok.set_quoted(TRUE);
	}
	else            // standard whitespace separated tokens
	{
	    for (i=0,tok_stuff[i++]=c; 
		 (
		  !CLASS(c,'@') &&
		  !CLASS(c=peekch_internal(),' ') && 
		  !CLASS(c,'@') &&
		  ( c != EOF )) ;)
	    {
		tok_stuff = check_extend_str(tok_stuff,i,&tok_stufflen);
		// note, we must have peeked to get here.
		tok_stuff[i++] = getpeeked_internal();
	    }
	}
	tok_stuff[i] = '\0';
	// Are there any punctuation symbols at the start?
	for (j=0; 
	     ((j < i) && CLASS2(tok_stuff[j], '$', '"'));
	     j++);
	if ((j > 0) && (j < i))  // there are
	{
	    tok_prepuncs = check_extend_str(tok_prepuncs,j+1,&tok_prepuncslen);
	    memmove(tok_prepuncs,tok_stuff,j);
	    tok_prepuncs[j] = '\0';
	    current_tok.set_prepunctuation(tok_prepuncs);
	    word=&tok_stuff[j];
	    i-=j;  // reduce size by number of prepuncs
	}
	else
	{
	    current_tok.set_prepunctuation(EST_String::Empty);
	    word = tok_stuff;
	}
	// Are there any punctuation symbols at the end
	for (j=i-1; 
	     ((j > 0) && CLASS2(word[j],'.','"'));
	     j--);
	if (word[j+1] != '\0')
	{
	    current_tok.set_punctuation(&word[j+1]);
	    word[j+1] = '\0';
	}
	else
	    current_tok.set_punctuation(EST_String::Empty);
	    
	current_tok.set_token(word);
	if (tok_wspace[0] == '\0') // feature paths will have null whitespace
	    current_tok.set_whitespace(EST_String::Empty);
	else
	    current_tok.set_whitespace(tok_wspace);
    }
    else
    {
	current_tok.set_token(EST_String::Empty);
	current_tok.set_whitespace(tok_wspace);
	current_tok.set_punctuation(EST_String::Empty);
	current_tok.set_prepunctuation(EST_String::Empty);
	eof_flag = TRUE;
    }
	
    return current_tok;
}

int EST_TokenStream::eoln(void)
{
    // This doesn't really work if there are blank lines (and you want
    // to know about them)

    if ((peek().whitespace().contains("\n")) ||	eof())
	return TRUE;
    else
	return FALSE;

}

EST_String quote_string(const EST_String &s,
			const EST_String &quote, 
			const EST_String &escape, 
			int force)
{
    // Quotes s always if force true, or iff s contains whitespace,
    // quotes or escapes force is false
    // Note quote and escape are assumed to be string of length 1
    EST_String quoted_form;
    if ((force) || 
	(s.contains(quote)) ||
	(s.contains(escape)) ||
	(s.contains(RXanywhitespace)) ||
	(s.length() == 0))
    {
	// bigger than the quoted form could ever be
	int i,j;
	char *quoted = new char[s.length()*(quote.length()+escape.length())+
		       1+quote.length()+quote.length()];
	quoted[0] = quote(0);
	for (i=1,j=0; j < s.length(); j++,i++)
	{
	    if (s(j) == quote(0))
		quoted[i++] = escape(0);
	    else if (s(j) == escape(0))
		quoted[i++] = escape(0);
	    quoted[i] = s(j);
	}
	quoted[i++] = quote(0);
	quoted[i] = '\0';
	quoted_form = quoted;
	delete [] quoted;
	return quoted_form;
    }
    else 
      return s;
}

const EST_String EST_TokenStream::pos_description()
{
    return Origin+":"+itoString(linepos);
}
