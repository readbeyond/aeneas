/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                      Copyright (c) 1995,1996                          */
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
/*                    Date   :  May 1996                                 */
/*-----------------------------------------------------------------------*/
/*            A class for representing ints floats and strings           */
/*                                                                       */
/*=======================================================================*/
#ifndef __EST_CONTENTS_H__
#define __EST_CONTENTS_H__

/** A class for containing some other (arbitrary) class    
 Not general enough to call itself a run-time type system  
 Is designed to solve the problem of holding user          
 specified information.
 Keeps reference count to know when to delete contents     
                                                           
 This is done on two levels EST_Contents and Contents_Data 
*/
class EST_Content_Data{
  private:
    int refs;
    void *data;
    void (*free_func)(void *data);
  public:
    EST_Content_Data(void *d,void (*f)(void *d)) {free_func=f; data=d; refs=1;}
    ~EST_Content_Data() { free_func(data); }
    ///
	int unref() { return --refs; }
    ///
    int ref() { return ++refs; }
    ///
    int the_refs() { return refs; }
    void *contents() { return data; }
    EST_Content_Data &operator=(const EST_Content_Data &c)
    {refs = c.refs; data = c.data; free_func = c.free_func; return *this;}
};

/** More contents */

class EST_Contents{
private:
    EST_Content_Data *content_data;
    void unref_contents(void)
       { if ((content_data != 0) &&
	     (content_data->unref() == 0))
	     delete content_data;}
public:
    EST_Contents() { content_data = 0; }
    EST_Contents(void *p,void (*free_func)(void *p))
         { content_data = new EST_Content_Data(p,free_func); }
    ~EST_Contents() { unref_contents(); }
    void set_contents(void *p,void (*free_func)(void *p))
         { unref_contents(); content_data = new EST_Content_Data(p,free_func);}
    void *get_contents() const 
         {return (content_data ? content_data->contents() : 0);}
    ///
    int refs() const { return ((content_data == 0) ? 0 :
			 content_data->the_refs());}
    EST_Contents &operator=(const EST_Contents &c)
         { unref_contents();
	   content_data = c.content_data; 
	   if (content_data != 0) content_data->ref();
           return *this; }
};

#endif

