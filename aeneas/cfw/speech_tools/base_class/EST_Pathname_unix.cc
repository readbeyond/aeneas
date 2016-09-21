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
 /************************************************************************/
 /*                 Author: Richard Caley (rjc@cstr.ed.ac.uk)            */
 /*                   Date: Tue Mar 18 1997                              */
 /************************************************************************/
 /*                                                                      */
 /* Implementation of a class for manipulating filenames and so on.      */
 /*                                                                      */
 /* This is all hard coded to be unix filenames. I think the best        */
 /* strategy is to have a separate version of this for any other         */
 /* pathname format rather than trying to parameterise this. Most of     */
 /* it is fairly simple.                                                 */
 /*                                                                      */
 /************************************************************************/

#include "EST_unix.h"
#include "EST_Pathname.h"

void EST_Pathname::setup(void)
{
}

int EST_Pathname::is_absolute(void) const
{
  return length()>0 && (*this)(0) == '/';
}

int EST_Pathname::is_dirname(void) const
{
  return length()>0 && (*this)(length()-1) == '/';
}

EST_Pathname EST_Pathname::directory(void) const {

  if (is_dirname())
    return *this;

  int pos;
  if ((pos=index("/", -1)) >=0)
    return before(pos+1);
  else
    return "./";
 }

EST_Pathname EST_Pathname::as_file(void) const
{
  if (is_filename())
    return *this;

  if (length() > 0)
    return before(-1);

  return ".";
}

EST_Pathname EST_Pathname::as_directory(void) const
{
  if (is_dirname())
    return *this;

  if (length() > 0)
  {
      EST_String xx;
      xx = EST_String(*this) + "/";
      return xx;
  }
  
  return "./";
}

EST_Pathname EST_Pathname::construct(EST_Pathname dir, 
				     EST_String filename)
{
  EST_String result(dir.as_directory());

  result += filename;
  return result;
}

EST_Pathname EST_Pathname::construct(EST_Pathname dir, 
				     EST_String basename, 
				     EST_String extension)
{
  EST_Pathname filename(basename + "." + extension);
  return EST_Pathname::construct(dir, filename);
}

EST_TList<EST_String> EST_Pathname::entries(int check_for_directories) const
{
  DIR *dir;
  EST_TList<EST_String> list;

  if ((dir = opendir(this->as_directory()))!=NULL)
    {
      struct dirent *entry;

      while ((entry = readdir(dir)) != NULL)
	{
	  EST_Pathname name(entry->d_name);
	  struct stat buf;

	  if (check_for_directories && 
	      stat((EST_String)this->as_directory() + (EST_String)name, &buf)==0 && 
	      (buf.st_mode & S_IFDIR))
	    list.append(name.as_directory());
	  else
	    list.append(name);
	}
      closedir(dir);
    }

  return list;
}

EST_Pathname operator + (const EST_Pathname p, const EST_Pathname addition) 
{return EST_Pathname::append(p, addition); }

EST_Pathname operator + (const char *p, const EST_Pathname addition) 
{return EST_Pathname::append(p, addition); }

#if 0
EST_Pathname operator += (EST_Pathname p, const EST_Pathname addition)
{ EST_String q = EST_Pathname::append(p, addition); return q; }
EST_Pathname operator += (EST_Pathname p, const EST_String addition)
{ EST_String q = EST_Pathname::append(p, addition); return q; }
#endif

EST_Pathname EST_Pathname::append(EST_Pathname directory, EST_Pathname addition)
{
  if (addition.is_absolute())
    return addition;

  EST_String add(addition);

  EST_String result(directory.as_directory());

  result += add;

  return result;
}


EST_String EST_Pathname::extension(void) const
{
    EST_String result("");
    
    if (length() <= 0)
	return result;
    
    if (contains("."))
	result = after(index(".",-1));

    return result;

}

EST_Pathname EST_Pathname::filename(void) const
{
  EST_String result(this->as_file());
    
    if (contains("/"))
	  result = result.after(index("/",-1));
    return result;
}

EST_String EST_Pathname::basename(int remove_all) const
{
    EST_String result(this->as_file().filename());
    
    if (remove_all)
      {
	if (result.contains("."))
	  result = result.before(".");
      }
    return result;
}
