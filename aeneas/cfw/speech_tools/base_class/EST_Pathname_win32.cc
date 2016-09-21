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

#include <cstdio>
#include "EST_System.h"
#include "EST_Pathname.h"

void EST_Pathname::setup(void)
{
  this->gsub("/", "\\");
}

int EST_Pathname::is_absolute(void) const
{
  return length()>0 && (*this)[0] == '\\';
}

int EST_Pathname::is_dirname(void) const
{
  return length()>0 && (*this)[length()-1] == '\\';
}

EST_Pathname EST_Pathname::directory(void) const {

  if (is_dirname())
    return *this;

  int pos;
  if ((pos=index("\\", -1)) >=0)
    return before(pos+1);
  else
    return ".\\";
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
    return ((EST_String)(*this) + (EST_String)"\\");

  return ".\\";
}

EST_Pathname EST_Pathname::construct(EST_Pathname dir, 
				     EST_String filename)
{
  EST_Pathname result(dir.as_directory());

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
  WIN32_FIND_DATA find_data;
  HANDLE handle;
  EST_TList<EST_String> list;
  EST_Pathname pattern(this->as_directory() + EST_Pathname("*"));

  handle = FindFirstFile(pattern, &find_data);
  if (handle !=  INVALID_HANDLE_VALUE)
    while (1==1)
      {
	EST_Pathname name(find_data.cFileName);

	if (check_for_directories 
	    && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
	  list.append(name.as_directory());
	else
	  list.append(name);
	if (!FindNextFile(handle, &find_data))
	  break;
      }
  FindClose(handle);
  return list;
}

EST_Pathname EST_Pathname::append(EST_Pathname directory, EST_Pathname addition)
{
  if (addition.is_absolute())
    return addition;

  EST_String add(addition);

  EST_Pathname result(directory.as_directory());

  result.EST_String::operator += (add);

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
    
    if (contains("\\"))
	  result = result.after(index("\\",-1));
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

EST_Pathname operator + (const EST_Pathname p, const EST_Pathname addition) 
{return EST_Pathname::append(p, addition); }

EST_Pathname operator + (const char *p, const EST_Pathname addition) 
{return EST_Pathname::append(p, addition); }

EST_Pathname &operator += (EST_Pathname p, const EST_Pathname addition)
{ p = EST_Pathname::append(p, addition); return p; }
EST_Pathname &operator += (EST_Pathname p, const EST_String addition)
{ p = EST_Pathname::append(p, addition); return p; }

