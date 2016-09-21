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
 /*                                                                       */
 /*                 Author: Richard Caley (rjc@cstr.ed.ac.uk)             */
 /*                   Date: Fri Oct 10 1997                               */
 /* --------------------------------------------------------------------  */
 /* A subclass of TMatrix which copies using memcopy. This isn't          */
 /* suitable for matrices of class objects which have to be copied        */
 /* using a constructor or specialised assignment operator.               */
 /*                                                                       */
 /*************************************************************************/

#ifndef __EST_TSIMPLEMATRIX_H__
#define __EST_TSIMPLEMATRIX_H__

#include "EST_TMatrix.h"
#include "instantiate/EST_TSimpleMatrixI.h"

template<class T>
class EST_TSimpleMatrix : public EST_TMatrix<T> {
protected:
    
    // just copy data, no resizing.
    void copy_data(const EST_TSimpleMatrix<T> &a); 

public:
    /// default constructor
    EST_TSimpleMatrix(void) : EST_TMatrix<T>() {};
    /// size constructor
    EST_TSimpleMatrix(int m, int n) : EST_TMatrix<T>(m, n) {};
    /// copy constructor
    EST_TSimpleMatrix(const EST_TSimpleMatrix<T> &m); 

  /// copy one matrix into another
    void copy(const EST_TSimpleMatrix<T> &a); 

    /// resize matrix
    void resize(int rows, int cols, int set=1); 

    /// assignment operator
    EST_TSimpleMatrix<T> &operator=(const EST_TSimpleMatrix<T> &s); 
};

#endif
