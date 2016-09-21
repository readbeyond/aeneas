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
/*                                                                      */
/*              Author: Richard Caley and Paul Taylor                   */
/*                   Date: May 1997, April 98                           */
/* -------------------------------------------------------------------  */
/*               Class for windowing speech waveforms                   */
/*                                                                      */
/************************************************************************/

#ifndef __EST_WINDOW_H__
#define __EST_WINDOW_H__

#include "EST_TBuffer.h"
#include "EST_Wave.h"

/**@name Function types used for parameters to functions. 
*/
//@{
  
/// Function which creates a window.
typedef void EST_WindowFunc(int size, EST_TBuffer<float> &r_window, int window_centre );

//@}


/** The EST_Window class provides functions for the creation and use
of signal processing windows.

Signal processing algorithms often work by on small sections of the
speech waveform known as {\em frames}. A full signal must first be
divided into these frames before these algorithms can work. While it
would be simple to just "cut out" the required frames from the
waveforms, this is usually undesirable as large discontinuities can
occur at the frame edges. Instead it is customary to cut out the frame
by means of a \{em window} function, which tapers the signal in the
frame so that it has high values in the middle and low or zero values
near the frame edges. The \Ref{EST_Window} class provides a wrap
around for such windowing operations.

There are several types of window function, including:

\begin{itemize}

\item {\bf Rectangular}, which is used to give a simple copy of the the
values between the window limits. 

\[w_{n} = \left\{ \begin{array}{ll}
1 & \mbox{$0 \leq n \leq N$} \\
0 & \mbox{otherwise}
\end{array}
\right. \]

\item {\bf Hanning}. The rectangular window can cause sharp discontinuities
at window edges. The hanning window solves this by ensuring that the
window edges taper to 0.

\[w_{n} = \left\{ \begin{array}{ll}
0.5 - 0.5 \cos(2\pi n / (N-1)) & \mbox{$0 \leq n \leq N$} \\
0 & \mbox{otherwise}
\end{array}
\right. \]

\item {\bf Hamming.} The hanning window causes considerable energy
loss, which the hamming window attempts to rectify.

\[w_{n} = \left\{ \begin{array}{ll}
0.54 - 0.46 \cos(2\pi n / (N-1)) & \mbox{$0 \leq n \leq N$} \\
0 & \mbox{otherwise}
\end{array}
\right. \]

\end{itemize}

The particular choice of window depends on the application. For
instance in most speech synthesis applications Hanning windows are the
most suitable as they don't have time domain discontinuities. For
analysis applications hamming windows are normally used.


For example code, see \Ref{Windowing}


*/ 

class EST_Window {
public:
  
  /// A function which creates a window
  typedef EST_WindowFunc Func;

  /**@name Functions for making windows.

    */
  //@{
  
  /** Make a Buffer of containing a window function of specified type.
      If window_centre < 0 (default -1), then a symmetric window is
      returned. For positive values of the window_centre argument, 
      asymmetric windows are returned.
    */
  static void make_window(EST_TBuffer<float> &window_vals, int size, 
    const char *name, int window_centre);

  /** Make a EST_FVector containing a window function of specified type. 
      If window_centre < 0 (default -1), then a symmetric window is
      returned. For positive values of the window_centre argument, 
      asymmetric windows are returned.
    */
  static void make_window(EST_FVector &window_vals, int size, 
    const char *name, int window_centre);

  /// Return the creation function for the given window type.
  static Func *creator(const char *name, bool report_error = false);
  //@}

/** @name Performing windowing on a section of speech. 

  */

//@{

  /** Window the waveform {\tt sig} starting at point {\tt start} for
    a duration of {\tt size} samples. The windowing function required
    is given as a function pointer {\tt *make_window} which has
    already been created by a function such as \Ref{creator}.
    The output windowed frame is placed in the buffer {\tt frame} which
    will have been resized accordingly within the function.
    */
    
  static void window_signal(const EST_Wave &sig, 
			    EST_WindowFunc *make_window, 
			    int start, int size, 
			    EST_TBuffer<float> &frame);

  /** Window the waveform {\tt sig} starting at point {\tt start} for
    a duration of {\tt size} samples. The windowing function required
    is given as a function pointer {\tt *make_window} which has
    already been created by a function such as \Ref{creator}.
    The output windowed frame is placed in the EST_FVector {\tt frame}.
    By default, it is assumed that this is already the correct size
    (i.e. {\tt size} samples long), but if resizing is required the
    last argument should be set to 1.
    */

  static void window_signal(const EST_Wave &sig, 
			    EST_WindowFunc *make_window, 
			    int start, int size, 
			    EST_FVector &frame,int resize=0);

  /** Window the waveform {\tt sig} starting at point {\tt start} for
    a duration of {\tt size} samples. The windowing function required
    is given as a string: this function will make a temporary window
    of this type.  The output windowed frame is placed in the
    EST_FVector {\tt frame}.  By default, it is assumed that this is
    already the correct size (i.e. {\tt size} samples long), but if
    resizing is required the last argument should be set to 1.  */

  static void window_signal(const EST_Wave &sig, 
			    const EST_String &window_name, 
			    int start, int size, 
			    EST_FVector &frame, int resize=0);

  
 /** Window the waveform {\tt sig} starting at point {\tt start} for
    a duration of {\tt size} samples. The window shape required
    is given as an array of floats.  The output windowed frame is placed in the
    EST_FVector {\tt frame}.  By default, it is assumed that this is
    already the correct size (i.e. {\tt size} samples long), but if
    resizing is required the last argument should be set to 1.  */
  static void window_signal(const EST_Wave &sig, 
			    EST_TBuffer<float> &window_vals,
			    int start, int size, 
			    EST_FVector &frame, int resize=0);

//@}


/**@name Utility window functions.

*/
//@{
  /// Return the description for a given window type.
  static EST_String description(const char *name);

  /// Return a paragraph describing the available windows.
  static EST_String options_supported(void);

  /// Return a comma separated list of the available window types.
  static EST_String options_short(void);

//@}
};

///For example code, see \Ref{Windowing}.

//@see Windowing mechanisms





#endif
