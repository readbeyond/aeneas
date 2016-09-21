/*************************************************************************/
/*                                                                       */
/* Copyright (c) 1997-98 Richard Tobin, Language Technology Group, HCRC, */
/* University of Edinburgh.                                              */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED ``AS IS'', WITHOUT WARRANTY OF ANY KIND,     */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHOR OR THE UNIVERSITY OF EDINBURGH BE LIABLE */
/* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF    */
/* CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION    */
/* WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.       */
/*                                                                       */
/*************************************************************************/
/* url.h	-- Henry Thompson
 *
 * $Header: /home/CVS/speech_tools/include/rxp/url.h,v 1.2 2001/04/04 13:11:27 awb Exp $
 */

#ifndef _URL_H
#define _URL_H

#ifndef FOR_LT
#define STD_API
#define EXPRT
#endif

#include <stdio.h>
#include "stdio16.h"
#include "charset.h"

extern STD_API char8 * EXPRT 
    url_merge(const char8 *url, const char8 *base,
	      char8 **scheme, char8 **host, int *port, char8 **path);
extern STD_API FILE16 *url_open(const char8 *url, const char8 *base, 
			    const char8 *type, char8 **merged_url);
extern STD_API char8 *EXPRT default_base_url(void);

#endif
