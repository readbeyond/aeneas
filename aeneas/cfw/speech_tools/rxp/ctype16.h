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
#ifndef CTYPE16_H
#define CTYPE16_H

#ifndef FOR_LT
#define STD_API
#endif

/* XML character types */

STD_API void init_ctype16(void);
STD_API int Toupper(int c);
STD_API int Tolower(int c);

extern STD_API unsigned char xml_char_map[];

#define xml_legal      0x01
#define xml_namestart  0x02
#define xml_namechar   0x04
#define xml_whitespace 0x08

#if CHAR_SIZE == 8

/* And with 0xff so that it works if char is signed */
#define is_xml_legal(c) (xml_char_map[(int)(c) & 0xff] & xml_legal)
#define is_xml_namestart(c) (xml_char_map[(int)(c) & 0xff] & xml_namestart)
#define is_xml_namechar(c) (xml_char_map[(int)(c) & 0xff] & xml_namechar)
#define is_xml_whitespace(c) (xml_char_map[(int)(c) & 0xff] & xml_whitespace)

#else

/* Note!  these macros evaluate their argument more than once! */

#define is_xml_legal(c) (c < 0x110000 && (c >= 0x10000 || (xml_char_map[c] & xml_legal)))
#define is_xml_namestart(c) (c < 0x10000 && (xml_char_map[c] & xml_namestart))
#define is_xml_namechar(c) (c < 0x10000 && (xml_char_map[c] & xml_namechar))
#define is_xml_whitespace(c) (c < 0x10000 && (xml_char_map[c] & xml_whitespace))

#endif

#endif /* CTYPE16_H */
