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
#ifndef STRING16_H
#define STRING16_H

#ifndef FOR_LT
#define STD_API
#define WIN_IMP
#endif

#include "charset.h"
#include <stddef.h>		/* for size_t */

/* String functions */
#include <string.h>
#if 0
/* Don't want to include string.h while testing */

int strcmp(const char *, const char *);
WIN_IMP int strncmp(const char *, const char *, size_t);
int strcasecmp(const char *, const char *);
size_t strlen(const char *);
WIN_IMP char *strchr(const char *, int);
char *strcpy(char *, const char *);
WIN_IMP char *strncpy(char *, const char *, size_t);
char *strcat(char *, const char *);
WIN_IMP char *strstr(const char *, const char *);
int memcmp(const void *, const void *, size_t);
#ifndef memcpy
void *memcpy(void *, const void *, size_t);
#endif
void *memset(void *, int, size_t);
WIN_IMP size_t strspn(const char *, const char *);
WIN_IMP size_t strcspn(const char *, const char *);
#endif

STD_API char8 *strdup8(const char8 *s);
#define strchr8(s, c)strchr((s), c)
#define strlen8(s) strlen((s))
#define strcmp8(s1, s2) strcmp((s1), (s2))
#define strncmp8(s1, s2, n) strncmp((s1), (s2), n)
#define strcpy8(s1, s2) strcpy((s1), (s2))
#define strncpy8(s1, s2, n) strncpy((s1), (s2), n)

#define strcat8(s1, s2) strcat((s1), (s2))
STD_API int strcasecmp8(const char8 *, const char8 *);
STD_API int strncasecmp8(const char8 *, const char8 *, size_t);
#define strstr8(s1, s2) strstr(s1, s2)

STD_API char16 *strdup16(const char16 *s);
STD_API char16 *strchr16(const char16 *, int);
STD_API size_t strlen16(const char16 *);
STD_API int strcmp16(const char16 *, const char16 *);
STD_API int strncmp16(const char16 *, const char16 *, size_t);
STD_API char16 *strcpy16(char16 *, const char16 *);
STD_API char16 *strncpy16(char16 *, const char16 *, size_t);
STD_API char16 *strcat16(char16 *, const char16 *);
STD_API int strcasecmp16(const char16 *, const char16 *);
STD_API int strncasecmp16(const char16 *, const char16 *, size_t);
STD_API char16 *strstr16(const char16 *, const char16 *);

STD_API char16 *char8tochar16(const char8 *s);
STD_API char8 *char16tochar8(const char16 *s);

#if CHAR_SIZE == 8

#define Strdup strdup8
#define Strchr strchr8
#define Strlen strlen8
#define Strcmp strcmp8
#define Strncmp strncmp8
#define Strcpy strcpy8
#define Strncpy strncpy8
#define Strcat strcat8
#define Strcasecmp strcasecmp8
#define Strncasecmp strncasecmp8
#define Strstr strstr8

#define char8toChar(x) (x)
#define Chartochar8(x) (x)

#else

#define Strdup strdup16
#define Strchr strchr16
#define Strlen strlen16
#define Strcmp strcmp16
#define Strncmp strncmp16
#define Strcpy strcpy16
#define Strncpy strncpy16
#define Strcat strcat16
#define Strcasecmp strcasecmp16
#define Strncasecmp strncasecmp16
#define Strstr strstr16

#define char8toChar char8tochar16
#define Chartochar8 char16tochar8

#endif

#endif /* STRING16_H */
