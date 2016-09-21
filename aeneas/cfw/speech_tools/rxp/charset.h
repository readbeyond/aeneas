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
#ifndef CHARSET_H
#define CHARSET_H

#ifndef FOR_LT
#define STD_API
#endif

STD_API void init_charset(void);

/* 
 * We'd like char8 to be unsigned char, but it causes too many problems.
 * For example:
 *     char8 name; ...; return name ? name : "<none>"
 * produces a warning with many compilers if char8 is unsigned.
 */

typedef char char8;
typedef unsigned short char16;
typedef unsigned int char32;

#if !defined(CHAR_SIZE)
#    error CHAR_SIZE not defined
#endif

#if CHAR_SIZE == 8
typedef char8 Char;
#elif CHAR_SIZE == 16
typedef char16 Char;
#else
#error CHAR_SIZE must be 8 or 16
#endif

/* Character encodings */

enum character_encoding {
    CE_unknown, CE_unspecified_ascii_superset,
    CE_UTF_8, CE_ISO_646, 
    CE_ISO_8859_1,

    CE_ISO_8859_2, CE_ISO_8859_3, CE_ISO_8859_4, CE_ISO_8859_5,
    CE_ISO_8859_6, CE_ISO_8859_7, CE_ISO_8859_8, CE_ISO_8859_9,

    CE_UTF_16B, CE_UTF_16L, CE_ISO_10646_UCS_2B, CE_ISO_10646_UCS_2L, 
    CE_enum_count
};

typedef enum character_encoding CharacterEncoding;

extern STD_API CharacterEncoding InternalCharacterEncoding;

extern STD_API const char8 *CharacterEncodingName[CE_enum_count];
extern STD_API const char8 *CharacterEncodingNameAndByteOrder[CE_enum_count];

struct character_encoding_alias {const char8 *name; CharacterEncoding enc;};
extern STD_API struct character_encoding_alias CharacterEncodingAlias[];
extern STD_API const int CE_alias_count;

STD_API int EncodingIsAsciiSuperset(CharacterEncoding enc);
STD_API int EncodingsCompatible(CharacterEncoding enc1, CharacterEncoding enc2,
			CharacterEncoding *enc3);
STD_API CharacterEncoding FindEncoding(char8 *name);

/* Translation tables for Latin-N - do this right sometime! XXX */

extern STD_API int iso_to_unicode[8][256];
extern STD_API int iso_max_val[8];
extern STD_API char8 *unicode_to_iso[8];

#endif /* CHARSET_H */
