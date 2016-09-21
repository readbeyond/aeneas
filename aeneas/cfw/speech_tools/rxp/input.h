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
#ifndef INPUT_H
#define INPUT_H

#ifndef FOR_LT
#define XML_API
#endif

#include <stdio.h>
#include "charset.h"
#include "stdio16.h"
#include "dtd.h"

/* Typedefs */

typedef struct input_source *InputSource;
typedef struct stream_ops *StreamOps;
typedef int ReadProc(StreamOps ops, unsigned char *buf, int max_count);
typedef int WriteProc(StreamOps ops, unsigned char *buf, int count);
typedef void CloseProc(StreamOps ops);
typedef int SeekProc(StreamOps ops, int offset);

/* Input sources */

XML_API InputSource SourceFromStream(const char8 *description, FILE *file);
XML_API InputSource EntityOpen(Entity e);
XML_API InputSource NewInputSource(Entity e, FILE16 *f16);
XML_API int SourceTell(InputSource s);
XML_API int SourceSeek(InputSource s, int offset);
XML_API int SourceLineAndChar(InputSource s, int *linenum, int *charnum);
XML_API void SourcePosition(InputSource s, Entity *entity, int *char_number);
XML_API int get_with_fill(InputSource s);
XML_API void determine_character_encoding(InputSource s);

struct input_source {
    Entity entity;		/* The entity from which the source reads */

    FILE16 *file16;

    Char *line;
    int line_alloc, line_length;
    int next;

    int seen_eoe;
    int complicated_utf8_line;
    int bytes_consumed;
    int bytes_before_current_line;
    int line_end_was_cr;

    int line_number;
    int not_read_yet;

    struct input_source *parent;

    int nextin;
    int insize;
    unsigned char inbuf[4096];
};

/* EOE used to be -2, but that doesn't work if Char is signed char */
#define XEOE (-999)

#define at_eol(s) ((s)->next == (s)->line_length)
#define get(s)    (at_eol(s) ? get_with_fill(s) : (s)->line[(s)->next++])
#define unget(s)  ((s)->seen_eoe ? (s)->seen_eoe= 0 : (s)->next--)

#endif /* INPUT_H */
