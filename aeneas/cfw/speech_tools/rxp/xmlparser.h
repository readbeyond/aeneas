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
/* 	$Id: xmlparser.h,v 1.2 2001/04/04 13:11:27 awb Exp $    */

#ifndef XMLPARSER_H
#define XMLPARSER_H

#include "dtd.h"
#include "input.h"

/* Typedefs */

typedef struct parser_state *Parser;
typedef struct attribute *Attribute;
typedef struct content_particle *ContentParticle;
typedef struct xbit *XBit;
typedef void CallbackProc(XBit bit, void *arg);
typedef InputSource EntityOpenerProc(Entity e, void *arg);

/* Bits */

enum xbit_type {
    XBIT_dtd,
    XBIT_start, XBIT_empty, XBIT_end, XBIT_eof, XBIT_pcdata,
    XBIT_pi, XBIT_comment, XBIT_cdsect, XBIT_xml, 
    XBIT_error, XBIT_warning, XBIT_none,
    XBIT_enum_count
};
typedef enum xbit_type XBitType;

extern XML_API const char8 *XBitTypeName[XBIT_enum_count];

struct attribute {
    AttributeDefinition definition; /* The definition of this attribute */
    Char *value;		/* The (possibly normalised) value */
    int quoted;			/* Was it quoted? */
    struct attribute *next;	/* The next attribute or null */
};

enum cp_type {
    CP_pcdata, CP_name, CP_seq, CP_choice
};
typedef enum cp_type CPType;

struct content_particle {
    enum cp_type type;
    int repetition;
    Char *name;
    int nchildren;
    struct content_particle **children;
};

struct xbit {
    Entity entity;
    int byte_offset;
    enum xbit_type type;
    char8 *s1, *s2;
    Char *S1, *S2;
    int i1, i2;
    Attribute attributes;
    ElementDefinition element_definition;
#ifndef FOR_LT
    int nchildren;
    struct xbit *parent;
    struct xbit **children;
#endif
};

#define pcdata_chars S1

#define pi_name S1
#define pi_chars S2

#define comment_chars S1

#define cdsect_chars S1

#define xml_version s1
#define xml_encoding_name s2
#define xml_standalone i1
#define xml_encoding i2

#define error_message s1

/* Parser flags */

enum parser_flag {
    ExpandCharacterEntities,
    ExpandGeneralEntities,
    XMLPiEnd,
    XMLEmptyTagEnd,
    XMLPredefinedEntities,
    ErrorOnUnquotedAttributeValues,
    NormaliseAttributeValues,
    NormalizeAttributeValues,
    ErrorOnBadCharacterEntities,
    ErrorOnUndefinedEntities,
    ReturnComments,
    CaseInsensitive,
    ErrorOnUndefinedElements,
    WarnOnUndefinedElements,
    ErrorOnUndefinedAttributes,
    WarnOnUndefinedAttributes,
    WarnOnRedefinitions,
    TrustSDD,
    XMLExternalIDs,
    ReturnDefaultedAttributes,
    MergePCData,
    XMLMiscWFErrors,
    XMLStrictWFErrors,
    AllowMultipleElements,
    CheckEndTagsMatch,
    IgnoreEntities,
    XMLLessThan
};
typedef enum parser_flag ParserFlag;

/* Parser */

enum parse_state 
    {PS_prolog1, PS_prolog2, PS_body, PS_epilog, PS_end, PS_error};

struct element_info {
    ElementDefinition definition;
    Entity entity;
};

struct parser_state {
    enum parse_state state;
    Entity document_entity;
    int have_dtd;		/* True if dtd has been processed */
    StandaloneDeclaration standalone;
    struct input_source *source;
    Char *name, *pbuf;
    int namelen, pbufsize, pbufnext;
    struct xbit xbit;
    int peeked;
    Dtd dtd;			/* The document's DTD */
    CallbackProc *dtd_callback;
    CallbackProc *warning_callback;
    EntityOpenerProc *entity_opener;
    unsigned int flags;
    struct element_info *element_stack;
    int element_stack_alloc;
    int element_depth;
    void *callback_arg;
    int external_pe_depth;	/* To keep track of whether we're in the */
				/* internal subset: 0 <=> yes */
};

XML_API int ParserInit(void);
XML_API Parser NewParser(void);
XML_API void FreeParser(Parser p);

XML_API Entity ParserRootEntity(Parser p);
XML_API InputSource ParserRootSource(Parser p);

XML_API XBit ReadXBit(Parser p);
XML_API XBit PeekXBit(Parser p);
XML_API void FreeXBit(XBit xbit);

#ifndef FOR_LT
XBit ReadXTree(Parser p);
void FreeXTree(XBit tree);
#endif

XML_API XBit ParseDtd(Parser p, Entity e);

XML_API void ParserSetWarningCallback(Parser p, CallbackProc cb);
XML_API void ParserSetDtdCallback(Parser p, CallbackProc cb);
XML_API void ParserSetEntityOpener(Parser p, EntityOpenerProc opener);
XML_API void ParserSetCallbackArg(Parser p, void *arg);

XML_API int ParserPush(Parser p, InputSource source);
XML_API void ParserPop(Parser p);

XML_API void ParserSetFlag(Parser p,  ParserFlag flag, int value);
#define ParserGetFlag(p, flag) ((p)->flags & (1 << (flag)))

XML_API void ParserPerror(Parser p, XBit bit);

#endif /* XMLPARSER_H */
