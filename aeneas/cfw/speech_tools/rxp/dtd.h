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
#ifndef DTD_H
#define DTD_H

#ifndef FOR_LT
#define XML_API
#endif

#include "charset.h"

/* Typedefs */

typedef struct dtd *Dtd;

typedef struct entity *Entity;

typedef struct element_definition *ElementDefinition;

typedef struct attribute_definition *AttributeDefinition;
AttributeDefinition NextAttributeDefinition(ElementDefinition element,
					    AttributeDefinition previous);

typedef struct notation_definition *NotationDefinition;

/* DTDs */

struct dtd {
    const Char *name;		/* The doctype name */
    Entity internal_part, external_part;
    Entity entities;
    Entity parameter_entities;
    Entity predefined_entities;
#ifdef FOR_LT
    NSL_Doctype_I *doctype;
#else
    ElementDefinition elements;
#endif
    NotationDefinition notations;
};

/* Entities */

enum entity_type {ET_external, ET_internal};
typedef enum entity_type EntityType;

enum markup_language {ML_xml, ML_nsl, ML_unspecified};
typedef enum markup_language MarkupLanguage;

enum standalone_declaration {
    /* NB must match NSL's rmdCode */
    SDD_unspecified, SDD_no, SDD_yes, SDD_enum_count
};
typedef enum standalone_declaration StandaloneDeclaration;

extern const char8 *StandaloneDeclarationName[SDD_enum_count];


struct entity {
    /* All entities */

    const Char *name;		/* The name in the entity declaration */
    EntityType type;		/* ET_external or ET_internal */
    const char8 *base_url;	/* If different from expected */
    struct entity *next;	/* For chaining a document's entity defns */
    CharacterEncoding encoding;	/* The character encoding of the entity */
    Entity parent;		/* The entity in which it is defined */
    const char8 *url;		/* URL of entity */

    /* Internal entities */

    const Char *text;		/* Text of the entity */
    int line_offset;		/* Line offset of definition */
    int line1_char_offset;	/* Char offset on first line */ 
    int matches_parent_text;	/* False if might contain expanded PEs */

    /* External entities */

    const char8 *systemid;	/* Declared public ID */
    const char8 *publicid;	/* Declared public ID */
    NotationDefinition notation; /* Binary entity's declared notation */
    MarkupLanguage ml_decl;	/* XML, NSL or not specified */
    const char8 *version_decl;	/* XML declarations found in entity, if any  */
    CharacterEncoding encoding_decl;
    StandaloneDeclaration standalone_decl;
    const char8 *ddb_filename;	/* filename in NSL declaration */
};

/* Elements */

enum content_type {
    /* NB this must match NSL's ctVals */
    CT_mixed, CT_any, CT_bogus1, CT_bogus2, CT_empty, CT_element, CT_enum_count
};
typedef enum content_type ContentType;

extern XML_API const char8 *ContentTypeName[CT_enum_count];

struct element_definition {
    const Char *name;		/* The element name */
    int namelen;
    int tentative;
#ifdef FOR_LT
    NSL_Doctype_I *doctype;
    NSL_ElementSummary_I *elsum;
#else
    ContentType type;		/* The declared content */
    Char *content;		/* Element content */
    AttributeDefinition attributes;
    struct element_definition *next;
#endif
};

/* Attributes */

enum default_type {
    /* NB this must match NSL's NSL_ADefType */
    DT_required, DT_bogus1, DT_implied, 
    DT_bogus2, DT_none, DT_fixed, DT_enum_count
};
typedef enum default_type DefaultType;

extern XML_API const char8 *DefaultTypeName[DT_enum_count];

enum attribute_type {
    /* NB this must match NSL's NSL_Attr_Dec_Value */
    AT_cdata, AT_bogus1, AT_bogus2, AT_nmtoken, AT_bogus3, AT_entity,
    AT_idref, AT_bogus4, AT_bogus5, AT_nmtokens, AT_bogus6, AT_entities,
    AT_idrefs, AT_id, AT_notation, AT_enumeration, AT_enum_count
};
typedef enum attribute_type AttributeType;

extern XML_API const char8 *AttributeTypeName[AT_enum_count];

struct attribute_definition {
#ifdef FOR_LT
    /* NB this must match NSL's AttributeSummary */
    /* We never really have one of these structures; only an AttributeSummary
       cast to this type.  We need to be able to access the type, so that
       we can normalise if appropriate.  We need to be able to refer to
       the default_type and default_value, but these don't need to work
       since we will never have ReturnDefaultedAttributes true in NSL. */
    int a, b, c;
    short d;
    char type, default_type;
    /* This had better never be accessed! */
    Char *default_value;
#else
    const Char *name;		/* The attribute name */
    int namelen;
    AttributeType type;		/* The declared type */
    Char **allowed_values;	/* List of allowed values, argv style */
    DefaultType default_type;	/* The type of the declared default */
    const Char *default_value;	/* The declared default value */
    struct attribute_definition *next;
#endif
};

/* Notations */

struct notation_definition {
    const Char *name;		/* The notation name */
    int tentative;
    const char8 *systemid;	/* System identifier */
    const char8 *publicid;	/* Public identifier */
    struct notation_definition *next;
};

/* Public functions */

XML_API Dtd NewDtd(void);
XML_API void FreeDtd(Dtd dtd);

XML_API Entity NewExternalEntityN(const Char *name, int namelen,
			  const char8 *publicid, const char8 *systemid,
			  NotationDefinition notation,
			  Entity parent);
XML_API Entity NewInternalEntityN(const Char *name, int namelen,
			  const Char *text, Entity parent,
			  int line_offset, int line1_char_offset, 
			  int matches_parent_text);
XML_API void FreeEntity(Entity e);

XML_API const char8 *EntityURL(Entity e);
XML_API const char8 *EntityDescription(Entity e);
XML_API void EntitySetBaseURL(Entity e, const char8 *url);
XML_API const char8 *EntityBaseURL(Entity e);

XML_API Entity DefineEntity(Dtd dtd, Entity entity, int pe);
XML_API Entity FindEntityN(Dtd dtd, const Char *name, int namelen, int pe);

#define NewExternalEntity(name, pub, sys, nnot, parent) \
    NewExternalEntityN(name, name ? Strlen(name) : 0, pub, sys, nnot, parent)
#define NewInternalEntity(name, test, parent, l, l1, mat) \
    NewInternalEntityN(name, name ? Strlen(name) : 0, test, parent, l, l1, mat)
#define FindEntity(dtd, name, pe) FindEntityN(dtd, name, Strlen(name), pe)

XML_API ElementDefinition DefineElementN(Dtd dtd, const Char *name, int namelen,
				 ContentType type, Char *content);
XML_API ElementDefinition TentativelyDefineElementN(Dtd dtd, 
					    const Char *name, int namelen);
XML_API ElementDefinition RedefineElement(ElementDefinition e, ContentType type,
				  Char *content);
XML_API ElementDefinition FindElementN(Dtd dtd, const Char *name, int namelen);
XML_API void FreeElementDefinition(ElementDefinition e);

#define DefineElement(dtd, name, type, content) \
    DefineElementN(dtd, name, Strlen(name), type, content)
#define TentativelyDefineElement(dtd, name) \
    TentativelyDefineElementN(dtd, name, Strlen(name))
#define FindElement(dtd, name) FindElementN(dtd, name, Strlen(name))

XML_API AttributeDefinition DefineAttributeN(ElementDefinition element,
				     const Char *name, int namelen,
				     AttributeType type, Char **allowed_values,
				     DefaultType default_type, 
				     const Char *default_value);
XML_API AttributeDefinition FindAttributeN(ElementDefinition element,
				   const Char *name, int namelen);
XML_API void FreeAttributeDefinition(AttributeDefinition a);

#define DefineAttribute(element, name, type, all, dt, dv) \
    DefineAttributeN(element, name, Strlen(name), type, all, dt, dv)
#define FindAttribute(element, name) \
    FindAttributeN(element, name, Strlen(name))

XML_API NotationDefinition DefineNotationN(Dtd dtd, const Char *name, int namelen,
				 const char8 *publicid, const char8 *systemid);
XML_API NotationDefinition TentativelyDefineNotationN(Dtd dtd,
					      const Char *name, int namelen);
XML_API NotationDefinition RedefineNotation(NotationDefinition n,
				 const char8 *publicid, const char8 *systemid);
XML_API NotationDefinition FindNotationN(Dtd dtd, const Char *name, int namelen);
XML_API void FreeNotationDefinition(NotationDefinition n);

#define DefineNotation(dtd, name, pub, sys) \
    DefineNotationN(dtd, name, Strlen(name), pub, sys)
#define TentativelyDefineNotation(dtd, name) \
    TentativelyDefineNotationN(dtd, name, Strlen(name))
#define FindNotation(dtd, name) FindNotationN(dtd, name, Strlen(name))

#endif /* DTD_H */
