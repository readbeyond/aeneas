#!/usr/bin/env python
# coding=utf-8

"""
Enumeration of the supported languages.
"""

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.3.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class Language(object):
    """
    Enumeration of the supported languages.

    The language is indicated by the ISO 639-1 code (2 or 3 letters),
    which is the same as the language code accepted by ``espeak``.
    """

    AF = "af"
    """ Afrikaans (not tested) """

    AN = "an"
    """ Aragonese (not tested) """

    BG = "bg"
    """ Bulgarian """

    BS = "bs"
    """ Bosnian (not tested) """

    CA = "ca"
    """ Catalan """

    CS = "cs"
    """ Czech (not tested) """

    CY = "cy"
    """ Welsh """

    DA = "da"
    """ Danish """

    DE = "de"
    """ German """

    EL = "el"
    """ Greek (Modern) """

    EN = "en"
    """ English """

    EO = "eo"
    """ Esperanto (not tested) """

    ES = "es"
    """ Spanish (Castillan) """

    ET = "et"
    """ Estonian """

    FA = "fa"
    """ Persian """

    FI = "fi"
    """ Finnish """

    FR = "fr"
    """ French """

    GA = "ga"
    """ Irish (Gaelic) """

    GRC = "grc"
    """ Greek (Ancient) """

    HI = "hi"
    """ Hindi (not tested) """

    HR = "hr"
    """ Croatian """

    HU = "hu"
    """ Hungarian """

    HY = "hy"
    """ Armenian (not tested) """

    ID = "id"
    """ Indonesian (not tested) """

    IS = "is"
    """ Icelandic """

    IT = "it"
    """ Italian """

    JBO = "jbo"
    """ Lojban (not tested) """

    KA = "ka"
    """ Georgian (not tested) """

    KN = "kn"
    """ Kannada (not tested) """

    KU = "ku"
    """ Kurdish (not tested) """

    LA = "la"
    """ Latin """

    LFN = "lfn"
    """ Lingua Franca Nova (not tested) """

    LT = "lt"
    """ Lithuanian """

    LV = "lv"
    """ Latvian """

    MK = "mk"
    """ Macedonian (not tested) """

    ML = "ml"
    """ Malayalam (not tested) """

    MS = "ms"
    """ Malay (not tested) """

    NE = "ne"
    """ Nepali (not tested) """

    NL = "nl"
    """ Dutch """

    NO = "no"
    """ Norwegian """

    PA = "pa"
    """ Punjabi (not tested) """

    PL = "pl"
    """ Polish """

    PT = "pt"
    """ Portuguese """

    RO = "ro"
    """ Romanian """

    RU = "ru"
    """ Russian """

    SQ = "sq"
    """ Albanian (not tested) """

    SK = "sk"
    """ Slovakian"""

    SR = "sr"
    """ Serbian """

    SV = "sv"
    """ Swedish """

    SW = "sw"
    """ Swahili """

    TA = "ta"
    """ Tamil (not tested) """

    TR = "tr"
    """ Turkish """

    UK = "uk"
    """ Ukrainian """

    VI = "vi"
    """ Vietnam (not tested) """

    ZH = "zh"
    """ Mandarin (not tested) """

    ALLOWED_VALUES = [
        AF,
        AN,
        BG,
        BS,
        CA,
        CS,
        CY,
        DA,
        DE,
        EL,
        EN,
        EO,
        ES,
        ET,
        FA,
        FI,
        FR,
        GA,
        GRC,
        HI,
        HR,
        HU,
        HY,
        ID,
        IS,
        IT,
        JBO,
        KA,
        KN,
        KU,
        LA,
        LFN,
        LT,
        LV,
        MK,
        ML,
        MS,
        NE,
        NL,
        NO,
        PA,
        PL,
        PT,
        RO,
        RU,
        SK,
        SQ,
        SR,
        SV,
        SW,
        TA,
        TR,
        UK,
        VI,
        ZH
    ]
    """ List of all the allowed values """



