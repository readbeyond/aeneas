#!/usr/bin/env python
# coding=utf-8

"""
Enumeration of the supported languages.
"""

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.4.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class Language(object):
    """
    Enumeration of the supported languages.

    The language is indicated by the language code accepted by ``espeak``,
    which is the ISO 639-1 code (2 or 3 letters) for most languages.
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

    EN_GB = "en-gb"
    """ English (British) """

    EN_SC = "en-sc"
    """ English (Scottish) (not tested) """

    EN_UK_NORTH = "en-uk-north"
    """ English (Northern) (not tested) """

    EN_UK_RP = "en-uk-rp"
    """ English (Received Pronunciation) (not tested) """

    EN_UK_WMIDS = "en-uk-wmids"
    """ English (Midlands) (not tested) """

    EN_US = "en-us"
    """ English (US) """

    EN_WI = "en-wi"
    """ English (West Indies) (not tested) """

    EO = "eo"
    """ Esperanto (not tested) """

    ES = "es"
    """ Spanish (Castillan) """

    ES_LA = "es-la"
    """ Spanish (Latin America) (not tested) """

    ET = "et"
    """ Estonian """

    FA = "fa"
    """ Persian """

    FA_PIN = "fa-pin"
    """ Persian (Pinglish) """

    FI = "fi"
    """ Finnish """

    FR = "fr"
    """ French """

    FR_BE = "fr-be"
    """ French (Belgium) (not tested) """

    FR_FR = "fr-fr"
    """ French (France) """

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

    HY_WEST = "hy-west"
    """ Armenian (West) (not tested) """

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

    PT_BR = "pt-br"
    """ Portuguese (Brazil) (not tested) """

    PT_PT = "pt-pt"
    """ Portuguese (Portugal) """

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

    VI_HUE = "vi-hue"
    """ Vietnam (hue) (not tested) """

    VI_SGN = "vi-sgn"
    """ Vietnam (sgn) (not tested) """

    ZH = "zh"
    """ Mandarin (not tested) """

    ZH_YUE = "zh-yue"
    """ Cantonese (not tested) """

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
        EN_GB,
        EN_SC,
        EN_UK_NORTH,
        EN_UK_RP,
        EN_UK_WMIDS,
        EN_US,
        EN_WI,
        EO,
        ES,
        ES_LA,
        ET,
        FA,
        FA_PIN,
        FI,
        FR,
        FR_BE,
        FR_FR,
        GA,
        GRC,
        HI,
        HR,
        HU,
        HY,
        HY_WEST,
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
        PT_BR,
        PT_PT,
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
        VI_HUE,
        VI_SGN,
        ZH,
        ZH_YUE
    ]
    """ List of all the allowed values """



