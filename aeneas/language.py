#!/usr/bin/env python
# coding=utf-8

"""
Enumeration of the supported languages.
"""

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl (www.readbeyond.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.0.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class Language(object):
    """
    Enumeration of the supported languages.

    The language is indicated by the ISO 639-1 code (2 or 3 letters),
    which is the same as the language code accepted by ``espeak``.
    """

    BG = "bg"
    """ Bulgarian """

    CA = "ca"
    """ Catalan """

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

    ES = "es"
    """ Spanish (Castillan) """

    ET = "et"
    """ Estonian """

    FI = "fi"
    """ Finnish """

    FR = "fr"
    """ French """

    GA = "ga"
    """ Irish (Gaelic) """

    GRC = "grc"
    """ Greek (Ancient) """

    HR = "hr"
    """ Croatian """

    HU = "hu"
    """ Hungarian """

    IS = "is"
    """ Icelandic """

    IT = "it"
    """ Italian """

    LA = "la"
    """ Latin """

    LT = "lt"
    """ Lithuanian """

    LV = "lv"
    """ Latvian """

    NL = "nl"
    """ Dutch """

    NO = "no"
    """ Norwegian """

    RO = "ro"
    """ Romanian """

    RU = "ru"
    """ Russian """

    PL = "pl"
    """ Polish """

    PT = "pt"
    """ Portuguese """

    SK = "sk"
    """ Slovakian"""

    SR = "sr"
    """ Serbian """

    SV = "sv"
    """ Swedish """

    TR = "tr"
    """ Turkish """

    UK = "uk"
    """ Ukrainian """

    ALLOWED_VALUES = [
        BG,
        CA,
        CY,
        DA,
        DE,
        EL,
        EN,
        ES,
        ET,
        FI,
        FR,
        GA,
        GRC,
        HR,
        HU,
        IS,
        IT,
        LA,
        LT,
        LV,
        NL,
        NO,
        RO,
        RU,
        PL,
        PT,
        SK,
        SR,
        SV,
        TR,
        UK
    ]
    """ List of all the allowed values """



