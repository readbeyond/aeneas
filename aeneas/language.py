#!/usr/bin/env python
# coding=utf-8

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2016, Alberto Pettarin (www.albertopettarin.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
This module contains the following classes:

* :class:`~aeneas.language.Language`, an enumeration of the supported languages.
"""


class Language(object):
    """
    Enumeration of the supported languages.

    A language is supported by ``aeneas`` if at least one of the built-in
    TTS engine wrappers supports it.
    Note that each TTS engine wrapper supports only a subset
    of the languages listed below.

    Consult the documentation of your TTS engine wrapper to
    see the list of languages supported by it:

    * :class:`~aeneas.ttswrappers.espeakttswrapper.ESPEAKTTSWrapper` (default TTS)
    * :class:`~aeneas.ttswrappers.espeakngttswrapper.ESPEAKNGTTSWrapper`
    * :class:`~aeneas.ttswrappers.festivalttswrapper.FESTIVALTTSWrapper`
    * :class:`~aeneas.ttswrappers.nuancettswrapper.NuanceTTSWrapper`

    Each language is indicated by its ISO 639-3 language code.
    """

    AFR = "afr"
    """ Afrikaans """

    AMH = "amh"
    """ Amharic """

    ARA = "ara"
    """ Arabic """

    ARG = "arg"
    """ Aragonese """

    ASM = "asm"
    """ Assamese """

    AZE = "aze"
    """ Azerbaijani """

    BEN = "ben"
    """ Bengali """

    BOS = "bos"
    """ Bosnian """

    BUL = "bul"
    """ Bulgarian """

    CAT = "cat"
    """ Catalan """

    CES = "ces"
    """ Czech """

    CMN = "cmn"
    """ Mandarin Chinese """

    CYM = "cym"
    """ Welsh """

    DAN = "dan"
    """ Danish """

    DEU = "deu"
    """ German """

    ELL = "ell"
    """ Greek (Modern) """

    ENG = "eng"
    """ English """

    EPO = "epo"
    """ Esperanto """

    EST = "est"
    """ Estonian """

    EUS = "eus"
    """ Basque """

    FAS = "fas"
    """ Persian """

    FIN = "fin"
    """ Finnish """

    FRA = "fra"
    """ French """

    GLA = "gla"
    """ Scottish Gaelic """

    GLE = "gle"
    """ Irish """

    GLG = "glg"
    """ Galician """

    GRC = "grc"
    """ Greek (Ancient) """

    GRN = "grn"
    """ Guarani """

    GUJ = "guj"
    """ Gujarati """

    HEB = "heb"
    """ Hebrew """

    HIN = "hin"
    """ Hindi """

    HRV = "hrv"
    """ Croatian """

    HUN = "hun"
    """ Hungarian """

    HYE = "hye"
    """ Armenian """

    INA = "ina"
    """ Interlingua """

    IND = "ind"
    """ Indonesian """

    ISL = "isl"
    """ Icelandic """

    ITA = "ita"
    """ Italian """

    JBO = "jbo"
    """ Lojban """

    JPN = "jpn"
    """ Japanese """

    KAL = "kal"
    """ Greenlandic """

    KAN = "kan"
    """ Kannada """

    KAT = "kat"
    """ Georgian """

    KIR = "kir"
    """ Kirghiz """

    KOR = "kor"
    """ Korean """

    KUR = "kur"
    """ Kurdish """

    LAT = "lat"
    """ Latin """

    LAV = "lav"
    """ Latvian """

    LFN = "lfn"
    """ Lingua Franca Nova """

    LIT = "lit"
    """ Lithuanian """

    MAL = "mal"
    """ Malayalam """

    MAR = "mar"
    """ Marathi """

    MKD = "mkd"
    """ Macedonian """

    MLT = "mlt"
    """ Maltese """

    MSA = "msa"
    """ Malay """

    MYA = "mya"
    """ Burmese """

    NAH = "nah"
    """ Nahuatl """

    NEP = "nep"
    """ Nepali """

    NLD = "nld"
    """ Dutch """

    NOR = "nor"
    """ Norwegian """

    ORI = "ori"
    """ Oriya """

    ORM = "orm"
    """ Oromo """

    PAN = "pan"
    """ Panjabi """

    PAP = "pap"
    """ Papiamento """

    POL = "pol"
    """ Polish """

    POR = "por"
    """ Portuguese """

    RON = "ron"
    """ Romanian """

    RUS = "rus"
    """ Russian """

    SIN = "sin"
    """ Sinhala """

    SLK = "slk"
    """ Slovak """

    SLV = "slv"
    """ Slovenian """

    SPA = "spa"
    """ Spanish """

    SQI = "sqi"
    """ Albanian """

    SRP = "srp"
    """ Serbian """

    SWA = "swa"
    """ Swahili """

    SWE = "swe"
    """ Swedish """

    TAM = "tam"
    """ Tamil """

    TAT = "tat"
    """ Tatar """

    TEL = "tel"
    """ Telugu """

    THA = "tha"
    """ Thai """

    TSN = "tsn"
    """ Tswana """

    TUR = "tur"
    """ Turkish """

    UKR = "ukr"
    """ Ukrainian """

    URD = "urd"
    """ Urdu """

    VIE = "vie"
    """ Vietnamese """

    YUE = "yue"
    """ Yue Chinese """

    ZHO = "zho"
    """ Chinese """

    ALLOWED_VALUES = [
        AFR,
        AMH,
        ARA,
        ARG,
        ASM,
        AZE,
        BEN,
        BOS,
        BUL,
        CAT,
        CES,
        CMN,
        CYM,
        DAN,
        DEU,
        ELL,
        ENG,
        EPO,
        EST,
        EUS,
        FAS,
        FIN,
        FRA,
        GLE,
        GLG,
        GRC,
        GRN,
        GUJ,
        HEB,
        HIN,
        HRV,
        HUN,
        HYE,
        INA,
        IND,
        ISL,
        ITA,
        JBO,
        JPN,
        KAL,
        KAN,
        KAT,
        KIR,
        KOR,
        KUR,
        LAT,
        LAV,
        LFN,
        LIT,
        MAL,
        MAR,
        MKD,
        MLT,
        MSA,
        MYA,
        NAH,
        NEP,
        NLD,
        NOR,
        ORI,
        ORM,
        PAN,
        PAP,
        POL,
        POR,
        RON,
        RUS,
        SIN,
        SLK,
        SLV,
        SPA,
        SQI,
        SRP,
        SWA,
        SWE,
        TAM,
        TAT,
        TEL,
        THA,
        TSN,
        TUR,
        UKR,
        URD,
        VIE,
        YUE,
        ZHO,
    ]
    """ List of all the allowed values """
