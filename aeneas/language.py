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

    * :class:`~aeneas.ttswrappers.awsttswrapper.AWSTTSWrapper`
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

    CODE_TO_HUMAN = {
        AFR: u"Afrikaans",
        AMH: u"Amharic",
        ARA: u"Arabic",
        ARG: u"Aragonese",
        ASM: u"Assamese",
        AZE: u"Azerbaijani",
        BEN: u"Bengali",
        BOS: u"Bosnian",
        BUL: u"Bulgarian",
        CAT: u"Catalan",
        CES: u"Czech",
        CMN: u"Mandarin Chinese",
        CYM: u"Welsh",
        DAN: u"Danish",
        DEU: u"German",
        ELL: u"Greek (Modern)",
        ENG: u"English",
        EPO: u"Esperanto",
        EST: u"Estonian",
        EUS: u"Basque",
        FAS: u"Persian",
        FIN: u"Finnish",
        FRA: u"French",
        GLA: u"Scottish Gaelic",
        GLE: u"Irish",
        GLG: u"Galician",
        GRC: u"Greek (Ancient)",
        GRN: u"Guarani",
        GUJ: u"Gujarati",
        HEB: u"Hebrew",
        HIN: u"Hindi",
        HRV: u"Croatian",
        HUN: u"Hungarian",
        HYE: u"Armenian",
        INA: u"Interlingua",
        IND: u"Indonesian",
        ISL: u"Icelandic",
        ITA: u"Italian",
        JBO: u"Lojban",
        JPN: u"Japanese",
        KAL: u"Greenlandic",
        KAN: u"Kannada",
        KAT: u"Georgian",
        KIR: u"Kirghiz",
        KOR: u"Korean",
        KUR: u"Kurdish",
        LAT: u"Latin",
        LAV: u"Latvian",
        LFN: u"Lingua Franca Nova",
        LIT: u"Lithuanian",
        MAL: u"Malayalam",
        MAR: u"Marathi",
        MKD: u"Macedonian",
        MLT: u"Maltese",
        MSA: u"Malay",
        MYA: u"Burmese",
        NAH: u"Nahuatl",
        NEP: u"Nepali",
        NLD: u"Dutch",
        NOR: u"Norwegian",
        ORI: u"Oriya",
        ORM: u"Oromo",
        PAN: u"Panjabi",
        PAP: u"Papiamento",
        POL: u"Polish",
        POR: u"Portuguese",
        RON: u"Romanian",
        RUS: u"Russian",
        SIN: u"Sinhala",
        SLK: u"Slovak",
        SLV: u"Slovenian",
        SPA: u"Spanish",
        SQI: u"Albanian",
        SRP: u"Serbian",
        SWA: u"Swahili",
        SWE: u"Swedish",
        TAM: u"Tamil",
        TAT: u"Tatar",
        TEL: u"Telugu",
        THA: u"Thai",
        TSN: u"Tswana",
        TUR: u"Turkish",
        UKR: u"Ukrainian",
        URD: u"Urdu",
        VIE: u"Vietnamese",
        YUE: u"Yue Chinese",
        ZHO: u"Chinese",
    }
    """ Map from language code to human-readable name """

    ALLOWED_VALUES = sorted(CODE_TO_HUMAN.keys())
    """ List of all the allowed values """

    CODE_TO_HUMAN_LIST = sorted([u"%s\t%s" % (k, v) for k, v in CODE_TO_HUMAN.items()])
    """ List of all language codes with their human-readable names """
