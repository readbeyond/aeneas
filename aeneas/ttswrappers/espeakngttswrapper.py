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

* :class:`~aeneas.ttswrappers.espeakngttswrapper.ESPEAKNGTTSWrapper`,
  a wrapper for the ``eSpeak-ng`` TTS engine.

Please refer to
https://github.com/espeak-ng/espeak-ng/
for further details.
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.exacttiming import TimeValue
from aeneas.language import Language
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.ttswrappers.basettswrapper import BaseTTSWrapper
import aeneas.globalfunctions as gf


class ESPEAKNGTTSWrapper(BaseTTSWrapper):
    """
    A wrapper for the ``eSpeak-ng`` TTS engine.

    This wrapper supports calling the TTS engine
    via ``subprocess``.

    Future support for calling via Python C extension
    is planned.

    In abstract terms, it performs one or more calls like ::

        $ espeak-ng -v voice_code -w /tmp/output_file.wav < text

    To use this TTS engine, specify ::

        "tts=espeak-ng"

    in the ``RuntimeConfiguration`` object.
    To execute from a non-default location: ::

        "tts=espeak-ng|tts_path=/path/to/espeak-ng"

    See :class:`~aeneas.ttswrappers.basettswrapper.BaseTTSWrapper`
    for the available functions.
    Below are listed the languages supported by this wrapper.

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    AFR = Language.AFR
    """ Afrikaans """

    AMH = Language.AMH
    """ Amharic (not tested) """

    ARG = Language.ARG
    """ Aragonese (not tested) """

    ASM = Language.ASM
    """ Assamese (not tested) """

    AZE = Language.AZE
    """ Azerbaijani (not tested) """

    BEN = Language.BEN
    """ Bengali (not tested) """

    BOS = Language.BOS
    """ Bosnian (not tested) """

    BUL = Language.BUL
    """ Bulgarian """

    CAT = Language.CAT
    """ Catalan """

    CES = Language.CES
    """ Czech """

    CMN = Language.CMN
    """ Mandarin Chinese (not tested) """

    CYM = Language.CYM
    """ Welsh """

    DAN = Language.DAN
    """ Danish """

    DEU = Language.DEU
    """ German """

    ELL = Language.ELL
    """ Greek (Modern) """

    ENG = Language.ENG
    """ English """

    EPO = Language.EPO
    """ Esperanto (not tested) """

    EST = Language.EST
    """ Estonian """

    EUS = "eus"
    """ Basque (not tested) """

    FAS = Language.FAS
    """ Persian """

    FIN = Language.FIN
    """ Finnish """

    FRA = Language.FRA
    """ French """

    GLA = Language.GLA
    """ Scottish Gaelic (not tested) """

    GLE = Language.GLE
    """ Irish """

    GRC = Language.GRC
    """ Greek (Ancient) """

    GRN = Language.GRN
    """ Guarani (not tested) """

    GUJ = Language.GUJ
    """ Gujarati (not tested) """

    HIN = Language.HIN
    """ Hindi (not tested) """

    HRV = Language.HRV
    """ Croatian """

    HUN = Language.HUN
    """ Hungarian """

    HYE = Language.HYE
    """ Armenian (not tested) """

    INA = Language.INA
    """ Interlingua (not tested) """

    IND = Language.IND
    """ Indonesian (not tested) """

    ISL = Language.ISL
    """ Icelandic """

    ITA = Language.ITA
    """ Italian """

    JBO = Language.JBO
    """ Lojban (not tested) """

    KAL = Language.KAL
    """ Greenlandic (not tested) """

    KAN = Language.KAN
    """ Kannada (not tested) """

    KAT = Language.KAT
    """ Georgian (not tested) """

    KIR = Language.KIR
    """ Kirghiz (not tested) """

    KUR = Language.KUR
    """ Kurdish (not tested) """

    LAT = Language.LAT
    """ Latin """

    LAV = Language.LAV
    """ Latvian """

    LFN = Language.LFN
    """ Lingua Franca Nova (not tested) """

    LIT = Language.LIT
    """ Lithuanian """

    MAL = Language.MAL
    """ Malayalam (not tested) """

    MAR = Language.MAR
    """ Marathi (not tested) """

    MKD = Language.MKD
    """ Macedonian (not tested) """

    MLT = Language.MLT
    """ Maltese (not tested) """

    MSA = Language.MSA
    """ Malay (not tested) """

    MYA = Language.MYA
    """ Burmese (not tested) """

    NAH = Language.NAH
    """ Nahuatl (not tested) """

    NEP = Language.NEP
    """ Nepali (not tested) """

    NLD = Language.NLD
    """ Dutch """

    NOR = Language.NOR
    """ Norwegian """

    ORI = Language.ORI
    """ Oriya (not tested) """

    ORM = Language.ORM
    """ Oromo (not tested) """

    PAN = Language.PAN
    """ Panjabi (not tested) """

    PAP = Language.PAP
    """ Papiamento (not tested) """

    POL = Language.POL
    """ Polish """

    POR = Language.POR
    """ Portuguese """

    RON = Language.RON
    """ Romanian """

    RUS = Language.RUS
    """ Russian """

    SIN = Language.SIN
    """ Sinhala (not tested) """

    SLK = Language.SLK
    """ Slovak """

    SLV = Language.SLV
    """ Slovenian (not tested) """

    SPA = Language.SPA
    """ Spanish """

    SQI = Language.SQI
    """ Albanian (not tested) """

    SRP = Language.SRP
    """ Serbian """

    SWA = Language.SWA
    """ Swahili """

    SWE = Language.SWE
    """ Swedish """

    TAM = Language.TAM
    """ Tamil (not tested) """

    TAT = Language.TAT
    """ Tatar (not tested) """

    TEL = Language.TEL
    """ Telugu (not tested) """

    TSN = Language.TSN
    """ Tswana (not tested) """

    TUR = Language.TUR
    """ Turkish """

    UKR = Language.UKR
    """ Ukrainian """

    URD = Language.URD
    """ Urdu (not tested) """

    VIE = Language.VIE
    """ Vietnamese (not tested) """

    YUE = Language.YUE
    """ Yue Chinese (not tested) """

    ZHO = Language.ZHO
    """ Chinese (not tested) """

    ENG_GBR = "eng-GBR"
    """ English (GB) """

    ENG_SCT = "eng-SCT"
    """ English (Scotland) (not tested) """

    ENG_USA = "eng-USA"
    """ English (USA) """

    SPA_ESP = "spa-ESP"
    """ Spanish (Castillan) """

    FRA_BEL = "fra-BEL"
    """ French (Belgium) (not tested) """

    FRA_FRA = "fra-FRA"
    """ French (France) """

    POR_BRA = "por-bra"
    """ Portuguese (Brazil) (not tested) """

    POR_PRT = "por-prt"
    """ Portuguese (Portugal) """

    AF = "af"
    """ Afrikaans """

    AN = "an"
    """ Aragonese (not tested) """

    AM = "am"
    """ Amharic (not tested) """

    AS = "as"
    """ Assamese (not tested) """

    AZ = "az"
    """ Azerbaijani (not tested) """

    BG = "bg"
    """ Bulgarian """

    BN = "bn"
    """ Bengali (not tested) """

    BS = "bs"
    """ Bosnian (not tested) """

    CA = "ca"
    """ Catalan """

    CS = "cs"
    """ Czech """

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
    """ English (GB) """

    EN_GB_SCOTLAND = "en-gb-scotland"
    """ English (Scotland) (not tested) """

    EN_GB_X_GBCLAN = "en-gb-x-gbclan"
    """ English (Northern) (not tested) """

    EN_GB_X_GBCWMD = "en-gb-x-gbcwmd"
    """ English (Midlands) (not tested) """

    EN_GB_X_RP = "en-gb-x-rp"
    """ English (Received Pronunciation) (not tested) """

    EN_US = "en-us"
    """ English (USA) """

    EN_029 = "en-029"
    """ English (West Indies) (not tested) """

    EO = "eo"
    """ Esperanto (not tested) """

    ES = "es"
    """ Spanish (Castillan) """

    ES_419 = "es-419"
    """ Spanish (Latin America) (not tested) """

    ET = "et"
    """ Estonian """

    EU = "eu"
    """ Basque (not tested) """

    FA = "fa"
    """ Persian """

    FA_LATN = "fa-Latn"
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
    """ Irish """

    GD = "gd"
    """ Scottish Gaelic (not tested) """

    GN = "gn"
    """ Guarani (not tested) """

    # NOTE already defined
    # COMMENTED GRC = "grc"
    # COMMENTED """ Greek (Ancient) """

    GU = "gu"
    """ Gujarati (not tested) """

    HI = "hi"
    """ Hindi (not tested) """

    HR = "hr"
    """ Croatian """

    HU = "hu"
    """ Hungarian """

    HY = "hy"
    """ Armenian (not tested) """

    HY_AREVMDA = "hy-arevmda"
    """ Armenian (West) (not tested) """

    IA = "ia"
    """ Interlingua (not tested) """

    ID = "id"
    """ Indonesian (not tested) """

    IS = "is"
    """ Icelandic """

    IT = "it"
    """ Italian """

    # NOTE already defined
    # COMMENTED JBO = "jbo"
    # COMMENTED """ Lojban (not tested) """

    KA = "ka"
    """ Georgian (not tested) """

    KL = "kl"
    """ Greenlandic (not tested) """

    KN = "kn"
    """ Kannada (not tested) """

    KU = "ku"
    """ Kurdish (not tested) """

    KY = "ky"
    """ Kirghiz (not tested) """

    LA = "la"
    """ Latin """

    # NOTE already defined
    # COMMENTED LFN = "lfn"
    # COMMENTED """ Lingua Franca Nova (not tested) """

    LT = "lt"
    """ Lithuanian """

    LV = "lv"
    """ Latvian """

    MK = "mk"
    """ Macedonian (not tested) """

    ML = "ml"
    """ Malayalam (not tested) """

    MR = "mr"
    """ Marathi (not tested) """

    MS = "ms"
    """ Malay (not tested) """

    MT = "mt"
    """ Maltese (not tested) """

    MY = "my"
    """ Burmese (not tested) """

    NCI = "nci"
    """ Nahuatl (not tested) """

    NE = "ne"
    """ Nepali (not tested) """

    NL = "nl"
    """ Dutch """

    NO = "no"
    """ Norwegian """

    OM = "om"
    """ Oromo (not tested) """

    OR = "or"
    """ Oriya (not tested) """

    PA = "pa"
    """ Panjabi (not tested) """

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

    SI = "si"
    """ Sinhala (not tested) """

    SK = "sk"
    """ Slovak """

    SL = "sl"
    """ Slovenian (not tested) """

    SQ = "sq"
    """ Albanian (not tested) """

    SR = "sr"
    """ Serbian """

    SV = "sv"
    """ Swedish """

    SW = "sw"
    """ Swahili """

    TA = "ta"
    """ Tamil (not tested) """

    TE = "te"
    """ Telugu (not tested) """

    TN = "tn"
    """ Tswana (not tested) """

    TR = "tr"
    """ Turkish """

    TT = "tt"
    """ Tatar (not tested) """

    UK = "uk"
    """ Ukrainian """

    UR = "ur"
    """ Urdu (not tested) """

    VI = "vi"
    """ Vietnamese (not tested) """

    VI_VN_X_CENTRAL = "vi-vn-x-central"
    """ Vietnamese (hue) (not tested) """

    VI_VN_X_SOUTH = "vi-vn-x-south"
    """ Vietnamese (sgn) (not tested) """

    ZH = "zh"
    """ Mandarin Chinese (not tested) """

    ZH_YUE = "zh-yue"
    """ Yue Chinese (not tested) """

    CODE_TO_HUMAN = {
        AFR: u"Afrikaans",
        AMH: u"Amharic (not tested)",
        ARG: u"Aragonese (not tested)",
        ASM: u"Assamese (not tested)",
        AZE: u"Azerbaijani (not tested)",
        BEN: u"Bengali (not tested)",
        BOS: u"Bosnian (not tested)",
        BUL: u"Bulgarian",
        CAT: u"Catalan",
        CES: u"Czech",
        CMN: u"Mandarin Chinese (not tested)",
        CYM: u"Welsh",
        DAN: u"Danish",
        DEU: u"German",
        ELL: u"Greek (Modern)",
        ENG: u"English",
        EPO: u"Esperanto (not tested)",
        EST: u"Estonian",
        EUS: u"Basque (not tested)",
        FAS: u"Persian",
        FIN: u"Finnish",
        FRA: u"French",
        GLA: u"Scottish Gaelic (not tested)",
        GLE: u"Irish",
        GRC: u"Greek (Ancient)",
        GRN: u"Guarani (not tested)",
        GUJ: u"Gujarati (not tested)",
        HIN: u"Hindi (not tested)",
        HRV: u"Croatian",
        HUN: u"Hungarian",
        HYE: u"Armenian (not tested)",
        INA: u"Interlingua (not tested)",
        IND: u"Indonesian (not tested)",
        ISL: u"Icelandic",
        ITA: u"Italian",
        JBO: u"Lojban (not tested)",
        KAL: u"Greenlandic (not tested)",
        KAN: u"Kannada (not tested)",
        KAT: u"Georgian (not tested)",
        KIR: u"Kirghiz (not tested)",
        KUR: u"Kurdish (not tested)",
        LAT: u"Latin",
        LAV: u"Latvian",
        LFN: u"Lingua Franca Nova (not tested)",
        LIT: u"Lithuanian",
        MAL: u"Malayalam (not tested)",
        MAR: u"Marathi (not tested)",
        MKD: u"Macedonian (not tested)",
        MLT: u"Maltese (not tested)",
        MSA: u"Malay (not tested)",
        MYA: u"Burmese (not tested)",
        NAH: u"Nahuatl (not tested)",
        NEP: u"Nepali (not tested)",
        NLD: u"Dutch",
        NOR: u"Norwegian",
        ORI: u"Oriya (not tested)",
        ORM: u"Oromo (not tested)",
        PAN: u"Panjabi (not tested)",
        PAP: u"Papiamento (not tested)",
        POL: u"Polish",
        POR: u"Portuguese",
        RON: u"Romanian",
        RUS: u"Russian",
        SIN: u"Sinhala (not tested)",
        SLK: u"Slovak",
        SLV: u"Slovenian (not tested)",
        SPA: u"Spanish",
        SQI: u"Albanian (not tested)",
        SRP: u"Serbian",
        SWA: u"Swahili",
        SWE: u"Swedish",
        TAM: u"Tamil (not tested)",
        TAT: u"Tatar (not tested)",
        TEL: u"Telugu (not tested)",
        TSN: u"Tswana (not tested)",
        TUR: u"Turkish",
        UKR: u"Ukrainian",
        URD: u"Urdu (not tested)",
        VIE: u"Vietnamese (not tested)",
        YUE: u"Yue Chinese (not tested)",
        ZHO: u"Chinese (not tested)",
        ENG_GBR: u"English (GB)",
        ENG_SCT: u"English (Scotland) (not tested)",
        ENG_USA: u"English (USA)",
        SPA_ESP: u"Spanish (Castillan)",
        FRA_BEL: u"French (Belgium) (not tested)",
        FRA_FRA: u"French (France)",
        POR_BRA: u"Portuguese (Brazil) (not tested)",
        POR_PRT: u"Portuguese (Portugal)",
        AF: u"Afrikaans",
        AN: u"Aragonese (not tested)",
        AM: u"Amharic (not tested)",
        AS: u"Assamese (not tested)",
        AZ: u"Azerbaijani (not tested)",
        BG: u"Bulgarian",
        BN: u"Bengali (not tested)",
        BS: u"Bosnian (not tested)",
        CA: u"Catalan",
        CS: u"Czech",
        CY: u"Welsh",
        DA: u"Danish",
        DE: u"German",
        EL: u"Greek (Modern)",
        EN: u"English",
        EN_GB: u"English (GB)",
        EN_GB_SCOTLAND: u"English (Scotland) (not tested)",
        EN_GB_X_GBCLAN: u"English (Northern) (not tested)",
        EN_GB_X_GBCWMD: u"English (Midlands) (not tested)",
        EN_GB_X_RP: u"English (Received Pronunciation) (not tested)",
        EN_US: u"English (USA)",
        EN_029: u"English (West Indies) (not tested)",
        EO: u"Esperanto (not tested)",
        ES: u"Spanish (Castillan)",
        ES_419: u"Spanish (Latin America) (not tested)",
        ET: u"Estonian",
        EU: u"Basque (not tested)",
        FA: u"Persian",
        FA_LATN: u"Persian (Pinglish)",
        FI: u"Finnish",
        FR: u"French",
        FR_BE: u"French (Belgium) (not tested)",
        FR_FR: u"French (France)",
        GA: u"Irish",
        GD: u"Scottish Gaelic (not tested)",
        GN: u"Guarani (not tested)",
        GU: u"Gujarati (not tested)",
        HI: u"Hindi (not tested)",
        HR: u"Croatian",
        HU: u"Hungarian",
        HY: u"Armenian (not tested)",
        HY_AREVMDA: u"Armenian (West) (not tested)",
        IA: u"Interlingua (not tested)",
        ID: u"Indonesian (not tested)",
        IS: u"Icelandic",
        IT: u"Italian",
        KA: u"Georgian (not tested)",
        KL: u"Greenlandic (not tested)",
        KN: u"Kannada (not tested)",
        KU: u"Kurdish (not tested)",
        KY: u"Kirghiz (not tested)",
        LA: u"Latin",
        LT: u"Lithuanian",
        LV: u"Latvian",
        MK: u"Macedonian (not tested)",
        ML: u"Malayalam (not tested)",
        MR: u"Marathi (not tested)",
        MS: u"Malay (not tested)",
        MT: u"Maltese (not tested)",
        MY: u"Burmese (not tested)",
        NCI: u"Nahuatl (not tested)",
        NE: u"Nepali (not tested)",
        NL: u"Dutch",
        NO: u"Norwegian",
        OM: u"Oromo (not tested)",
        OR: u"Oriya (not tested)",
        PA: u"Panjabi (not tested)",
        PL: u"Polish",
        PT: u"Portuguese",
        PT_BR: u"Portuguese (Brazil) (not tested)",
        PT_PT: u"Portuguese (Portugal)",
        RO: u"Romanian",
        RU: u"Russian",
        SI: u"Sinhala (not tested)",
        SK: u"Slovak",
        SL: u"Slovenian (not tested)",
        SQ: u"Albanian (not tested)",
        SR: u"Serbian",
        SV: u"Swedish",
        SW: u"Swahili",
        TA: u"Tamil (not tested)",
        TE: u"Telugu (not tested)",
        TN: u"Tswana (not tested)",
        TR: u"Turkish",
        TT: u"Tatar (not tested)",
        UK: u"Ukrainian",
        UR: u"Urdu (not tested)",
        VI: u"Vietnamese (not tested)",
        VI_VN_X_CENTRAL: u"Vietnamese (hue) (not tested)",
        VI_VN_X_SOUTH: u"Vietnamese (sgn) (not tested)",
        ZH: u"Mandarin Chinese (not tested)",
        ZH_YUE: u"Yue Chinese (not tested)",
    }

    CODE_TO_HUMAN_LIST = sorted([u"%s\t%s" % (k, v) for k, v in CODE_TO_HUMAN.items()])

    LANGUAGE_TO_VOICE_CODE = {
        AF: "af",
        AM: "am",
        AN: "an",
        AS: "as",
        AZ: "az",
        BG: "bg",
        BN: "bn",
        BS: "bs",
        CA: "ca",
        CS: "cs",
        CY: "cy",
        DA: "da",
        DE: "de",
        EL: "el",
        EN: "en",
        EN_029: "en-029",
        EN_GB: "en-gb",
        EN_GB_SCOTLAND: "en-gb-scotland",
        EN_GB_X_GBCLAN: "en-gb-x-gbclan",
        EN_GB_X_GBCWMD: "en-gb-x-gbcwmd",
        EN_GB_X_RP: "en-gb-x-rp",
        EN_US: "en-us",
        EO: "eo",
        ES: "es",
        ES_419: "es-419",
        ET: "et",
        EU: "eu",
        FA: "fa",
        FA_LATN: "fa-Latn",
        FI: "fi",
        FR: "fr",
        FR_BE: "fr-be",
        FR_FR: "fr-fr",
        GA: "ga",
        GD: "gd",
        # COMMENTED GRC: "grc",
        GN: "gn",
        GU: "gu",
        HI: "hi",
        HR: "hr",
        HU: "hu",
        HY: "hy",
        HY_AREVMDA: "hy-arevmda",
        IA: "ia",
        ID: "id",
        IS: "is",
        IT: "it",
        # COMMENTED JBO: "jbo",
        KA: "ka",
        KL: "kl",
        KN: "kn",
        KU: "ku",
        KY: "ky",
        LA: "la",
        # COMMENTED LFN: "lfn",
        LT: "lt",
        LV: "lv",
        MK: "mk",
        ML: "ml",
        MR: "mr",
        MS: "ms",
        MT: "mt",
        MY: "my",
        NCI: "nci",
        NE: "ne",
        NL: "nl",
        NO: "no",
        OM: "om",
        OR: "or",
        PA: "pa",
        # COMMENTED PAP: "pap",
        PL: "pl",
        PT: "pt",
        PT_BR: "pt-br",
        PT_PT: "pt-pt",
        RO: "ro",
        RU: "ru",
        SI: "si",
        SK: "sk",
        SL: "sl",
        SQ: "sq",
        SR: "sr",
        SV: "sv",
        SW: "sw",
        TA: "ta",
        TE: "te",
        TN: "tn",
        TR: "tr",
        TT: "tt",
        UK: "ru",   # NOTE mocking support for Ukrainian with Russian voice
        UR: "ur",
        VI: "vi",
        VI_VN_X_CENTRAL: "vi-vn-x-central",
        VI_VN_X_SOUTH: "vi-vn-x-south",
        ZH: "zh",
        ZH_YUE: "zh-yue",
        AFR: "af",
        AMH: "am",
        ARG: "an",
        ASM: "as",
        AZE: "az",
        BEN: "bn",
        BOS: "bs",
        BUL: "bg",
        CAT: "ca",
        CES: "cs",
        CMN: "zh",
        CYM: "cy",
        DAN: "da",
        DEU: "de",
        ELL: "el",
        ENG: "en",
        EPO: "eo",
        EST: "et",
        FAS: "fa",
        FIN: "fi",
        FRA: "fr",
        GLA: "gd",
        GLE: "ga",
        GRC: "grc",
        GRN: "gn",
        GUJ: "gu",
        HIN: "hi",
        HRV: "hr",
        HUN: "hu",
        HYE: "hy",
        INA: "ia",
        IND: "id",
        ISL: "is",
        ITA: "it",
        JBO: "jbo",
        KAL: "kl",
        KAN: "kn",
        KAT: "ka",
        KIR: "ky",
        KUR: "ku",
        LAT: "la",
        LAV: "lv",
        LFN: "lfn",
        LIT: "lt",
        MAL: "ml",
        MAR: "mr",
        MKD: "mk",
        MLT: "mt",
        MSA: "ms",
        MYA: "my",
        NAH: "nci",
        NEP: "ne",
        NLD: "nl",
        NOR: "no",
        ORI: "or",
        ORM: "om",
        PAN: "pa",
        PAP: "pap",
        POL: "pl",
        POR: "pt",
        RON: "ro",
        RUS: "ru",
        SIN: "si",
        SLK: "sk",
        SLV: "sl",
        SPA: "es",
        SQI: "sq",
        SRP: "sr",
        SWA: "sw",
        SWE: "sv",
        TAM: "ta",
        TAT: "tt",
        TEL: "te",
        TSN: "tn",
        TUR: "tr",
        UKR: "ru",  # NOTE mocking support for Ukrainian with Russian voice
        URD: "ur",
        VIE: "vi",
        YUE: "zh-yue",
        ZHO: "zh",
        ENG_GBR: "en-gb",
        ENG_SCT: "en-gb-scotland",
        ENG_USA: "en-us",
        SPA_ESP: "es-es",
        FRA_BEL: "fr-be",
        FRA_FRA: "fr-fr",
        POR_BRA: "pt-br",
        POR_PRT: "pt-pt"
    }
    DEFAULT_LANGUAGE = ENG

    DEFAULT_TTS_PATH = "espeak-ng"

    OUTPUT_AUDIO_FORMAT = ("pcm_s16le", 1, 22050)

    HAS_SUBPROCESS_CALL = True

    TAG = u"ESPEAKNGTTSWrapper"

    def __init__(self, rconf=None, logger=None):
        super(ESPEAKNGTTSWrapper, self).__init__(rconf=rconf, logger=logger)
        self.set_subprocess_arguments([
            self.tts_path,
            u"-v",
            self.CLI_PARAMETER_VOICE_CODE_STRING,
            u"-w",
            self.CLI_PARAMETER_WAVE_PATH,
            self.CLI_PARAMETER_TEXT_STDIN
        ])
