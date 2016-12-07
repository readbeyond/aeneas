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

* :class:`~aeneas.ttswrappers.espeakttswrapper.ESPEAKTTSWrapper`,
  a wrapper for the ``eSpeak`` TTS engine.

Please refer to
http://espeak.sourceforge.net/
for further details.
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.exacttiming import TimeValue
from aeneas.language import Language
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.ttswrappers.basettswrapper import BaseTTSWrapper
import aeneas.globalfunctions as gf


class ESPEAKTTSWrapper(BaseTTSWrapper):
    """
    A wrapper for the ``eSpeak`` TTS engine.

    This wrapper is the default TTS engine for ``aeneas``.

    This wrapper supports calling the TTS engine
    via ``subprocess`` or via Python C extension.

    In abstract terms, it performs one or more calls like ::

        $ espeak -v voice_code -w /tmp/output_file.wav < text

    To use this TTS engine, specify ::

        "tts=espeak"

    in the ``RuntimeConfiguration`` object.
    (You can omit this, since eSpeak is the default TTS engine.)
    To execute from a non-default location: ::

        "tts=espeak|tts_path=/path/to/espeak"

    To run the ``cew`` Python C extension
    in a separate process via
    :class:`~aeneas.cewsubprocess.CEWSubprocess`, use ::

        "cew_subprocess_enabled=True|cew_subprocess_path=/path/to/python"

    in the ``rconf`` object.

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

    ARG = Language.ARG
    """ Aragonese (not tested) """

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

    FAS = Language.FAS
    """ Persian """

    FIN = Language.FIN
    """ Finnish """

    FRA = Language.FRA
    """ French """

    GLE = Language.GLE
    """ Irish """

    GRC = Language.GRC
    """ Greek (Ancient) """

    HIN = Language.HIN
    """ Hindi (not tested) """

    HRV = Language.HRV
    """ Croatian """

    HUN = Language.HUN
    """ Hungarian """

    HYE = Language.HYE
    """ Armenian (not tested) """

    IND = Language.IND
    """ Indonesian (not tested) """

    ISL = Language.ISL
    """ Icelandic """

    ITA = Language.ITA
    """ Italian """

    JBO = Language.JBO
    """ Lojban (not tested) """

    KAN = Language.KAN
    """ Kannada (not tested) """

    KAT = Language.KAT
    """ Georgian (not tested) """

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

    MKD = Language.MKD
    """ Macedonian (not tested) """

    MSA = Language.MSA
    """ Malay (not tested) """

    NEP = Language.NEP
    """ Nepali (not tested) """

    NLD = Language.NLD
    """ Dutch """

    NOR = Language.NOR
    """ Norwegian """

    PAN = Language.PAN
    """ Panjabi (not tested) """

    POL = Language.POL
    """ Polish """

    POR = Language.POR
    """ Portuguese """

    RON = Language.RON
    """ Romanian """

    RUS = Language.RUS
    """ Russian """

    SLK = Language.SLK
    """ Slovak """

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

    TUR = Language.TUR
    """ Turkish """

    UKR = Language.UKR
    """ Ukrainian """

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

    BG = "bg"
    """ Bulgarian """

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

    EN_SC = "en-sc"
    """ English (Scotland) (not tested) """

    EN_UK_NORTH = "en-uk-north"
    """ English (Northern) (not tested) """

    EN_UK_RP = "en-uk-rp"
    """ English (Received Pronunciation) (not tested) """

    EN_UK_WMIDS = "en-uk-wmids"
    """ English (Midlands) (not tested) """

    EN_US = "en-us"
    """ English (USA) """

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
    """ Irish """

    # NOTE already defined
    # COMMENTED GRC = "grc"
    # COMMENTED """ Greek (Ancient) """

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

    # NOTE already defined
    # COMMENTED JBO = "jbo"
    # COMMENTED """ Lojban (not tested) """

    KA = "ka"
    """ Georgian (not tested) """

    KN = "kn"
    """ Kannada (not tested) """

    KU = "ku"
    """ Kurdish (not tested) """

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

    MS = "ms"
    """ Malay (not tested) """

    NE = "ne"
    """ Nepali (not tested) """

    NL = "nl"
    """ Dutch """

    NO = "no"
    """ Norwegian """

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

    SQ = "sq"
    """ Albanian (not tested) """

    SK = "sk"
    """ Slovak """

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
    """ Vietnamese (not tested) """

    VI_HUE = "vi-hue"
    """ Vietnamese (hue) (not tested) """

    VI_SGN = "vi-sgn"
    """ Vietnamese (sgn) (not tested) """

    ZH = "zh"
    """ Mandarin Chinese (not tested) """

    ZH_YUE = "zh-yue"
    """ Yue Chinese (not tested) """

    CODE_TO_HUMAN = {
        AFR: u"Afrikaans",
        ARG: u"Aragonese (not tested)",
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
        FAS: u"Persian",
        FIN: u"Finnish",
        FRA: u"French",
        GLE: u"Irish",
        GRC: u"Greek (Ancient)",
        HIN: u"Hindi (not tested)",
        HRV: u"Croatian",
        HUN: u"Hungarian",
        HYE: u"Armenian (not tested)",
        IND: u"Indonesian (not tested)",
        ISL: u"Icelandic",
        ITA: u"Italian",
        JBO: u"Lojban (not tested)",
        KAN: u"Kannada (not tested)",
        KAT: u"Georgian (not tested)",
        KUR: u"Kurdish (not tested)",
        LAT: u"Latin",
        LAV: u"Latvian",
        LFN: u"Lingua Franca Nova (not tested)",
        LIT: u"Lithuanian",
        MAL: u"Malayalam (not tested)",
        MKD: u"Macedonian (not tested)",
        MSA: u"Malay (not tested)",
        NEP: u"Nepali (not tested)",
        NLD: u"Dutch",
        NOR: u"Norwegian",
        PAN: u"Panjabi (not tested)",
        POL: u"Polish",
        POR: u"Portuguese",
        RON: u"Romanian",
        RUS: u"Russian",
        SLK: u"Slovak",
        SPA: u"Spanish",
        SQI: u"Albanian (not tested)",
        SRP: u"Serbian",
        SWA: u"Swahili",
        SWE: u"Swedish",
        TAM: u"Tamil (not tested)",
        TUR: u"Turkish",
        UKR: u"Ukrainian",
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
        BG: u"Bulgarian",
        BS: u"Bosnian (not tested)",
        CA: u"Catalan",
        CS: u"Czech",
        CY: u"Welsh",
        DA: u"Danish",
        DE: u"German",
        EL: u"Greek (Modern)",
        EN: u"English",
        EN_GB: u"English (GB)",
        EN_SC: u"English (Scotland) (not tested)",
        EN_UK_NORTH: u"English (Northern) (not tested)",
        EN_UK_RP: u"English (Received Pronunciation) (not tested)",
        EN_UK_WMIDS: u"English (Midlands) (not tested)",
        EN_US: u"English (USA)",
        EN_WI: u"English (West Indies) (not tested)",
        EO: u"Esperanto (not tested)",
        ES: u"Spanish (Castillan)",
        ES_LA: u"Spanish (Latin America) (not tested)",
        ET: u"Estonian",
        FA: u"Persian",
        FA_PIN: u"Persian (Pinglish)",
        FI: u"Finnish",
        FR: u"French",
        FR_BE: u"French (Belgium) (not tested)",
        FR_FR: u"French (France)",
        GA: u"Irish",
        HI: u"Hindi (not tested)",
        HR: u"Croatian",
        HU: u"Hungarian",
        HY: u"Armenian (not tested)",
        HY_WEST: u"Armenian (West) (not tested)",
        ID: u"Indonesian (not tested)",
        IS: u"Icelandic",
        IT: u"Italian",
        KA: u"Georgian (not tested)",
        KN: u"Kannada (not tested)",
        KU: u"Kurdish (not tested)",
        LA: u"Latin",
        LT: u"Lithuanian",
        LV: u"Latvian",
        MK: u"Macedonian (not tested)",
        ML: u"Malayalam (not tested)",
        MS: u"Malay (not tested)",
        NE: u"Nepali (not tested)",
        NL: u"Dutch",
        NO: u"Norwegian",
        PA: u"Panjabi (not tested)",
        PL: u"Polish",
        PT: u"Portuguese",
        PT_BR: u"Portuguese (Brazil) (not tested)",
        PT_PT: u"Portuguese (Portugal)",
        RO: u"Romanian",
        RU: u"Russian",
        SQ: u"Albanian (not tested)",
        SK: u"Slovak",
        SR: u"Serbian",
        SV: u"Swedish",
        SW: u"Swahili",
        TA: u"Tamil (not tested)",
        TR: u"Turkish",
        UK: u"Ukrainian",
        VI: u"Vietnamese (not tested)",
        VI_HUE: u"Vietnamese (hue) (not tested)",
        VI_SGN: u"Vietnamese (sgn) (not tested)",
        ZH: u"Mandarin Chinese (not tested)",
        ZH_YUE: u"Yue Chinese (not tested)",
    }

    CODE_TO_HUMAN_LIST = sorted([u"%s\t%s" % (k, v) for k, v in CODE_TO_HUMAN.items()])

    LANGUAGE_TO_VOICE_CODE = {
        AF: "af",
        AN: "an",
        BG: "bg",
        BS: "bs",
        CA: "ca",
        CS: "cs",
        CY: "cy",
        DA: "da",
        DE: "de",
        EL: "el",
        EN: "en",
        EN_GB: "en-gb",
        EN_SC: "en-sc",
        EN_UK_NORTH: "en-uk-north",
        EN_UK_RP: "en-uk-rp",
        EN_UK_WMIDS: "en-uk-wmids",
        EN_US: "en-us",
        EN_WI: "en-wi",
        EO: "eo",
        ES: "es",
        ES_LA: "es-la",
        ET: "et",
        FA: "fa",
        FA_PIN: "fa-pin",
        FI: "fi",
        FR: "fr",
        FR_BE: "fr-be",
        FR_FR: "fr-fr",
        GA: "ga",
        # COMMENTED GRC: "grc",
        HI: "hi",
        HR: "hr",
        HU: "hu",
        HY: "hy",
        HY_WEST: "hy-west",
        ID: "id",
        IS: "is",
        IT: "it",
        # COMMENTED JBO: "jbo",
        KA: "ka",
        KN: "kn",
        KU: "ku",
        LA: "la",
        # COMMENTED LFN: "lfn",
        LT: "lt",
        LV: "lv",
        MK: "mk",
        ML: "ml",
        MS: "ms",
        NE: "ne",
        NL: "nl",
        NO: "no",
        PA: "pa",
        PL: "pl",
        PT: "pt",
        PT_BR: "pt-br",
        PT_PT: "pt-pt",
        RO: "ro",
        RU: "ru",
        SQ: "sq",
        SK: "sk",
        SR: "sr",
        SV: "sv",
        SW: "sw",
        TA: "ta",
        TR: "tr",
        UK: "ru",   # NOTE mocking support for Ukrainian with Russian voice
        VI: "vi",
        VI_HUE: "vi-hue",
        VI_SGN: "vi-sgn",
        ZH: "zh",
        ZH_YUE: "zh-yue",
        AFR: "af",
        ARG: "an",
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
        GLE: "ga",
        GRC: "grc",
        HIN: "hi",
        HRV: "hr",
        HUN: "hu",
        HYE: "hy",
        IND: "id",
        ISL: "is",
        ITA: "it",
        JBO: "jbo",
        KAN: "kn",
        KAT: "ka",
        KUR: "ku",
        LAT: "la",
        LAV: "lv",
        LFN: "lfn",
        LIT: "lt",
        MAL: "ml",
        MKD: "mk",
        MSA: "ms",
        NEP: "ne",
        NLD: "nl",
        NOR: "no",
        PAN: "pa",
        POL: "pl",
        POR: "pt",
        RON: "ro",
        RUS: "ru",
        SLK: "sk",
        SPA: "es",
        SQI: "sq",
        SRP: "sr",
        SWA: "sw",
        SWE: "sv",
        TAM: "ta",
        TUR: "tr",
        UKR: "ru",  # NOTE mocking support for Ukrainian with Russian voice
        VIE: "vi",
        YUE: "zh-yue",
        ZHO: "zh",
        ENG_GBR: "en-gb",
        ENG_SCT: "en-sc",
        ENG_USA: "en-us",
        SPA_ESP: "es-es",
        FRA_BEL: "fr-be",
        FRA_FRA: "fr-fr",
        POR_BRA: "pt-br",
        POR_PRT: "pt-pt"
    }
    DEFAULT_LANGUAGE = ENG

    DEFAULT_TTS_PATH = "espeak"

    OUTPUT_AUDIO_FORMAT = ("pcm_s16le", 1, 22050)

    HAS_SUBPROCESS_CALL = True

    HAS_C_EXTENSION_CALL = True

    C_EXTENSION_NAME = "cew"

    TAG = u"ESPEAKTTSWrapper"

    def __init__(self, rconf=None, logger=None):
        super(ESPEAKTTSWrapper, self).__init__(rconf=rconf, logger=logger)
        self.set_subprocess_arguments([
            self.tts_path,
            u"-v",
            self.CLI_PARAMETER_VOICE_CODE_STRING,
            u"-w",
            self.CLI_PARAMETER_WAVE_PATH,
            self.CLI_PARAMETER_TEXT_STDIN
        ])

    def _synthesize_multiple_c_extension(self, text_file, output_file_path, quit_after=None, backwards=False):
        """
        Synthesize multiple text fragments, using the cew extension.

        Return a tuple (anchors, total_time, num_chars).

        :rtype: (bool, (list, :class:`~aeneas.exacttiming.TimeValue`, int))
        """
        self.log(u"Synthesizing using C extension...")

        # convert parameters from Python values to C values
        try:
            c_quit_after = float(quit_after)
        except TypeError:
            c_quit_after = 0.0
        c_backwards = 0
        if backwards:
            c_backwards = 1
        self.log([u"output_file_path: %s", output_file_path])
        self.log([u"c_quit_after:     %.3f", c_quit_after])
        self.log([u"c_backwards:      %d", c_backwards])
        self.log(u"Preparing u_text...")
        u_text = []
        fragments = text_file.fragments
        for fragment in fragments:
            f_lang = fragment.language
            f_text = fragment.filtered_text
            if f_lang is None:
                f_lang = self.DEFAULT_LANGUAGE
            f_voice_code = self._language_to_voice_code(f_lang)
            if f_text is None:
                f_text = u""
            u_text.append((f_voice_code, f_text))
        self.log(u"Preparing u_text... done")

        # call C extension
        sr = None
        sf = None
        intervals = None
        if self.rconf[RuntimeConfiguration.CEW_SUBPROCESS_ENABLED]:
            self.log(u"Using cewsubprocess to call aeneas.cew")
            try:
                self.log(u"Importing aeneas.cewsubprocess...")
                from aeneas.cewsubprocess import CEWSubprocess
                self.log(u"Importing aeneas.cewsubprocess... done")
                self.log(u"Calling aeneas.cewsubprocess...")
                cewsub = CEWSubprocess(rconf=self.rconf, logger=self.logger)
                sr, sf, intervals = cewsub.synthesize_multiple(output_file_path, c_quit_after, c_backwards, u_text)
                self.log(u"Calling aeneas.cewsubprocess... done")
            except Exception as exc:
                self.log_exc(u"An unexpected error occurred while running cewsubprocess", exc, False, None)
                # NOTE not critical, try calling aeneas.cew directly
                # COMMENTED return (False, None)

        if sr is None:
            self.log(u"Preparing c_text...")
            if gf.PY2:
                # Python 2 => pass byte strings
                c_text = [(gf.safe_bytes(t[0]), gf.safe_bytes(t[1])) for t in u_text]
            else:
                # Python 3 => pass Unicode strings
                c_text = [(gf.safe_unicode(t[0]), gf.safe_unicode(t[1])) for t in u_text]
            self.log(u"Preparing c_text... done")

            self.log(u"Calling aeneas.cew directly")
            try:
                self.log(u"Importing aeneas.cew...")
                import aeneas.cew.cew
                self.log(u"Importing aeneas.cew... done")
                self.log(u"Calling aeneas.cew...")
                sr, sf, intervals = aeneas.cew.cew.synthesize_multiple(
                    output_file_path,
                    c_quit_after,
                    c_backwards,
                    c_text
                )
                self.log(u"Calling aeneas.cew... done")
            except Exception as exc:
                self.log_exc(u"An unexpected error occurred while running cew", exc, False, None)
                return (False, None)

        self.log([u"sr: %d", sr])
        self.log([u"sf: %d", sf])

        # create output
        anchors = []
        current_time = TimeValue("0.000")
        num_chars = 0
        if backwards:
            fragments = fragments[::-1]
        for i in range(sf):
            # get the correct fragment
            fragment = fragments[i]
            # store for later output
            anchors.append([
                TimeValue(intervals[i][0]),
                fragment.identifier,
                fragment.filtered_text
            ])
            # increase the character counter
            num_chars += fragment.characters
            # update current_time
            current_time = TimeValue(intervals[i][1])

        # return output
        # NOTE anchors do not make sense if backwards == True
        self.log([u"Returning %d time anchors", len(anchors)])
        self.log([u"Current time %.3f", current_time])
        self.log([u"Synthesized %d characters", num_chars])
        self.log(u"Synthesizing using C extension... done")
        return (True, (anchors, current_time, num_chars))
