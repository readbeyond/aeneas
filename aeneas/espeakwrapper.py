#!/usr/bin/env python
# coding=utf-8

"""
This module contains the following classes:

* :class:`~aeneas.espeakwrapper.ESPEAKWrapper`, a wrapper for the ``eSpeak`` TTS engine.
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.language import Language
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.timevalue import TimeValue
from aeneas.ttswrapper import TTSWrapper
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.5.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class ESPEAKWrapper(TTSWrapper):
    """
    A wrapper for the ``espeak`` TTS engine.

    This wrapper is the default TTS engine for ``aeneas``.

    This wrapper supports calling the TTS engine
    via ``subprocess`` or via Python C extension.

    In abstract terms, it performs one or more calls like ::

        $ espeak -v voice_code -w /tmp/output_file.wav < text

    To specify the path of the TTS executable, use ::

        "tts=espeak|tts_path=/path/to/espeak"

    in the ``rconf`` object.

    To run the ``cew`` Python C extension
    in a separate process via
    :class:`~aeneas.cewsubprocess.CEWSubprocess`, use ::

        "cew_subprocess_enabled=True|cew_subprocess_path=/path/to/python"

    in the ``rconf`` object.

    See :class:`~aeneas.ttswrapper.TTSWrapper` for the available functions.
    Below are listed the languages supported by this wrapper.

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    AFR = Language.AFR
    """ Afrikaans (not tested) """

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
    #GRC = "grc"
    #""" Greek (Ancient) """

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
    #JBO = "jbo"
    #""" Lojban (not tested) """

    KA = "ka"
    """ Georgian (not tested) """

    KN = "kn"
    """ Kannada (not tested) """

    KU = "ku"
    """ Kurdish (not tested) """

    LA = "la"
    """ Latin """

    # NOTE already defined
    #LFN = "lfn"
    #""" Lingua Franca Nova (not tested) """

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

    LANGUAGE_TO_VOICE_CODE = {
        AF : "af",
        AN : "an",
        BG : "bg",
        BS : "bs",
        CA : "ca",
        CS : "cs",
        CY : "cy",
        DA : "da",
        DE : "de",
        EL : "el",
        EN : "en",
        EN_GB : "en-gb",
        EN_SC : "en-sc",
        EN_UK_NORTH : "en-uk-north",
        EN_UK_RP : "en-uk-rp",
        EN_UK_WMIDS : "en-uk-wmids",
        EN_US : "en-us",
        EN_WI : "en-wi",
        EO : "eo",
        ES : "es",
        ES_LA : "es-la",
        ET : "et",
        FA : "fa",
        FA_PIN : "fa-pin",
        FI : "fi",
        FR : "fr",
        FR_BE : "fr-be",
        FR_FR : "fr-fr",
        GA : "ga",
        #GRC : "grc",
        HI : "hi",
        HR : "hr",
        HU : "hu",
        HY : "hy",
        HY_WEST : "hy-west",
        ID : "id",
        IS : "is",
        IT : "it",
        #JBO : "jbo",
        KA : "ka",
        KN : "kn",
        KU : "ku",
        LA : "la",
        #LFN : "lfn",
        LT : "lt",
        LV : "lv",
        MK : "mk",
        ML : "ml",
        MS : "ms",
        NE : "ne",
        NL : "nl",
        NO : "no",
        PA : "pa",
        PL : "pl",
        PT : "pt",
        PT_BR : "pt-br",
        PT_PT : "pt-pt",
        RO : "ro",
        RU : "ru",
        SQ : "sq",
        SK : "sk",
        SR : "sr",
        SV : "sv",
        SW : "sw",
        TA : "ta",
        TR : "tr",
        UK : "ru", # NOTE mocking support for Ukrainian with Russian voice
        VI : "vi",
        VI_HUE : "vi-hue",
        VI_SGN : "vi-sgn",
        ZH : "zh",
        ZH_YUE : "zh-yue",
        AFR : "af",
        ARG : "an",
        BOS : "bs",
        BUL : "bg",
        CAT : "ca",
        CES : "cs",
        CMN : "zh",
        CYM : "cy",
        DAN : "da",
        DEU : "de",
        ELL : "el",
        ENG : "en",
        EPO : "eo",
        EST : "et",
        FAS : "fa",
        FIN : "fi",
        FRA : "fr",
        GLE : "ga",
        GRC : "grc",
        HIN : "hi",
        HRV : "hr",
        HUN : "hu",
        HYE : "hy",
        IND : "id",
        ISL : "is",
        ITA : "it",
        JBO : "jbo",
        KAN : "kn",
        KAT : "ka",
        KUR : "ku",
        LAT : "la",
        LAV : "lv",
        LFN : "lfn",
        LIT : "lt",
        MAL : "ml",
        MKD : "mk",
        MSA : "ms",
        NEP : "ne",
        NLD : "nl",
        NOR : "no",
        PAN : "pa",
        POL : "pl",
        POR : "pt",
        RON : "ro",
        RUS : "ru",
        SLK : "sk",
        SPA : "es",
        SQI : "sq",
        SRP : "sr",
        SWA : "sw",
        SWE : "sv",
        TAM : "ta",
        TUR : "tr",
        UKR : "ru", # NOTE mocking support for Ukrainian with Russian voice
        VIE : "vi",
        YUE : "zh-yue",
        ZHO : "zh",
        ENG_GBR : "en-gb",
        ENG_SCT : "en-sc",
        ENG_USA : "en-us",
        SPA_ESP : "es-es",
        FRA_BEL : "fr-be",
        FRA_FRA : "fr-fr",
        POR_BRA : "pt-br",
        POR_PRT : "pt-pt"
    }
    DEFAULT_LANGUAGE = ENG

    OUTPUT_MONO_WAVE = True

    TAG = u"ESPEAKWrapper"

    def __init__(self, rconf=None, logger=None):
        super(ESPEAKWrapper, self).__init__(
            has_subprocess_call=True,
            has_c_extension_call=True,
            has_python_call=False,
            rconf=rconf,
            logger=logger
        )
        self.set_subprocess_arguments([
            self.rconf[RuntimeConfiguration.TTS_PATH],
            u"-v",
            TTSWrapper.CLI_PARAMETER_VOICE_CODE_STRING,
            u"-w",
            TTSWrapper.CLI_PARAMETER_WAVE_PATH,
            TTSWrapper.CLI_PARAMETER_TEXT_STDIN
        ])

    def _synthesize_multiple_c_extension(self, text_file, output_file_path, quit_after=None, backwards=False):
        """
        Synthesize multiple text fragments, using the cew extension.

        Return a tuple (anchors, total_time, num_chars).

        :rtype: (bool, (list, :class:`~aeneas.timevalue.TimeValue`, int))
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
                #return (False, None)

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

    def _synthesize_single_c_extension(self, text, voice_code, output_file_path):
        """
        Synthesize a single text fragment, using the cew extension.

        Return the duration of the synthesized text, in seconds.

        :rtype: (bool, (:class:`~aeneas.timevalue.TimeValue`, ))
        """
        self.log(u"Synthesizing using C extension...")

        end = None
        if self.rconf[RuntimeConfiguration.CEW_SUBPROCESS_ENABLED]:
            self.log(u"Using cewsubprocess to call aeneas.cew")
            try:
                self.log(u"Importing aeneas.cewsubprocess...")
                from aeneas.cewsubprocess import CEWSubprocess
                self.log(u"Importing aeneas.cewsubprocess... done")
                self.log(u"Calling aeneas.cewsubprocess...")
                cewsub = CEWSubprocess(rconf=self.rconf, logger=self.logger)
                end = cewsub.synthesize_single(output_file_path, voice_code, text)
                self.log(u"Calling aeneas.cewsubprocess... done")
            except Exception as exc:
                self.log_exc(u"An unexpected error occurred while running cewsubprocess", exc, False, None)
                # NOTE not critical, try calling aeneas.cew directly
                #return (False, None)

        if end is None:
            self.log(u"Preparing c_text...")
            if gf.PY2:
                # Python 2 => pass byte strings
                c_text = gf.safe_bytes(text)
            else:
                # Python 3 => pass Unicode strings
                c_text = gf.safe_unicode(text)
            self.log(u"Preparing c_text... done")

            self.log(u"Calling aeneas.cew directly")
            try:
                self.log(u"Importing aeneas.cew...")
                import aeneas.cew.cew
                self.log(u"Importing aeneas.cew... done")
                self.log(u"Calling aeneas.cew...")
                sr, begin, end = aeneas.cew.cew.synthesize_single(
                    output_file_path,
                    voice_code,
                    c_text
                )
                end = TimeValue(end)
                self.log(u"Calling aeneas.cew... done")
            except Exception as exc:
                self.log_exc(u"An unexpected error occurred while running cew", exc, False, None)
                return (False, None)

        self.log(u"Synthesizing using C extension... done")
        return (True, (end, ))



