#!/usr/bin/env python
# coding=utf-8

"""
A wrapper for the ``speect`` TTS engine.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import speect
import speect.audio
import speect.audio_riff

from aeneas.audiofile import AudioFile
from aeneas.language import Language
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

class CustomTTSWrapper(TTSWrapper):
    """
    A wrapper for the ``speect`` TTS engine.

    This wrapper supports calling the TTS engine
    only via Python.

    To use this TTS engine, specify ::

        "tts=custom|tts_path=/path/to/this/file.py"

    in the ``RuntimeConfiguration`` object.

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    TAG = u"CustomTTSWrapper"

    #
    # NOTE in this example we load an English voice,
    #      hence we support only English language,
    #      and we map it to a dummy voice code
    #
    ENG = Language.ENG
    """ English """
    LANGUAGE_TO_VOICE_CODE = {
        ENG : ENG
    }
    DEFAULT_LANGUAGE = ENG

    #
    # NOTE in this example we load a voice producing
    #      audio data in PCM16 mono WAVE (RIFF) format
    #
    OUTPUT_MONO_WAVE = True

    def __init__(self, rconf=None, logger=None):
        super(CustomTTSWrapper, self).__init__(
            has_subprocess_call=False,
            has_c_extension_call=False,
            has_python_call=True,
            rconf=rconf,
            logger=logger)

    def _synthesize_multiple_python(self, text_file, output_file_path, quit_after=None, backwards=False):
        """
        Synthesize multiple text fragments, via Python call.

        Return a tuple (anchors, total_time, num_chars).

        :rtype: (bool, (list, TimeValue, int))
        """
        #
        # TODO in the Speect Python API I was not able to find a way
        #      to generate the wave incrementally
        #      so I essentially copy the subprocess call mechanism:
        #      generating wave data for each fragment,
        #      and concatenating them together
        #
        self.log(u"Calling TTS engine via Python...")
        try:
            # get sample rate and encoding
            du_nu, sample_rate, encoding, da_nu = self._synthesize_single_helper(
                text=u"Dummy text to get sample_rate",
                voice_code=self.DEFAULT_LANGUAGE
            )

            # open output file
            output_file = AudioFile(rconf=self.rconf, logger=self.logger)
            output_file.audio_format = encoding
            output_file.audio_channels = 1
            output_file.audio_sample_rate = sample_rate

            # create output
            anchors = []
            current_time = TimeValue("0.000")
            num = 0
            num_chars = 0
            fragments = text_file.fragments
            if backwards:
                fragments = fragments[::-1]
            for fragment in fragments:
                # language to voice code
                #
                # NOTE since voice_code is actually ignored
                # in _synthesize_single_helper(),
                # the value of voice_code is irrelevant
                #
                # however, in general you need to apply
                # the _language_to_voice_code() function that maps
                # the text language to a voice code
                #
                # here we apply the _language_to_voice_code() defined in super()
                # that sets voice_code = fragment.language
                #
                voice_code = self._language_to_voice_code(fragment.language)
                # synthesize and get the duration of the output file
                self.log([u"Synthesizing fragment %d", num])
                duration, sr_nu, enc_nu, data = self._synthesize_single_helper(
                    text=(fragment.filtered_text + u" "),
                    voice_code=voice_code
                )
                # store for later output
                anchors.append([current_time, fragment.identifier, fragment.text])
                # increase the character counter
                num_chars += fragment.characters
                # append new data
                self.log([u"Fragment %d starts at: %.3f", num, current_time])
                if duration > 0:
                    self.log([u"Fragment %d duration: %.3f", num, duration])
                    current_time += duration
                    # if backwards, we append the data reversed
                    output_file.add_samples(data, reverse=backwards)
                else:
                    self.log([u"Fragment %d has zero duration", num])
                # increment fragment counter
                num += 1
                # check if we must stop synthesizing because we have enough audio
                if (quit_after is not None) and (current_time > quit_after):
                    self.log([u"Quitting after reached duration %.3f", current_time])
                    break

            # if backwards, we need to reverse the audio samples again
            if backwards:
                output_file.reverse()

            # write output file
            self.log([u"Writing audio file '%s'", output_file_path])
            output_file.write(file_path=output_file_path)
        except Exception as exc:
            self.log_exc(u"An unexpected error occurred while calling TTS engine via Python", exc, False, None)
            return (False, None)

        # return output
        # NOTE anchors do not make sense if backwards
        self.log([u"Returning %d time anchors", len(anchors)])
        self.log([u"Current time %.3f", current_time])
        self.log([u"Synthesized %d characters", num_chars])
        self.log(u"Calling TTS engine via Python... done")
        return (True, (anchors, current_time, num_chars))

    def _synthesize_single_python(self, text, voice_code, output_file_path):
        """
        Synthesize a single text fragment via Python call.

        :rtype: tuple (result, (duration, sample_rate, encoding, data))
        """
        self.log(u"Synthesizing using Python call...")
        data = self._synthesize_single_helper(text, voice_code, output_file_path)
        return (True, data)

    def _synthesize_single_helper(self, text, voice_code, output_file_path=None):
        """
        This is an helper function to synthesize a single text fragment via Python call.

        The caller can choose whether the output file should be written to disk or not.

        :rtype: tuple (result, (duration, sample_rate, encoding, data))
        """
        #
        # NOTE in this example, we assume that the Speect voice data files
        #      are located in the same directory of this .py source file
        #      and that the voice JSON file is called "voice.json"
        #
        # NOTE the voice_code value is ignored in this example,
        #      but in general one might select a voice file to load,
        #      depending on voice_code
        #
        voice_json_path = gf.safe_str(gf.absolute_path("voice.json", __file__))
        voice = speect.SVoice(voice_json_path)
        utt = voice.synth(text)
        audio = utt.features["audio"]
        if output_file_path is not None:
            audio.save_riff(gf.safe_str(output_file_path))

        # get length and data using speect Python API
        waveform = audio.get_audio_waveform()
        audio_sample_rate = int(waveform["samplerate"])
        audio_length = TimeValue(audio.num_samples() / audio_sample_rate)
        audio_format = "pcm16"
        audio_samples = numpy.fromstring(waveform["samples"], dtype=numpy.int16).astype("float64") / 32768

        # return data
        return (audio_length, audio_sample_rate, audio_format, audio_samples)



