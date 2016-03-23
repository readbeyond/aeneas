#!/usr/bin/env python
# coding=utf-8

"""
This module contains the following classes:

* :class:`~aeneas.plotter.Plotter`, for plotting waveforms and labels to image files;
* :class:`~aeneas.plotter.PlotterColors`, enumerating colors;
* :class:`~aeneas.plotter.PlotElement`, representing a generic plot element;
* :class:`~aeneas.plotter.PlotTimeScale`, representing a time scale;
* :class:`~aeneas.plotter.PlotLabelset`, representing a set of labels (annotations);
* :class:`~aeneas.plotter.PlotWaveform`, representing a waveform.

.. note:: This module requires Python module ``PIL`` (``pip install Pillow``).

.. warning:: This module is likely to be refactored in a future version

.. versionadded:: 1.5.0
"""

from __future__ import absolute_import
from __future__ import print_function
from PIL import Image, ImageDraw, ImageFont
import math
import numpy

from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration
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

class PlotterColors(object):
    """
    Enumeration of colors for :class:`~aeneas.plotter.Plotter`.
    """

    AUDACITY_BACKGROUND_GREY = (192, 192, 192)
    """ Audacity background grey """

    AUDACITY_DARK_BLUE = (50, 50, 200)
    """ Audacity dark blue """

    AUDACITY_LIGHT_BLUE = (100, 100, 220)
    """ Audacity light blue """

    BLACK = (0, 0, 0)
    """ Black """

    BLUE = (0, 0, 255)
    """ Blue """

    GREEN = (0, 255, 0)
    """ Green """

    RED = (255, 0, 0)
    """ Red """

    WHITE = (255, 255, 255)
    """ White """



class Plotter(Loggable):
    """
    Plot waveforms and labels to image files.

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    TAG = u"Plotter"

    def __init__(self, rconf=None, logger=None):
        super(Plotter, self).__init__(rconf=rconf, logger=logger)
        self.waveform = None
        self.timescale = None
        self.labelsets = []

    def add_waveform(self, waveform):
        """
        Add a waveform to the plot.

        :param waveform: the waveform to be added
        :type  waveform: :class:`~aeneas.plotter.PlotWaveform`
        :raises: TypeError: if ``waveform`` is not an instance of :class:`~aeneas.plotter.PlotWaveform`
        """
        if not isinstance(waveform, PlotWaveform):
            self.log_exc(u"waveform must be an instance of PlotWaveform", None, True, TypeError)
        self.waveform = waveform
        self.log(u"Added waveform")

    def add_timescale(self, timescale):
        """
        Add a time scale to the plot.

        :param timescale: the timescale to be added
        :type  timescale: :class:`~aeneas.plotter.PlotTimeScale`
        :raises: TypeError: if ``timescale`` is not an instance of :class:`~aeneas.plotter.PlotTimeScale`
        """
        if not isinstance(timescale, PlotTimeScale):
            self.log_exc(u"timescale must be an instance of PlotTimeScale", None, True, TypeError)
        self.timescale = timescale
        self.log(u"Added timescale")

    def add_labelset(self, labelset):
        """
        Add a set of labels to the plot.

        :param labelset: the set of labels to be added
        :type  labelset: :class:`~aeneas.plotter.PlotLabelset`
        :raises: TypeError: if ``labelset`` is not an instance of :class:`~aeneas.plotter.PlotLabelset`
        """
        if not isinstance(labelset, PlotLabelset):
            self.log_exc(u"labelset must be an instance of PlotLabelset", None, True, TypeError)
        self.labelsets.append(labelset)
        self.log(u"Added labelset")

    def draw_png(self, output_file_path, h_zoom=5, v_zoom=30):
        """
        Draw the current plot to a PNG file.

        :param string output_path: the path of the output file to be written
        :param int h_zoom: the horizontal zoom
        :param int v_zoom: the vertical zoom
        :raises: ImportError: if module ``PIL`` cannot be imported
        :raises: OSError: if ``output_file_path`` cannot be written
        """
        # check that output_file_path can be written
        if not gf.file_can_be_written(output_file_path):
            self.log_exc(u"Cannot write to output file '%s'" % (output_file_path), None, True, OSError)

        # get widths and cumulative height, in modules
        widths = [ls.width for ls in self.labelsets]
        sum_height = sum([ls.height for ls in self.labelsets])
        if self.waveform is not None:
            widths.append(self.waveform.width)
            sum_height += self.waveform.height
        if self.timescale is not None:
            sum_height += self.timescale.height
        # in modules
        image_width = max(widths)
        image_height = sum_height
        # in pixels
        image_width_px = image_width * h_zoom
        image_height_px = image_height * v_zoom

        # build image object
        self.log([u"Building image with size (modules): %d %d", image_width, image_height])
        self.log([u"Building image with size (px):      %d %d", image_width_px, image_height_px])
        image_obj = Image.new("RGB", (image_width_px, image_height_px), color=PlotterColors.AUDACITY_BACKGROUND_GREY)
        current_y = 0
        if self.waveform is not None:
            self.log(u"Drawing waveform")
            self.waveform.draw_png(image_obj, h_zoom, v_zoom, current_y)
            current_y += self.waveform.height
        timescale_y = current_y
        if self.timescale is not None:
            # NOTE draw as the last thing
            #self.log(u"Drawing timescale")
            #self.timescale.draw_png(image_obj, h_zoom, v_zoom, current_y)
            current_y += self.timescale.height
        for labelset in self.labelsets:
            self.log(u"Drawing labelset")
            labelset.draw_png(image_obj, h_zoom, v_zoom, current_y)
            current_y += labelset.height
        if self.timescale is not None:
            self.log(u"Drawing timescale")
            self.timescale.draw_png(image_obj, h_zoom, v_zoom, timescale_y)
        self.log([u"Saving to file '%s'", output_file_path])
        image_obj.save(output_file_path)



class PlotElement(Loggable):
    """
    A generic element of a Plot.

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    FONT_PATH = gf.absolute_path("res/LiberationMono-Regular.ttf", __file__)
    """ Path of the font to be used for drawing """

    TICK_WIDTH = 2
    """ A tick will be drawn with (1 + 2 times this value) pixels """

    TEXT_MARGIN = 2
    """ Margin between text and anchor point, in pixels """

    TAG = u"PlotElement"

    def __init__(self, label=None, rconf=None, logger=None):
        super(PlotElement, self).__init__(rconf=rconf, logger=logger)
        self.label = label

    @property
    def height(self):
        """
        The height of this element, in modules.

        :rtype: int
        """
        return 0

    @property
    def width(self):
        """
        The width of this element, in modules.

        :rtype: int
        """
        return 0

    def text_bounding_box(self, size_pt, text):
        """
        Return the bounding box of the given text
        at the given font size.

        :param int size_pt: the font size in points
        :param string text: the text

        :rtype: tuple (width, height)
        """
        if size_pt == 12:
            mult = {"h": 9, "w_digit": 5, "w_space": 2}
        elif size_pt == 18:
            mult = {"h": 14, "w_digit": 9, "w_space": 2}
        num_chars = len(text)
        return (num_chars * mult["w_digit"] + (num_chars - 1) * mult["w_space"] + 1, mult["h"])



class PlotTimeScale(PlotElement):
    """
    A time scale.

    :param float max_time: the maximum length of the time scale
    :param int time_step: the step of the time scale numbers
    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    TAG = u"PlotTimeScale"

    def __init__(self, max_time, time_step=1, rconf=None, logger=None):
        super(PlotTimeScale, self).__init__(rconf=rconf, logger=logger)
        self.max_time = max_time
        self.time_step = time_step
        self.log(u"Created time scale with")
        self.log([u"  max_time:  %.3f", self.max_time])
        self.log([u"  time_step: %d", self.time_step])

    @property
    def height(self):
        return 1

    @property
    def width(self):
        return int(self.max_time / self.rconf.mws)

    def _time_string(self, value):
        """
        Get a suitable time string
        ("ss", "mm:ss", "hh:mm:ss"),
        according to the maximum time.

        :param int value: the time value
        :rtype: string
        """
        if self.max_time < 60:
            return "%02d" % (value)
        elif self.max_time < 3600:
            mm = value // 60
            ss = value - mm * 60
            return "%02d:%02d" % (mm, ss)
        hh = value // 3600
        mm = (value - hh * 3600) // 60
        ss = (value - hh * 3600 - mm * 60)
        return "%02d:%02d:%02d" % (hh, mm, ss)

    def draw_png(self, image, h_zoom, v_zoom, current_y):
        """
        Draw this time scale to PNG.

        :param image: the image to draw onto
        :param int h_zoom: the horizontal zoom
        :param int v_zoom: the vertical zoom
        :param int current_y: the current y offset, in modules
        :type  image: :class:`PIL.Image`
        """
        # PIL object
        draw = ImageDraw.Draw(image)
        mws = self.rconf.mws
        pixels_per_second = int(h_zoom / mws)
        current_y_px = current_y * v_zoom

        # create font, as tall as possible
        font_height_pt = 18
        font = ImageFont.truetype(self.FONT_PATH, font_height_pt)

        # draw a tick every self.time_step seconds
        for i in range(0, 1 + int(self.max_time), self.time_step):
            # base x position
            begin_px = i * pixels_per_second

            # tick
            left_px = begin_px - self.TICK_WIDTH
            right_px = begin_px + self.TICK_WIDTH
            top_px = current_y_px
            bottom_px = current_y_px + v_zoom
            draw.rectangle((left_px, top_px, right_px, bottom_px), fill=PlotterColors.BLACK)

            # text
            time_text = self._time_string(i)
            left_px = begin_px + self.TICK_WIDTH + self.TEXT_MARGIN
            top_px = current_y_px + (v_zoom - self.text_bounding_box(font_height_pt, time_text)[1]) // 2
            draw.text((left_px, top_px), time_text, PlotterColors.BLACK, font=font)



class PlotLabelset(PlotElement):
    """
    A set of labels.

    :param list labelset: a list of triples ``(begin, end, label)``
                          of type ``(float, float, string)``, times in seconds
    :param string label: a label for this set
    :param dict parameters: a dictionary holding drawing parameters
    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    DEFAULT_PARAMETERS = {
        "labels": False,
        "begin_time": False,
        "end_time": False,
        "begin_guide": False,
        "end_guide": False,
        "color": PlotterColors.BLACK
    }

    TAG = u"PlotLabelset"

    def __init__(self, labelset, label=None, parameters=None, rconf=None, logger=None):
        super(PlotLabelset, self).__init__(label=label, rconf=rconf, logger=logger)
        self.labelset = labelset
        self.parameters = dict(self.DEFAULT_PARAMETERS) if parameters is None else parameters
        self.log(u"Created label set with")
        self.log([u"  label:            %s", self.label])
        self.log([u"  number of labels: %d", len(self.labelset)])
        self.log([u"  parameters:       %s", self.parameters])

    @property
    def height(self):
        return 2

    @property
    def width(self):
        try:
            return int(self.labelset[-1][1] / self.rconf.mws)
        except:
            return 0

    def draw_png(self, image, h_zoom, v_zoom, current_y):
        """
        Draw this set of labels to PNG.

        :param image: the image to draw onto
        :param int h_zoom: the horizontal zoom
        :param int v_zoom: the vertical zoom
        :param int current_y: the current y offset, in modules
        :type  image: :class:`PIL.Image`
        """
        # PIL object
        draw = ImageDraw.Draw(image)
        mws = self.rconf.mws
        pixels_per_second = int(h_zoom / mws)

        # font for begin/end times
        time_font_height_pt = 12
        time_font = ImageFont.truetype(self.FONT_PATH, time_font_height_pt)

        # font for labels
        label_font_height_pt = 18
        label_font = ImageFont.truetype(self.FONT_PATH, label_font_height_pt)

        current_y_px = current_y * v_zoom + 0.25 * v_zoom
        for (begin, end, label) in self.labelset:
            # base x position
            begin_px = int(begin * pixels_per_second)
            end_px = int(end * pixels_per_second)

            # select color for the horizontal bar
            if label == "speech":
                color = PlotterColors.RED
            elif label == "nonspeech":
                color = PlotterColors.GREEN
            else:
                color = self.parameters["color"]

            # horizontal bar
            bar_top_px = current_y_px + v_zoom * 0.5 - self.TICK_WIDTH
            bar_bottom_px = bar_top_px + 2 * self.TICK_WIDTH
            bar_left_px = begin_px
            bar_right_px = end_px
            draw.rectangle((bar_left_px, bar_top_px, bar_right_px, bar_bottom_px), fill=color)

            # left guide
            if self.parameters["begin_guide"]:
                top_px = 0
                bottom_px = current_y_px + v_zoom
                left_px = begin_px
                draw.rectangle((left_px, top_px, left_px, bottom_px), fill=color)

            # left tick
            top_px = current_y_px
            bottom_px = current_y_px + v_zoom
            left_px = begin_px
            right_px = begin_px + self.TICK_WIDTH
            draw.rectangle((left_px, top_px, right_px, bottom_px), fill=PlotterColors.BLACK)

            # right guide
            if self.parameters["end_guide"]:
                top_px = 0
                bottom_px = current_y_px + v_zoom
                left_px = end_px
                draw.rectangle((left_px, top_px, left_px, bottom_px), fill=color)

            # right tick
            top_px = current_y_px
            bottom_px = current_y_px + v_zoom
            left_px = end_px - self.TICK_WIDTH
            right_px = end_px
            draw.rectangle((left_px, top_px, right_px, bottom_px), fill=PlotterColors.BLACK)

            # begin time
            if self.parameters["begin_time"]:
                sb = ("%.03f" % (begin - int(begin)))[2:]
                left_px = begin_px + self.TICK_WIDTH + self.TEXT_MARGIN
                top_px = current_y_px - self.TEXT_MARGIN
                draw.text((left_px, top_px), sb, PlotterColors.BLACK, font=time_font)

            # end time
            if self.parameters["end_time"]:
                se = ("%.03f" % (end - int(end)))[2:]
                left_px = end_px - self.TEXT_MARGIN - self.TICK_WIDTH - self.text_bounding_box(time_font_height_pt, se)[0]
                top_px = current_y_px + v_zoom - self.text_bounding_box(time_font_height_pt, sb)[1]
                draw.text((left_px, top_px), se, PlotterColors.BLACK, font=time_font)

            # interval label
            if self.parameters["labels"]:
                left_px = begin_px + (end_px - begin_px - self.text_bounding_box(label_font_height_pt, label)[0]) // 2
                top_px = current_y_px + v_zoom
                draw.text((left_px, top_px), label, PlotterColors.BLACK, font=label_font)

        # label
        left_px = 0
        top_px = current_y_px + v_zoom
        if self.label is not None:
            draw.text((left_px, top_px), self.label, PlotterColors.BLACK, font=label_font)



class PlotWaveform(PlotElement):
    """
    An audio file waveform.

    :param audio_file: the audio file from which the waveform must be drawn
    :type  audio_file: :class:`~aeneas.audiofile.AudioFile`
    :param string label: a label for this waveform
    :param bool fast: if ``True``, plot fast (only max, mirrored)
    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    TAG = u"PlotWaveform"

    def __init__(self, audio_file, label=None, fast=False, rconf=None, logger=None):
        super(PlotWaveform, self).__init__(label=label, rconf=rconf, logger=logger)
        self.audio_file = audio_file
        self.fast = fast
        self.log(u"Created waveform with")
        self.log([u"  label:        %s", self.label])
        self.log([u"  audio_length: %.3f", self.audio_file.audio_length])
        self.log([u"  fast:         %s", str(self.fast)])

    @property
    def height(self):
        return 10

    @property
    def width(self):
        try:
            return int(self.audio_file.audio_length / self.rconf.mws)
        except:
            return 0

    def draw_png(self, image, h_zoom, v_zoom, current_y):
        """
        Draw this waveform to PNG.

        :param image: the image to draw onto
        :param int h_zoom: the horizontal zoom
        :param int v_zoom: the vertical zoom
        :param int current_y: the current y offset, in modules
        :type  image: :class:`PIL.Image`
        """
        draw = ImageDraw.Draw(image)
        mws = self.rconf.mws
        rate = self.audio_file.audio_sample_rate
        samples = self.audio_file.audio_samples
        duration = self.audio_file.audio_length

        current_y_px = current_y * v_zoom
        half_waveform_px = (self.height // 2) * v_zoom
        zero_y_px = current_y_px + half_waveform_px

        samples_per_pixel = int(rate * mws / h_zoom)
        pixels_per_second = int(h_zoom / mws)
        windows = len(samples) // samples_per_pixel

        if self.label is not None:
            font_height_pt = 18
            font = ImageFont.truetype(self.FONT_PATH, font_height_pt)
            draw.text((0, current_y_px), self.label, PlotterColors.BLACK, font=font)

        for i in range(windows):
            x = i * samples_per_pixel
            pos = numpy.clip(samples[x:x+samples_per_pixel], 0.0, 1.0)
            mpos = numpy.max(pos) * half_waveform_px
            if self.fast:
                # just draw a simple version, mirroring max positive samples
                draw.line((i, zero_y_px + mpos, i, zero_y_px - mpos), fill=PlotterColors.AUDACITY_DARK_BLUE, width=1)
            else:
                # draw a better version, taking min and std of positive and negative samples
                neg = numpy.clip(samples[x:x+samples_per_pixel], -1.0, 0.0)
                spos = numpy.std(pos) * half_waveform_px
                sneg = numpy.std(neg) * half_waveform_px
                mneg = numpy.min(neg) * half_waveform_px
                draw.line((i, zero_y_px - mneg, i, zero_y_px - mpos), fill=PlotterColors.AUDACITY_DARK_BLUE, width=1)
                draw.line((i, zero_y_px + sneg, i, zero_y_px - spos), fill=PlotterColors.AUDACITY_LIGHT_BLUE, width=1)



