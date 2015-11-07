#!/usr/bin/env python
# coding=utf-8

"""
Enumeration of the available algorithms to adjust
the boundary point between two fragments.

.. versionadded:: 1.0.4
"""

import copy

from aeneas.logger import Logger

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

class AdjustBoundaryAlgorithm(object):
    """
    Enumeration of the available algorithms to adjust
    the boundary point between two consecutive fragments.

    :param algorithm: the boundary adjustment algorithm to be used
    :type  algorithm: string (from :class:`aeneas.adjustboundaryalgorithm.AdjustBoundaryAlgorithm` enumeration)
    :param text_map: a text map list [[start, end, id, text], ..., []]
    :type  text_map: list
    :param speech: a list of time intervals [[s_1, e_1,], ..., [s_k, e_k]]
                   containing speech
    :type  speech: list
    :param nonspeech: a list of time intervals [[s_1, e_1,], ..., [s_j, e_j]]
                      not containing speech
    :type  nonspeech: list
    :param value: an optional parameter to be passed
                  to the boundary adjustment algorithm,
                  it will be converted (to int, to float) as needed,
                  depending on the selected algorithm
    :type  value: string
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`

    :raises ValueError: if one of `text_map`, `speech` or `nonspeech` is `None` or `algorithm` value is not allowed
    """

    AFTERCURRENT = "aftercurrent"
    """ Set the boundary at ``value`` seconds
    after the end of the current fragment """

    AUTO = "auto"
    """ Auto (no adjustment) """

    BEFORENEXT = "beforenext"
    """ Set the boundary at ``value`` seconds
    before the beginning of the next fragment """

    OFFSET = "offset"
    """ Offset the current boundaries by ``value`` seconds

    .. versionadded:: 1.1.0
    """

    PERCENT = "percent"
    """ Set the boundary at ``value`` percent of
    the nonspeech interval between the current and the next fragment """

    RATE = "rate"
    """ Adjust boundaries trying to respect the
    ``value`` characters/second constraint """

    RATEAGGRESSIVE = "rateaggressive"
    """ Adjust boundaries trying to respect the
    ``value`` characters/second constraint (aggressive mode)

    .. versionadded:: 1.1.0
    """

    ALLOWED_VALUES = [
        AFTERCURRENT,
        AUTO,
        BEFORENEXT,
        OFFSET,
        PERCENT,
        RATE,
        RATEAGGRESSIVE
    ]
    """ List of all the allowed values """

    DEFAULT_MAX_RATE = 21.0
    """ Default max rate (used only when ``RATE`` or ``RATEAGGRESSIVE``
    algorithms are used) """

    DEFAULT_PERCENT = 50
    """ Default percent value (used only when ``PERCENT`` algorithm is used) """

    TOLERANCE = 0.001
    """ Tolerance when comparing floats """

    TAG = "AdjustBoundaryAlgorithm"

    def __init__(
            self,
            algorithm,
            text_map,
            speech,
            nonspeech,
            value=None,
            logger=None
        ):
        if algorithm not in self.ALLOWED_VALUES:
            raise ValueError("Algorithm value not allowed")
        if text_map is None:
            raise ValueError("Text map is None")
        if speech is None:
            raise ValueError("Speech list is None")
        if nonspeech is None:
            raise ValueError("Nonspeech list is None")
        self.algorithm = algorithm
        self.text_map = copy.deepcopy(text_map)
        self.speech = speech
        self.nonspeech = nonspeech
        self.value = value
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()
        self._parse_value()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def _parse_value(self):
        """
        Parse the self.value value
        """
        if self.algorithm == self.AUTO:
            return
        elif self.algorithm == self.PERCENT:
            try:
                self.value = int(self.value)
            except ValueError:
                self.value = self.DEFAULT_PERCENT
            self.value = max(min(self.value, 100), 0)
        else:
            try:
                self.value = float(self.value)
            except ValueError:
                self.value = 0.0
            if (
                    (self.value <= 0) and
                    (self.algorithm in [self.RATE, self.RATEAGGRESSIVE])
            ):
                self.value = self.DEFAULT_MAX_RATE

    def adjust(self):
        """
        Adjust the boundaries of the text map.

        :rtype: list of intervals
        """
        if self.algorithm == self.AUTO:
            return self._adjust_auto()
        elif self.algorithm == self.AFTERCURRENT:
            return self._adjust_aftercurrent()
        elif self.algorithm == self.BEFORENEXT:
            return self._adjust_beforenext()
        elif self.algorithm == self.OFFSET:
            return self._adjust_offset()
        elif self.algorithm == self.PERCENT:
            return self._adjust_percent()
        elif self.algorithm == self.RATE:
            return self._adjust_rate(False)
        elif self.algorithm == self.RATEAGGRESSIVE:
            return self._adjust_rate(True)
        return self.text_map

    def _adjust_auto(self):
        self._log("Called _adjust_auto: returning text_map unchanged")
        return self.text_map

    def _adjust_offset(self):
        self._log("Called _adjust_offset")
        try:
            for index in range(1, len(self.text_map)):
                current = self.text_map[index]
                previous = self.text_map[index - 1]
                if self.value >= 0:
                    offset = min(self.value, current[1] - current[0])
                else:
                    offset = -min(-self.value, previous[1] - previous[0])
                previous[1] += offset
                current[0] += offset
        except:
            self._log("Exception in _adjust_offset: returning text_map unchanged")
        return self.text_map

    def _adjust_percent(self):
        def new_time(current_boundary, nsi):
            duration = nsi[1] - nsi[0]
            percent = self.value / 100.0
            return nsi[0] + duration * percent
        return self._adjust_on_nsi(new_time)

    def _adjust_aftercurrent(self):
        def new_time(current_boundary, nsi):
            duration = nsi[1] - nsi[0]
            try:
                delay = max(min(self.value, duration), 0)
                if delay == 0:
                    return current_boundary
                return nsi[0] + delay
            except:
                return current_boundary
        return self._adjust_on_nsi(new_time)

    def _adjust_beforenext(self):
        def new_time(current_boundary, nsi):
            duration = nsi[1] - nsi[0]
            try:
                delay = max(min(self.value, duration), 0)
                if delay == 0:
                    return current_boundary
                return nsi[1] - delay
            except:
                return current_boundary
        return self._adjust_on_nsi(new_time)

    def _adjust_on_nsi(self, new_time_function):
        nsi_index = 0
        # TODO numpy-fy this loop?
        for index in range(len(self.text_map) - 1):
            current_boundary = self.text_map[index][1]
            self._log(["current_boundary: %.3f", current_boundary])
            # the tolerance comparison seems necessary
            while (
                    (nsi_index < len(self.nonspeech)) and
                    (self.nonspeech[nsi_index][1] + self.TOLERANCE <= current_boundary)
                ):
                nsi_index += 1
            nsi = None
            if (
                    (nsi_index < len(self.nonspeech)) and
                    (current_boundary >= self.nonspeech[nsi_index][0] - self.TOLERANCE)
                ):
                nsi = self.nonspeech[nsi_index]
                nsi_index += 1
            if nsi:
                self._log(["  in interval %.3f %.3f", nsi[0], nsi[1]])
                new_time = new_time_function(current_boundary, nsi)
                self._log(["  new_time: %.3f", new_time])
                new_start = self.text_map[index][0]
                new_end = self.text_map[index + 1][1]
                if self._time_in_interval(new_time, new_start, new_end):
                    self._log(["  updating %.3f => %.3f", current_boundary, new_time])
                    self.text_map[index][1] = new_time
                    self.text_map[index + 1][0] = new_time
                else:
                    #print "  new_time outside: no adjustment performed"
                    self._log("  new_time outside: no adjustment performed")
            else:
                #print "  no nonspeech interval found: no adjustment performed"
                self._log("  no nonspeech interval found: no adjustment performed")
        return self.text_map

    def _len(self, string):
        """
        Return the length of the given string.
        If it is greater than 2 times the self.value (= user max rate),
        one space will become a newline,
        and hence we do not count it
        (e.g., value = 21 => max 42 chars per line).

        :param string: the string to be counted
        :type  string: string
        :rtype: int
        """
        # TODO this should depend on the number of lines
        #      in the text fragment; current code assumes
        #      at most 2 lines of at most value characters each
        #      (the effect of this finesse is negligible in practice)
        if string is None:
            return 0
        length = len(string)
        if length > 2 * self.value:
            length -= 1
        return length

    def _time_in_interval(self, time, start, end):
        """
        Decides whether the given time is within the given interval.

        :param time: a time value
        :type  time: float
        :param start: the start of the interval
        :type  start: float
        :param end: the end of the interval
        :type  end: float
        :rtype: bool
        """
        return (time >= start) and (time <= end)

    # TODO a more efficient search (e.g., binary) is possible
    # the tolerance comparison seems necessary
    def _find_interval_containing(self, intervals, time):
        """
        Return the interval containing the given time,
        or None if no such interval exists.

        :param intervals: a list of time intervals
                          [[s_1, e_1], ..., [s_k, e_k]]
        :type  intervals: list of lists
        :param time: a time value
        :type  time: float
        :rtype: a time interval ``[s, e]`` or ``None``
        """
        for interval in intervals:
            start = interval[0] - self.TOLERANCE
            end = interval[1] + self.TOLERANCE
            if self._time_in_interval(time, start, end):
                return interval
        return None

    def _compute_rate_raw(self, start, end, length):
        """
        Compute the rate of a fragment, that is,
        the number of characters per second.

        :param start: the start time
        :type  start: float
        :param end: the end time
        :type  end: float
        :param length: the number of character (possibly adjusted) of the text
        :type length: int
        :rtype: float
        """
        duration = end - start
        if duration > 0:
            return length / duration
        return 0

    def _compute_rate(self, index):
        """
        Compute the rate of a fragment, that is,
        the number of characters per second.

        :param index: the index of the fragment in the text map
        :type  index: int
        :rtype: float
        """
        if (index < 0) or (index >= len(self.text_map)):
            return 0
        fragment = self.text_map[index]
        start = fragment[0]
        end = fragment[1]
        length = self._len(fragment[3])
        return self._compute_rate_raw(start, end, length)

    def _compute_slack(self, index):
        """
        Return the slack of a fragment, that is,
        the difference between the current duration
        of the fragment and the duration it should have
        if its rate was exactly self.value (= max rate)

        If the slack is positive, the fragment
        can be shrinken; if the slack is negative,
        the fragment should be stretched.

        The returned value can be None,
        in case the index is out of self.text_map bounds.

        :param index: the index of the fragment in the text map
        :type  index: int
        :rtype: float
        """
        if (index < 0) or (index >= len(self.text_map)):
            return None
        fragment = self.text_map[index]
        start = fragment[0]
        end = fragment[1]
        length = self._len(fragment[3])
        duration = end - start
        return duration - (length / self.value)

    def _adjust_rate(self, aggressive=False):
        faster = []

        # TODO numpy-fy this loop?
        for index in range(len(self.text_map)):
            fragment = self.text_map[index]
            self._log(["Fragment %d", index])
            rate = self._compute_rate(index)
            self._log(["  %.3f %.3f => %.3f", fragment[0], fragment[1], rate])
            if rate > self.value:
                self._log("  too fast")
                faster.append(index)

        if len(self.text_map) == 1:
            self._log("Only one fragment, and it is too fast")
            return self.text_map

        if len(faster) == 0:
            self._log(["No fragment faster than max rate %.3f", self.value])
            return self.text_map

        # TODO numpy-fy this loop?
        # try fixing faster fragments
        self._log("Fixing faster fragments...")
        for index in faster:
            self._log(["Fixing faster fragment %d ...", index])
            if aggressive:
                try:
                    self._rateaggressive_fix_fragment(index)
                except:
                    self._log("Exception in _rateaggressive_fix_fragment")
            else:
                try:
                    self._rate_fix_fragment(index)
                except:
                    self._log("Exception in _rate_fix_fragment")
            self._log(["Fixing faster fragment %d ... done", index])
        self._log("Fixing faster fragments... done")
        return self.text_map

    def _rate_fix_fragment(self, index):
        """
        Fix index-th fragment using the rate algorithm (standard variant).
        """
        succeeded = False
        current = self.text_map[index]
        current_start = current[0]
        current_end = current[1]
        current_rate = self._compute_rate(index)
        previous_slack = self._compute_slack(index - 1)
        current_slack = self._compute_slack(index)
        next_slack = self._compute_slack(index + 1)
        if previous_slack is not None:
            previous = self.text_map[index - 1]
            self._log(["  previous:       %.3f %.3f => %.3f", previous[0], previous[1], self._compute_rate(index - 1)])
            self._log(["  previous slack: %.3f", previous_slack])
        if current_slack is not None:
            self._log(["  current:        %.3f %.3f => %.3f", current_start, current_end, current_rate])
            self._log(["  current  slack: %.3f", current_slack])
        if next_slack is not None:
            nextf = self.text_map[index]
            self._log(["  next:           %.3f %.3f => %.3f", nextf[0], nextf[1], self._compute_rate(index + 1)])
            self._log(["  next     slack: %.3f", next_slack])

        # try expanding into the previous fragment
        new_start = current_start
        new_end = current_end
        if (previous_slack is not None) and (previous_slack > 0):
            self._log("  can expand into previous")
            nsi = self._find_interval_containing(self.nonspeech, current[0])
            previous = self.text_map[index - 1]
            if nsi is not None:
                if nsi[0] > previous[0]:
                    self._log(["  found suitable nsi: %.3f %.3f", nsi[0], nsi[1]])
                    previous_slack = min(current[0] - nsi[0], previous_slack)
                    self._log(["  previous slack after min: %.3f", previous_slack])
                    if previous_slack + current_slack >= 0:
                        self._log("  enough slack to completely fix")
                        steal_from_previous = -current_slack
                        succeeded = True
                    else:
                        self._log("  not enough slack to completely fix")
                        steal_from_previous = previous_slack
                    new_start = current_start - steal_from_previous
                    self.text_map[index - 1][1] = new_start
                    self.text_map[index][0] = new_start
                    new_rate = self._compute_rate(index)
                    self._log(["    old: %.3f %.3f => %.3f", current_start, current_end, current_rate])
                    self._log(["    new: %.3f %.3f => %.3f", new_start, new_end, new_rate])
                else:
                    self._log("  nsi found is not suitable")
            else:
                self._log("  no nsi found")
        else:
            self._log("  cannot expand into previous")

        if succeeded:
            self._log("  succeeded: returning")
            return

        # recompute current fragment
        current_rate = self._compute_rate(index)
        current_slack = self._compute_slack(index)
        current_rate = self._compute_rate(index)

        # try expanding into the next fragment
        new_start = current_start
        new_end = current_end
        if (next_slack is not None) and (next_slack > 0):
            self._log("  can expand into next")
            nsi = self._find_interval_containing(self.nonspeech, current[1])
            previous = self.text_map[index - 1]
            if nsi is not None:
                if nsi[0] > previous[0]:
                    self._log(["  found suitable nsi: %.3f %.3f", nsi[0], nsi[1]])
                    next_slack = min(nsi[1] - current[1], next_slack)
                    self._log(["  next slack after min: %.3f", next_slack])
                    if next_slack + current_slack >= 0:
                        self._log("  enough slack to completely fix")
                        steal_from_next = -current_slack
                        succeeded = True
                    else:
                        self._log("  not enough slack to completely fix")
                        steal_from_next = next_slack
                    new_end = current_end + steal_from_next
                    self.text_map[index][1] = new_end
                    self.text_map[index + 1][0] = new_end
                    new_rate = self._compute_rate(index)
                    self._log(["    old: %.3f %.3f => %.3f", current_start, current_end, current_rate])
                    self._log(["    new: %.3f %.3f => %.3f", new_start, new_end, new_rate])
                else:
                    self._log("  nsi found is not suitable")
            else:
                self._log("  no nsi found")
        else:
            self._log("  cannot expand into next")

        if succeeded:
            self._log("  succeeded: returning")
            return

        self._log("  not succeeded, returning")

    def _rateaggressive_fix_fragment(self, index):
        """
        Fix index-th fragment using the rate algorithm (aggressive variant).
        """
        current = self.text_map[index]
        current_start = current[0]
        current_end = current[1]
        current_rate = self._compute_rate(index)
        previous_slack = self._compute_slack(index - 1)
        current_slack = self._compute_slack(index)
        next_slack = self._compute_slack(index + 1)
        if previous_slack is not None:
            self._log(["  previous slack: %.3f", previous_slack])
        if current_slack is not None:
            self._log(["  current  slack: %.3f", current_slack])
        if next_slack is not None:
            self._log(["  next     slack: %.3f", next_slack])
        steal_from_previous = 0
        steal_from_next = 0
        if (
                (previous_slack is not None) and
                (next_slack is not None) and
                (previous_slack > 0) and
                (next_slack > 0)
            ):
            self._log("  can expand into both previous and next")
            total_slack = previous_slack + next_slack
            self._log(["  total    slack: %.3f", total_slack])
            if total_slack + current_slack >= 0:
                self._log("  enough total slack to completely fix")
                # partition the needed slack proportionally
                previous_percentage = previous_slack / total_slack
                self._log(["    previous percentage: %.3f", previous_percentage])
                steal_from_previous = -current_slack * previous_percentage
                steal_from_next = -current_slack - steal_from_previous
            else:
                self._log("  not enough total slack to completely fix")
                # consume all the available slack
                steal_from_previous = previous_slack
                steal_from_next = next_slack
        elif (previous_slack is not None) and (previous_slack > 0):
            self._log("  can expand into previous only")
            if previous_slack + current_slack >= 0:
                self._log("  enough previous slack to completely fix")
                steal_from_previous = -current_slack
            else:
                self._log("  not enough previous slack to completely fix")
                steal_from_previous = previous_slack
        elif (next_slack is not None) and (next_slack > 0):
            self._log("  can expand into next only")
            if next_slack + current_slack >= 0:
                self._log("  enough next slack to completely fix")
                steal_from_next = -current_slack
            else:
                self._log("  not enough next slack to completely fix")
                steal_from_next = next_slack
        else:
            self._log(["  fragment %d cannot be fixed", index])

        self._log(["    steal from previous: %.3f", steal_from_previous])
        self._log(["    steal from next:     %.3f", steal_from_next])
        new_start = current_start - steal_from_previous
        new_end = current_end + steal_from_next
        if index - 1 >= 0:
            self.text_map[index - 1][1] = new_start
        self.text_map[index][0] = new_start
        self.text_map[index][1] = new_end
        if index + 1 < len(self.text_map):
            self.text_map[index + 1][0] = new_end
        new_rate = self._compute_rate(index)
        self._log(["    old: %.3f %.3f => %.3f", current_start, current_end, current_rate])
        self._log(["    new: %.3f %.3f => %.3f", new_start, new_end, new_rate])



