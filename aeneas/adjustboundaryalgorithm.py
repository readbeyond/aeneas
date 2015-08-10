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
    Copyright 2013-2015, ReadBeyond Srl (www.readbeyond.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.0.4"
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
    """

    AFTERCURRENT = "aftercurrent"
    """ Set the boundary at ``value`` seconds
    after the end of the current fragment """

    AUTO = "auto"
    """ Auto (no adjustment) """

    BEFORENEXT = "beforenext"
    """ Set the boundary at ``value`` seconds
    before the beginning of the next fragment """

    PERCENT = "percent"
    """ Set the boundary at ``value`` percent of
    the nonspeech interval between the current and the next fragment """

    RATE = "rate"
    """ Adjust boundaries trying to respect the
    ``value`` characters/second constraint """

    ALLOWED_VALUES = [
        AFTERCURRENT,
        AUTO,
        BEFORENEXT,
        PERCENT,
        RATE
    ]
    """ List of all the allowed values """

    DEFAULT_MAX_RATE = 21.0
    """ Default max rate (used only when RATE algorithm is used) """

    TAG = "AdjustBoundaryAlgorithm"

    def __init__(self, algorithm, text_map, speech, nonspeech, value=None, logger=None):
        self.algorithm = algorithm
        self.text_map = copy.deepcopy(text_map)
        self.speech = speech
        self.nonspeech = nonspeech
        self.value = value
        self.logger = logger
        self.max_rate = self.DEFAULT_MAX_RATE
        if self.logger == None:
            self.logger = Logger()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def adjust(self):
        """
        Adjust the boundaries of the text map.

        :rtype: list of intervals
        """
        if self.text_map == None:
            # TODO raise instead?
            return None
        if self.algorithm == self.AUTO:
            return self.text_map
        elif self.algorithm == self.RATE:
            return self._adjust_rate()
        elif self.algorithm == self.PERCENT:
            return self._adjust_percent()
        elif self.algorithm == self.AFTERCURRENT:
            return self._adjust_aftercurrent()
        elif self.algorithm == self.BEFORENEXT:
            return self._adjust_beforenext()
        return self.text_map

    def _adjust_percent(self):
        def new_time(current_boundary, nsi):
            duration = nsi[1] - nsi[0]
            try:
                percent = max(min(int(self.value), 100), 0) / 100.0
            except:
                percent = 0.500
            return nsi[0] + duration * percent
        return self._adjust_on_nsi(new_time)

    def _adjust_aftercurrent(self):
        def new_time(current_boundary, nsi):
            duration = nsi[1] - nsi[0]
            try:
                delay = max(min(float(self.value), duration), 0)
                return nsi[0] + delay
            except:
                return current_boundary
        return self._adjust_on_nsi(new_time)

    def _adjust_beforenext(self):
        def new_time(current_boundary, nsi):
            duration = nsi[1] - nsi[0]
            try:
                delay = max(min(float(self.value), duration), 0)
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
            # the -/+ 0.001 tolerance comparison seems necessary
            while (
                    (nsi_index < len(self.nonspeech)) and
                    (self.nonspeech[nsi_index][1] + 0.001 <= current_boundary)
                ):
                nsi_index += 1
            nsi = None
            if (
                    (nsi_index < len(self.nonspeech)) and
                    (current_boundary >= self.nonspeech[nsi_index][0] - 0.001)
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

    def _time_in_interval(self, time, start, end):
        return (time >= start) and (time <= end)

    def _len(self, string):
        # return the length of the string
        # if more than 2 times the max_rate,
        # one space will become a newline
        # hence we do not count it
        # e.g., max_rate = 21 => max 42 chars per line
        #
        # TODO this should depend on the number of lines
        #      in the text fragment; current code assumes
        #      at most 2 lines of at most max_rate characters each
        #      (the effect of this finesse is negligible in practice)
        if string is None:
            return 0
        length = len(string)
        if length > 2 * self.max_rate:
            length -= 1
        return length

    # TODO a more efficient search (e.g., binary) is possible
    # the -/+ 0.001 tolerance comparison seems necessary
    def _find_interval_containing(self, intervals, time):
        for interval in intervals:
            start = interval[0] - 0.001
            end = interval[1] + 0.001
            if self._time_in_interval(time, start, end):
                return interval
        return None

    def _compute_rate(self, fragment=None, start=None, end=None, num=None):
        if fragment:
            start = fragment[0]
            end = fragment[1]
            num = self._len(fragment[3])
        duration = end - start
        if duration > 0:
            return num / (end - start)
        return 0

    def _adjust_rate(self):
        try:
            self.max_rate = float(self.value)
        except:
            pass 
        faster = []
        # TODO numpy-fy this loop?
        for index in range(len(self.text_map)):
            fragment = self.text_map[index]
            self._log(["Fragment %d", index])
            rate = self._compute_rate(fragment=fragment)
            self._log(["  %.3f %.3f => %.3f", fragment[0], fragment[1], rate])
            if rate > self.max_rate:
                self._log("  too fast")
                faster.append(index)

        if len(self.text_map) == 1:
            self._log("Only one fragment, and it is too fast")
            return self.text_map

        if len(faster) == 0:
            self._log(["No fragment faster than max rate %.3f", self.max_rate])
            #return text_map

        # TODO numpy-fy this loop?
        # try fixing faster fragments
        for index in faster:
            self._log(["Faster fragment %d", index])
            # determine which direction we can expand the fragment
            expand_back = True
            expand_forward = True
            if index == 0:
                self._log("first fragment => only choice is expanding forward")
                expand_back = False
            if index == len(self.text_map) - 1:
                self._log("last fragment => only choice is expanding backward")
                expand_forward = False
            if (index + 1) in faster:
                self._log("the next one is also faster => only choice is expanding backward")
                expand_forward = False
            # try to fix the current faster fragment
            succeeded = False
            if expand_back:
                # first, try expading backward, if possible
                try:
                    self._log("  Trying to expand backward")
                    succeeded = self._expand_backward(index)
                except:
                    self._log("Exception in _expand_backward")
            if (not succeeded) and (expand_forward):
                # if not succeeded, try expanding forward, if possible
                try:
                    self._log("  Trying to expand forward")
                    succeeded = self._expand_forward(index)
                except:
                    self._log("Exception in _expand_forward")
            if not succeeded:
                self._log(["Not succeeded in fixing fragment %d", index])
        
        return self.text_map

    # TODO unify with _expand_forward
    def _expand_backward(self, index):
        self._log(["Expanding backward fragment %d", index])
        previous = self.text_map[index - 1]
        previous_rate = self._compute_rate(fragment=previous)
        self._log(["  previous: %.3f %.3f => %.3f", previous[0], previous[1], previous_rate])
        current = self.text_map[index]
        current_rate = self._compute_rate(fragment=current)
        self._log(["  current: %.3f %.3f => %.3f", current[0], current[1], current_rate])
        nsi = self._find_interval_containing(self.nonspeech, current[0])
        if nsi:
            if nsi[0] > previous[0]:
                self._log(["  found suitable nsi starting at %.3f", nsi[0]])
                new_boundary = current[0]
                satisfied = False
                # TODO can we perform a direct computation
                # to find the "optimal" boundary?
                while (not satisfied) and (nsi[0] <= new_boundary):
                    self._log(["   evaluating new_boundary at %.3f", new_boundary])
                    if self._compute_rate(start=previous[0], end=new_boundary, num=len(previous[3])) <= self.max_rate:
                        if self._compute_rate(start=new_boundary, end=current[1], num=len(current[3])) <= self.max_rate:
                            self._log("   current rate satisfied")
                            satisfied = True
                        else:
                            self._log("   current rate not satisfied")
                            new_boundary -= 0.001
                    else:
                        self._log("   previous rate not satisfied")
                        break
                new_boundary = max(new_boundary, nsi[0])
                self.text_map[index - 1][1] = new_boundary
                self.text_map[index][0] = new_boundary
                self._log(["   new boundary set at %.3f", new_boundary])
                self._log(["   new previous rate   %.3f", self._compute_rate(fragment=self.text_map[index - 1])])
                self._log(["   new current  rate   %.3f", self._compute_rate(fragment=self.text_map[index])])
                self._log(["   current fragment fixed? %d", satisfied])
                return satisfied
            else:
                self._log("  nsi found is not suitable")
        else:
            self._log("  no nsi found")
        self._log("  current fragment not fixed")
        return False

    # TODO unify with _expand_backward
    def _expand_forward(self, index):
        self._log(["Expanding forward fragment %d", index])
        current = self.text_map[index]
        current_rate = self._compute_rate(fragment=current)
        self._log(["  current: %.3f %.3f => %.3f", current[0], current[1], current_rate])
        nextf = self.text_map[index + 1]
        nextf_rate = self._compute_rate(fragment=nextf)
        self._log(["  next: %.3f %.3f => %.3f", nextf[0], nextf[1], nextf_rate])
        nsi = self._find_interval_containing(self.nonspeech, current[1])
        if nsi:
            if nsi[1] < nextf[1]:
                self._log(["  found suitable nsi ending at %.3f", nsi[1]])
                new_boundary = current[1]
                satisfied = False
                # TODO can we perform a direct computation
                # to find the "optimal" boundary?
                while (not satisfied) and (new_boundary <= nsi[1]):
                    self._log(["   evaluating new_boundary at %.3f", new_boundary])
                    if self._compute_rate(start=new_boundary, end=nextf[1], num=len(nextf[3])) <= self.max_rate:
                        if self._compute_rate(start=current[0], end=new_boundary, num=len(current[3])) <= self.max_rate:
                            self._log("   current rate satisfied")
                            satisfied = True
                        else:
                            self._log("   current rate not satisfied")
                            new_boundary += 0.001
                    else:
                        self._log("   next rate not satisfied")
                        break
                new_boundary = min(new_boundary, nsi[1])
                self.text_map[index][1] = new_boundary
                self.text_map[index + 1][0] = new_boundary
                self._log(["   new boundary set at %.3f", new_boundary])
                self._log(["   new current rate    %.3f", self._compute_rate(fragment=self.text_map[index])])
                self._log(["   new next    rate    %.3f", self._compute_rate(fragment=self.text_map[index + 1])])
                self._log(["   current fragment fixed? %d", satisfied])
                return satisfied
            else:
                self._log("  nsi found is not suitable")
        else:
            self._log("  no nsi found")
        self._log("  current fragment not fixed")
        return False 



