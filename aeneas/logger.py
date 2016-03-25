#!/usr/bin/env python
# coding=utf-8

"""
A logger class to help with debugging and performance tests.
"""

from __future__ import absolute_import
from __future__ import print_function
import datetime
import io

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

class Logger(object):
    """
    A logger class to help with debugging and performance tests.

    :param bool tee: if ``True``, tee (i.e., log and print to stdout)
    :param int indentation: the initial indentation of the log
    :param bool tee_show_datetime: if ``True``, print date and time when teeing
    :param rconf: a runtime configuration. Default: ``None``, meaning that
                  default settings will be used.
    :type  rconf: :class:`aeneas.runtimeconfiguration.RuntimeConfiguration`
    """

    DEBUG = "DEBU"
    """ ``DEBUG`` severity """

    INFO = "INFO"
    """ ``INFO`` severity """

    WARNING = "WARN"
    """ ``WARNING`` severity """

    CRITICAL = "CRIT"
    """ ``CRITICAL`` severity """

    ERROR = "ERRO"
    """ ``ERRO`` (error) message """

    SUCCESS = "SUCC"
    """ ``SUCC`` (success) message """

    def __init__(self, tee=False, indentation=0, tee_show_datetime=True, rconf=None):
        self.entries = []
        self.tee = tee
        self.indentation = indentation
        self.tee_show_datetime = tee_show_datetime
        self.rconf = rconf if rconf is not None else RuntimeConfiguration()

    def __len__(self):
        return len(self.entries)

    def __unicode__(self):
        return self.pretty_print()

    def __str__(self):
        return gf.safe_str(self.__unicode__())

    def __repr__(self):
        return u"Logger(tee=%s, indentation=%d, tee_show_datetime=%s)" % (
            self.tee,
            self.indentation,
            self.tee_show_datetime
        )

    @property
    def tee(self):
        """
        If ``True``, tee (i.e., log and print to stdout).

        :rtype: bool
        """
        return self.__tee
    @tee.setter
    def tee(self, tee):
        self.__tee = tee

    @property
    def tee_show_datetime(self):
        """
        If ``True``, print date and time when teeing.

        :rtype: bool
        """
        return self.__tee_show_datetime
    @tee_show_datetime.setter
    def tee_show_datetime(self, tee_show_datetime):
        self.__tee_show_datetime = tee_show_datetime

    @property
    def indentation(self):
        """
        The current indentation of the log.
        Useful to visually distinguish log levels.

        :rtype: int
        """
        return self.__indentation
    @indentation.setter
    def indentation(self, indentation):
        self.__indentation = indentation

    def pretty_print(self, as_list=False, show_datetime=True):
        """
        Return a Unicode string pretty print of the log entries.

        :param bool as_list: if ``True``, return a list of Unicode strings,
                             one for each entry, instead of a Unicode string
        :param bool show_datetime: if ``True``, show the date and time of the entries
        :rtype: string or list of strings
        """
        ppl = [entry.pretty_print(show_datetime) for entry in self.entries]
        if as_list:
            return ppl
        return u"\n".join(ppl)

    def log(self, message, severity=INFO, tag=u""):
        """
        Add a given message to the log, and return its time.

        :param string message: the message to be added
        :param severity: the severity of the message
        :type  severity: :class:`aeneas.logger.Logger`
        :param string tag: the tag associated with the message;
                           usually, the name of the class generating the entry
        :rtype: datetime
        """
        entry = _LogEntry(
            severity=severity,
            time=datetime.datetime.now(),
            tag=tag,
            indentation=self.indentation,
            message=self._sanitize(message)
        )
        self.entries.append(entry)
        if self.tee:
            gf.safe_print(entry.pretty_print(show_datetime=self.tee_show_datetime))
        return entry.time

    def clear(self):
        """
        Clear the contents of the log.
        """
        self.entries = []

    def write(self, path):
        """
        Output the log to file.

        :param string path: the path of the log file to be written
        """
        with io.open(path, "w", encoding="utf-8") as log_file:
            log_file.write(self.pretty_print())

    @classmethod
    def _sanitize(cls, message):
        """
        Sanitize the given message,
        dealing with multiple arguments
        and/or string formatting.

        :param message: the log message to be sanitized
        :type  message: string or list of strings
        :rtype: string
        """
        if isinstance(message, list):
            if len(message) == 0:
                sanitized = u"Empty log message"
            elif len(message) == 1:
                sanitized = message[0]
            else:
                sanitized = message[0] % tuple(message[1:])
        else:
            sanitized = message
        if not gf.is_unicode(sanitized):
            raise TypeError("The given log message is not a Unicode string")
        return sanitized



class _LogEntry(object):
    """
    A structure for a log entry.
    """

    def __init__(self, message, severity, tag, indentation, time):
        self.message = message
        self.severity = severity
        self.tag = tag
        self.indentation = indentation
        self.time = time

    def pretty_print(self, show_datetime=True):
        """
        Returns a Unicode string containing
        the pretty printing of a given log entry.

        :param bool show_datetime: if ``True``, print the date and time of the entry
        :rtype: string
        """
        if show_datetime:
            return u"[%s] %s %s%s: %s" % (
                self.severity,
                gf.object_to_unicode(self.time),
                u" " * self.indentation,
                self.tag,
                self.message
            )
        return u"[%s] %s%s: %s" % (
            self.severity,
            u" " * self.indentation,
            self.tag,
            self.message
        )

    @property
    def message(self):
        """
        The message of this log entry.

        :rtype: string
        """
        return self.__message
    @message.setter
    def message(self, message):
        self.__message = message

    @property
    def severity(self):
        """
        The severity of this log entry.

        :rtype: :class:`aeneas.logger.Logger`
        """
        return self.__severity
    @severity.setter
    def severity(self, severity):
        self.__severity = severity

    @property
    def tag(self):
        """
        The tag of this log entry.

        :rtype: string
        """
        return self.__tag
    @tag.setter
    def tag(self, tag):
        self.__tag = tag

    @property
    def indentation(self):
        """
        The indentation of this log entry.

        :rtype: string
        """
        return self.__indentation
    @indentation.setter
    def indentation(self, indentation):
        self.__indentation = indentation

    @property
    def time(self):
        """
        The date and time of this log entry.

        :rtype: datetime.time
        """
        return self.__time
    @time.setter
    def time(self, time):
        self.__time = time



