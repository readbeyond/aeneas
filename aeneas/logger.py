#!/usr/bin/env python
# coding=utf-8

"""
A logger class to help with debugging and performance tests.
"""

import datetime

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

class Logger(object):
    """
    A logger class to help with debugging and performance tests.

    :param tee: if ``True``, tee (i.e., log and print to stdout)
    :type  tee: bool
    :param indentation: the initial indentation of the log
    :type  indentation: int
    """

    DEBUG = "DEBU"
    """ ``DEBUG`` severity """

    INFO = "INFO"
    """ ``INFO`` severity """

    WARNING = "WARN"
    """ ``WARNING`` severity """

    CRITICAL = "CRIT"
    """ ``CRITICAL`` severity """

    def __init__(self, tee=False, indentation=0):
        self.entries = []
        self.tee = tee
        self.indentation = indentation

    def __len__(self):
        return len(self.entries)

    #@property
    #def entries(self):
    #    """
    #    The entries currently in the log.
    #
    #    :rtype: list of :class:`aeneas.logger._LogEntry`
    #    """
    #    return self.__entries
    #@entries.setter
    #def entries(self, entries):
    #    self.__entries = entries

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

    def log(self, message, severity=INFO, tag=""):
        """
        Add a given message to the log.

        :param message: the message to be added
        :type  message: string or list
        :param severity: the severity of the message
        :type  severity: string (from the :class:`aeneas.logger.Logger` enum)
        :param tag: the tag associated with the message;
                    usually, the name of the class generating the entry
        :type  tag: string
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
            print self._pretty_print(entry)

    def _sanitize(self, message):
        """
        Sanitize the given message,
        dealing with unicode and/or multiple arguments,
        and string formatting

        :param message: the log message to be sanitized
        :type  message: string, unicode or list

        :rtype: string
        """
        sanitized = message
        if isinstance(sanitized, list):
            if len(sanitized) == 0:
                sanitized = "Empty log message"
            elif len(sanitized) == 1:
                sanitized = sanitized[0]
            else:
                model = self._safe_unicode_to_str(sanitized[0])
                args = tuple()
                for arg in sanitized[1:]:
                    if isinstance(arg, unicode) or isinstance(arg, str):
                        args += (self._safe_unicode_to_str(arg),)
                    else:
                        args += (arg,)
                sanitized = model % args
        return self._safe_unicode_to_str(sanitized)

    def _safe_unicode_to_str(self, value):
        """
        Safely convert a string or unicode value
        to string.

        :param value: the value to be converted
        :type  value: string or unicode

        :rtype: string
        """
        sanitized = value
        if isinstance(sanitized, unicode):
            try:
                sanitized = sanitized.encode("utf-8")
            except UnicodeError:
                sanitized = sanitized.encode("ascii", "replace")
        return sanitized

    def clear(self):
        """
        Clear the contents of the log.
        """
        self.entries = []

    def _pretty_print(self, entry):
        """
        Returns a string containing the pretty printing
        of a given log entry.

        :param entry: the log entry
        :type  entry: :class:`aeneas.logger._LogEntry`
        :rtype: string
        """
        return "[%s] %s %s%s: %s" % (
            entry.severity,
            str(entry.time),
            " " * entry.indentation,
            entry.tag,
            entry.message
        )

    def to_list_of_strings(self):
        """
        Return the log entries as a list of pretty printed strings.

        :rtype: list of strings
        """
        return [self._pretty_print(entry) for entry in self.entries]


    def __str__(self):
        return "\n".join(self.to_list_of_strings())

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

        :rtype: string (from the :class:`aeneas.logger.Logger` enum)
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
        The time of this log entry.

        :rtype: datetime.time
        """
        return self.__time
    @time.setter
    def time(self, time):
        self.__time = time



