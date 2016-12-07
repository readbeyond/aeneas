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

* :class:`~aeneas.configuration.Configuration`
  which is a dictionary with a fixed set of keys,
  possibly with default values and key aliases.

.. versionadded:: 1.4.1
"""

from __future__ import absolute_import
from __future__ import print_function
from copy import deepcopy

from aeneas.exacttiming import Decimal
from aeneas.exacttiming import TimeValue
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf


class Configuration(object):
    """
    A generic configuration object, that is,
    a dictionary with a fixed set of keys,
    each with a type, a default value, and possibly aliases.

    Keys are (unique) Unicode strings.

    Values are stored as Unicode strings (or ``None``), and casted
    to the type of the field (``int``, ``float``,
    ``bool``, :class:`~aeneas.exacttiming.TimeValue`, etc.)
    when accessed.

    For ``bool`` keys, values listed in
    :data:`~aeneas.configuration.Configuration.TRUE_ALIASES`
    are considered equivalent to a ``True`` value.

    If ``config_string`` is not ``None``, the given string will be parsed
    and ``key=value`` pairs will be stored in the object,
    provided that ``key`` is listed in :data:`~aeneas.configuration.Configuration.FIELDS`.

    :param string config_string: the configuration string to be parsed
    :raises: TypeError: if ``config_string`` is not ``None`` and it is not a Unicode string
    :raises: KeyError: if trying to access a key not listed above
    """

    FIELDS = [
        #
        # in subclasses, create fields like this:
        # (field_name, (default_value, conversion_function, [alias1, alias2, ...], human_description))
        #
        # examples:
        # (gc.FOO, (None, None, ["foo"], u"path to foo"))
        # (gc.BAR, (0.0, float, ["bar", "barrr"], u"bar threshold"))
        # (gc.BAZ, (None, TimeValue, ["baz"], u"duration, in seconds, of baz"))
        #
    ]
    """
    The fields, that is, key names each with associated
    default value, type, and possibly aliases,
    of this object.
    """

    TRUE_ALIASES = [True, u"TRUE", u"True", u"true", u"YES", u"Yes", u"yes", u"1", 1]
    """
    Aliases for a ``True`` value for ``bool`` fields
    """

    TAG = u"Configuration"

    def __init__(self, config_string=None):
        if (config_string is not None) and (not gf.is_unicode(config_string)):
            raise TypeError(u"config_string is not a Unicode string")

        # set dictionaries up to keep the config data
        self.data = {}
        self.types = {}
        self.aliases = {}
        self.desc = {}
        for (field, info) in self.FIELDS:
            (fdefault, ftype, faliases, fdesc) = info
            self.data[field] = fdefault
            self.types[field] = ftype
            self.desc[field] = fdesc
            for alias in faliases:
                self.aliases[alias] = field

        if config_string is not None:
            # strip leading/trailing " or ' characters
            if (
                (len(config_string) > 0) and
                (config_string[0] == config_string[-1]) and
                (config_string[0] in [u"\"", u"'"])
            ):
                config_string = config_string[1:-1]
            # populate values from config_string,
            # ignoring keys not present in FIELDS
            properties = gf.config_string_to_dict(config_string)
            for key in set(properties.keys()) & set(self.data.keys()):
                self.data[key] = properties[key]

    def __contains__(self, key):
        return (key in self.data) or (key in self.aliases)

    def __setitem__(self, key, value):
        if key in self.aliases:
            key = self.aliases[key]
        if key in self.data:
            self.data[key] = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key in self.aliases:
            key = self.aliases[key]
        if key in self.data:
            return self._cast(key, self.data[key])
        else:
            raise KeyError(key)

    def __unicode__(self):
        return u"\n".join([u"%s: '%s'" % (fn, self.data[fn]) for fn in sorted(self.data.keys())])

    def __str__(self):
        return gf.safe_str(self.__unicode__())

    def _cast(self, key, value):
        if (value is not None) and (self.types[key] is not None):
            if self.types[key] is bool:
                return value in self.TRUE_ALIASES
            else:
                return self.types[key](value)
        return value

    def clone(self):
        """
        Return a deep copy of this configuration object.

        .. versionadded:: 1.7.0

        :rtype: :class:`~aeneas.configuration.Configuration`
        """
        return deepcopy(self)

    @property
    def config_string(self):
        """
        Build the storable string corresponding
        to this configuration object.

        :rtype: string
        """
        return (gc.CONFIG_STRING_SEPARATOR_SYMBOL).join(
            [u"%s%s%s" % (fn, gc.CONFIG_STRING_ASSIGNMENT_SYMBOL, self.data[fn]) for fn in sorted(self.data.keys()) if self.data[fn] is not None]
        )

    @classmethod
    def parameters(cls, sort=True, as_strings=False):
        """
        Return a list of tuples ``(field, description, type, default)``,
        one for each field of the configuration.

        :param bool sort: if ``True``, return the list sorted by field
        :param bool as_strings: if ``True``, return formatted strings instead
        :rtype: list
        """
        def cft(ftype, fdefault):
            """ Convert field type and default value to string """
            if ftype is None:
                return u""
            if ftype in [TimeValue, Decimal, float]:
                cftype = u"float"
                cfdefault = u"%.3f" % ftype(fdefault) if fdefault is not None else u"None"
            elif ftype == int:
                cftype = u"int"
                cfdefault = u"%d" % ftype(fdefault) if fdefault is not None else u"None"
            elif ftype == bool:
                cftype = u"bool"
                cfdefault = u"%s" % fdefault if fdefault is not None else u"None"
            else:
                cftype = u"unknown"
                cfdefault = u"%s" % fdefault if fdefault is not None else u"None"
            return u" (%s, %s)" % (cftype, cfdefault)

        parameters = [(field, fdesc, ftype, fdefault) for (field, (fdefault, ftype, faliases, fdesc)) in cls.FIELDS]
        if sort:
            parameters = sorted(parameters)
        if as_strings:
            l = max([len(t[0]) for t in parameters])
            parameters = [u"%s : %s%s" % (f.ljust(l), d, cft(t, df)) for (f, d, t, df) in parameters]
        return parameters
