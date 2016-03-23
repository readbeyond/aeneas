#!/usr/bin/env python
# coding=utf-8

"""
This module contains the following classes:

* :class:`~aeneas.configuration.Configuration`
  which is a dictionary with a fixed set of keys,
  possibly with default values and key aliases.

.. versionadded:: 1.4.1
"""

from __future__ import absolute_import
from __future__ import print_function

import aeneas.globalconstants as gc
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

class Configuration(object):
    """
    A generic configuration object, that is,
    a dictionary with a fixed set of keys,
    each with a type, a default value, and possibly aliases.

    Keys are (unique) Unicode strings.

    Values are stored as Unicode strings (or ``None``), and casted
    to the type of the field (``int``, ``float``,
    ``bool``, :class:`~aeneas.timevalue.TimeValue`, etc.)
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
        # (field_name, (default_value, conversion_function, [alias1, alias2, ...]))
        #
        # examples:
        # (gc.FOO, (None, None, ["foo"]))
        # (gc.BAR, (0.0, float, ["bar", "barrr"]))
        # (gc.BAZ, (None, TimeValue, ["baz"]))
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
        for (field, info) in self.FIELDS:
            (fdefault, ftype, faliases) = info
            self.data[field] = fdefault
            self.types[field] = ftype
            for alias in faliases:
                self.aliases[alias] = field

        if config_string is not None:
            # strip leading/trailing " or ' characters
            if (len(config_string) > 0) and (config_string[0] == config_string[-1]) and (config_string[0] in [u"\"", u"'"]):
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

    def config_string(self):
        """
        Build the storable string corresponding
        to this configuration object.

        :rtype: string
        """
        return (gc.CONFIG_STRING_SEPARATOR_SYMBOL).join(
            [u"%s%s%s" % (fn, gc.CONFIG_STRING_ASSIGNMENT_SYMBOL, self.data[fn]) for fn in sorted(self.data.keys()) if self.data[fn] is not None]
        )



