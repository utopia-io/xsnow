"""Helper classes and exceptions
"""
# To enable postponed evaluation of annotations (default for 3.10)
from __future__ import annotations

from enum import Enum
from typing import Union, List

import datetime as dt

__author__ = 'arwi'

Ordered = Union[dt.date, int]


class FloatEnum(float, Enum):
    """Enum where members are also (and must be) floats. (Copied from Enum source.)"""

    def __str__(self):
        return "%s" % (self._name_,)

    def __format__(self, format_spec):
        """Returns format using actual value unless __str__ has been overridden."""
        str_overridden = type(self).__str__ != FloatEnum.__str__
        if str_overridden:
            cls = str
            val = str(self)
        else:
            cls = self._member_type_
            val = self._value_
        return cls.__format__(val, format_spec)


class RegobsError(Exception):
    pass


class NoObservationError(RegobsError):
    pass