"""
Base class for context components.
"""

from numpy import random

from .console import ConsolePrinter


class BaseObject(object):

    """
    A base class to provide naming and console functionality.
    """

    __counts = {}

    def __init__(self, name=None, color=None, textcolor=None, **kwargs):
        """
        Configure object with name and console output functions.

        Arguments
        ---------
        name : str
            A unique string name for this object. By default, a name will be
            created from the object's class's name and a count

        color : string, valid color name from pouty.console
            Prefix text color for console output from this object

        textcolor : string, valid color name from pouty.conole
            Message text color for console output from this object

        Any remaining keyword arguments will be consumed but warnings will also
        be issued, since they should typically be consumed earlier in the MRO.
        """
        self._initialized = False

        # Set color/textcolor defaults for BaseObject instances
        color = 'green' if color is None else color
        textcolor = 'default' if textcolor is None else textcolor

        # Set the class name to an instance attribute
        if hasattr(self.__class__, 'name'):
            self.klass = self.__class__.name
        else:
            self.klass = self.__class__.__name__

        # Set the instance name to a class-based count if unspecified
        if name is None:
            if self.klass in self.__counts:
                self.__counts[self.klass] += 1
            else:
                self.__counts[self.klass] = 0
            c = self.__counts[self.klass]
            if c == 0:
                self.name = self.klass
            else:
                self.name = f'{self.klass}_{c:03d}'
        else:
            self.name = name

        # Add a ConsolePrinter instance attribute for printing
        if not hasattr(self, 'out'):
            self.out = ConsolePrinter(prefix=self.name, prefix_color=color,
                    message_color=textcolor)

        # Add a debug function
        if not hasattr(self, 'debug'):
            self.debug = self.out.debug

        # Warn about unconsumed kwargs
        for key, value in kwargs.items():
            self.out(f'{key} = {value!r}', prefix='UnconsumedKwargs',
                     warning=True)

        self._initialized = True

    def __repr__(self):
        if hasattr(self, '__qualname__'):
            return self.__qualname__
        if hasattr(self, '__module__') and hasattr(self, '__name__'):
            return f'{self.__module__}{self.__class__.__name__}'
        return object.__repr__(self)

    def __format__(self, fmtspec):
        return self.name.__format__(fmtspec)
