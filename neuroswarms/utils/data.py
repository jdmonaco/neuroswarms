"""
Manage an HDF file with methods for flexibly creating data structures.
"""

__all__ = ('DataStore', 'close_all')


import os
import sys
import time
import warnings
import subprocess
import functools

import numpy as np
import tables as tb

from .console import tilde
from .handles import TABLES_FILES
from .base import BaseObject

from .datapath import join, split


warnings.filterwarnings('ignore', category=tb.NaturalNameWarning)


def close_all():
    """Close all HDF files and clear the cache."""
    for k in TABLES_FILES:
        TABLES_FILES[k].close()
    TABLES_FILES.clear()
    tb.file._open_files.close_all()


class OverwriteDisallowed(Exception):
    pass


# Decorators for methods that access and/or modify the HDF file

def datamethod(func):
    @functools.wraps(func)
    def wrapped_data_method(self, *args, **kwargs):
        self.get(readonly=kwargs.pop('readonly', None))
        return func(self, *args, **kwargs)
    return wrapped_data_method

def readmethod(func):
    @functools.wraps(func)
    def wrapped_data_method(self, *args, **kwargs):
        self.get(readonly=True)
        return func(self, *args, **kwargs)
    return wrapped_data_method

def writemethod(func):
    @functools.wraps(func)
    def wrapped_data_method(self, *args, **kwargs):
        self.get(readonly=False)
        result = func(self, *args, **kwargs)
        self.flush()
        return result
    return wrapped_data_method


class DataStore(BaseObject):

    """
    Manage a data file (HDF) and create data structures and nodes.
    """

    def __init__(self, name=None, stem='data', where=None):
        """
        Set up the file paths for the HDF file and backup directory, etc.
        """
        # Allow construction for a positional path argument to existing file
        if name and type(name) is str and os.path.exists(name):
            where, stem = os.path.split(os.path.abspath(name))
            stem, _ = os.path.splitext(stem)
        super().__init__(name=name, color='yellow')
        self.stem = stem
        self.parent = os.getcwd() if where is None else os.path.abspath(where)
        self.backup_path = os.path.join(self.parent, 'backups')
        self.h5path = os.path.join(self.parent, f'{stem}.h5')
        self.h5file = None
        self.pdfile = None

    def _check_cache(self):
        try:
            self.h5file = TABLES_FILES[self.h5path]
        except KeyError:
            pass
        else:
            if not self.h5file.isopen:
                try:
                    del TABLES_FILES[self.h5path]
                except KeyError:
                    pass
                self.h5file = None

    def __str__(self):
        self._check_cache()
        status = ''
        if self.h5file and self.h5file.isopen:
            mode = 'ro'
            if self.h5file.mode in 'aw':
                mode = 'w'
            status = ' (%s)' % mode
        return '%s%s' % (tilde(self.h5path), status)

    def __repr__(self):
        self._check_cache()
        status = 'Unopened'
        if self.h5file:
            if self.h5file.isopen:
                mode = 'Readonly'
                if self.h5file.mode in 'aw':
                    mode = 'Writeable'
                status = 'Open: %s' % mode
            else:
                status = 'Closed'
        return "<DataStoreFile(%s) at '%s'>" % (status, tilde(self.h5path))

    def path(self):
        return self.h5path

    def __enter__(self):
        """
        Open file within a context statement.
        """
        return self.get(readonly=False)

    def __exit__(self, etype, evalue, tb):
        """
        Close file at the end of a context.
        """
        self.close()

    def get(self, readonly=None):
        """
        Get handle to the data file.
        """
        self._check_cache()
        if self.h5file and self.h5file.isopen:
            if self._check_mode(readonly):
                return self.h5file
            self.close()

        if readonly is None:
            readonly = os.path.isfile(self.h5path)

        self._open_file(readonly)
        return self.h5file

    def _check_mode(self, readonly):
        if readonly is None:
            return True
        return ((self.h5file.mode == 'r' and readonly) or
                (self.h5file.mode == 'a' and not readonly))

    def _open_file(self, readonly):
        """
        Open a new file handle to the data file.
        """
        if not os.path.isdir(self.parent):
            os.makedirs(self.parent)

        mode = readonly and 'r' or 'a'
        try:
            self.h5file = tb.open_file(self.h5path, mode=mode)
        except IOError:
            self.out(self.h5path, prefix='DataFileIOError', error=True)
        else:
            TABLES_FILES[self.h5path] = self.h5file

    def flush(self):
        """
        Flush the data file.
        """
        self._check_cache()
        if self.h5file and self.h5file.isopen:
            self.h5file.flush()

    def close(self):
        """
        Close the data file if it's open.
        """
        self._check_cache()
        if not (self.h5file and self.h5file.isopen):
            return

        try:
            self.h5file.close()
        except IOError:
            self.out(self.h5path, prefix='DataFileIOError', error=True)
        else:
            try:
                del TABLES_FILES[self.h5path]
            except KeyError:
                pass

    def exists(self):
        """
        Return whether the data file exists.
        """
        return os.path.exists(self.h5path)

    def backup(self, tag=None):
        """
        Move the data file to a backup folder and create a clean copy in place.
        """
        if not self.exists():
            self.out(self.h5path, prefix='MissingDataFile', error=True)
            return

        label = time.strftime('%Y-%m-%d-%H-%M') if tag is None else tag
        filename = f'{self.stem}-{label}.h5'
        dest = os.path.join(self.backup_path, filename)
        if not os.path.isdir(self.backup_path):
            os.makedirs(self.backup_path)

        self.close()

        if subprocess.call(['mv', self.h5path, dest]) != 0:
            self.out(self.h5path, prefix='BackupMoveFailed', error=True)
            return

        with tb.open_file(dest, mode='r') as df:
            df.copy_file(self.h5path)

        self.out(dest, prefix='DatafileBackedUp')

    @datamethod
    def has_node(self, where, name=None):
        """
        Indicate whether a node exists at the specified path.
        """
        try:
            node = self.h5file.get_node(where, name=name)
        except tb.NoSuchNodeError:
            return False
        return True

    @datamethod
    def get_node(self, where, name=None):
        """
        Return the node object that exists at the specified path.
        """
        try:
            node = self.h5file.get_node(where, name=name)
        except tb.NoSuchNodeError:
            p = where
            if name: p = join(where, name)
            self.out(p, prefix='InvalidPath', error=True)
            return None
        return node

    @writemethod
    def remove_node(self, where, name=None, recursive=True):
        """
        Remove node or subtree at the specified path.
        """
        self.h5file.remove_node(where, name=name, recursive=recursive)

    @writemethod
    def new_array(self, where, name, x, overwrite=True, **kwargs):
        """
        Add `overwrite` keyword to tables.File.create_array().

        Note: Masked arrays are handled by creating a Group at the specifed
        path and saving the data and mask buffers as Array objects within that
        Group. An attribute is set on the Group node indicating to read_array()
        that the child Arrays should be read and returned as a masked array.

        Returns `tables.Array` node.
        """
        try:
            array = self.h5file.get_node(where, name=name)
        except tb.NoSuchNodeError:
            pass
        else:
            if not overwrite:
                raise OverwriteDisallowed('%s array already exists' %
                        array._v_pathname)
            self.h5file.remove_node(array, recursive=True)
        kwargs.update(createparents=True)

        if isinstance(x, np.ma.MaskedArray):
            grp = self.new_group(where, name)
            grp._v_attrs['masked_array_group'] = True
            grp._v_attrs['title'] = kwargs.pop('title', 'MaskedArray')
            self.h5file.create_array(grp, 'data', obj=x.data, title='Data',
                    **kwargs)
            self.h5file.create_array(grp, 'mask', obj=x.mask, title='Mask',
                    **kwargs)
            return grp
        return self.h5file.create_array(where, name, obj=x, **kwargs)

    @readmethod
    def read_array(self, where, name=None):
        """
        Read array data from the given node.
        """
        try:
            node = self.h5file.get_node(where, name=name)
        except tb.NoSuchNodeError:
            p = where
            if name is not None: p = join(p, name)
            self.out(p, prefix='MissingDataNode', error=True)
            return

        arr = None
        if isinstance(node, tb.Group):
            if 'masked_array_group' in node._v_attrs:
                arr = np.ma.MaskedArray(data=node.data.read(),
                                        mask=node.mask.read())
        elif isinstance(node, tb.Array):
            arr = node.read()
        if arr is None:
            raise TypeError('not an array: {}'.format(node_v_pathname))
        return arr

    @writemethod
    def new_group(self, where, name, **kwargs):
        """
        Enforce `createparents=True` to create a new group.
        """
        kwargs.update(createparents=True)
        try:
            grp = self.h5file.create_group(where, name, **kwargs)
        except tb.NodeError:
            grp = self.h5file.get_node(where, name=name)
            self.out(grp._v_pathname, prefix='GroupExists', warning=True)
        return grp