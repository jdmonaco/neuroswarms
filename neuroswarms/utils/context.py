"""
Shared context for filesystem organization of simulation output.

Author :: Joseph Monaco (jmonaco@jhu.edu)
Affiliation: Johns Hopkins University
Created :: 2019-08-18
Updated :: 2020-11-17

Related paper: 

  Monaco, J.D., Hwang, G.M., Schultz, K.M. et al. Cognitive swarming in complex
      environments with attractor dynamics and oscillatory computing. Biol Cybern
      114, 269–284 (2020). https://doi.org/10.1007/s00422-020-00823-z

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.ph
"""

import os
import sys

import numpy as np

from .. import NAME, VERSION, REPOPATH, PROJDIR, RUNDIR, DATADIR

from .console import ConsolePrinter


class NeuroswarmsContext(object):

    """
    A resource object for displaying output and writing files.
    """

    def __init__(self, name=NAME, version=VERSION, repodir=REPOPATH, 
        projdir=PROJDIR, rundir=RUNDIR, datadir=DATADIR):
        """
        Set up the simulation context.
        """
        self._name = name
        self._version = version
        self._repodir = repodir
        self._projdir = projdir
        self._rundir = rundir
        self._datadir = datadir

        self.out = ConsolePrinter(prefix='NeuroSwarms', prefix_color='orange')

    # Console output methods

    def printf(self, *args, **kwargs):
        self.out.printf(*args, **kwargs)

    def box(self, filled=True, color=None):
        self.out.box(filled=filled, color=color)

    def newline(self):
        self.out.newline()

    def hline(self, color='white'):
        self.out.hline(color=color)

    # Run path methods

    def path(self, *path):
        """
        Get an absolute path to a file in the run directory, making sure that
        the parent directory exists.
        """
        if path and os.path.isabs(path[0]):
            return os.path.join(*path)

        pth = os.path.join(self._rundir, *path)
        head, _ = os.path.split(pth)

        if not os.path.isdir(head):
            os.makedirs(head)
        
        return pth

    def mkdir(self, *rpath, base=None):
        """
        Create a subdirectory within the run directory.
        """
        dpath = self.path(*rpath, base=base)
        if os.path.isdir(dpath):
            return dpath
        os.makedirs(dpath)
        return dpath
    
    # Data methods

    def save_array(self, X, name):
        """
        Save a numpy array to the run directory.
        """
        pth = self.path('simdata', name)
        if not pth.endswith('.npy'):
            pth += '.npy'
        try:
            np.save(pth, X)
        except:
            self.out(pth, prefix='ErrorSaving', error=True)
        else:
            self.out(pth, prefix='SavedArray')