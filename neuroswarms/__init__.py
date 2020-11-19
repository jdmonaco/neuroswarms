"""
Source code package for the NeuroSwarms controller model.

Author: Joseph Monaco (jmonaco@jhu.edu)
Affiliation: Johns Hopkins University
Created: 2019-08-18
Updated: 2020-11-17

Requirements: numpy, scipy, matplotlib, pytables, pillow.

Related paper:

  Monaco, J.D., Hwang, G.M., Schultz, K.M. et al. Cognitive swarming in complex
      environments with attractor dynamics and oscillatory computing. Biol Cybern
      114, 269–284 (2020). https://doi.org/10.1007/s00422-020-00823-z

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.ph
"""

NAME = 'neuroswarms'
VERSION = '1.0.0'

import os
import sys


# Configure movie parameters

MOVIE_FPS = 100.0  # 100.0 = frame-rate match for simulation timescale
MOVIE_DPI = 227    # 151 = default scaled resolution of Macbook Pro retina screen


# Set up project and data paths
#
# Note: Edit the definition of PROJDIR to set the path for simulation output.
# It defaults to ~/neuroswarms as defined below.

if sys.platform == 'win32':
    HOME = os.getenv('USERPROFILE')
else:
    HOME = os.getenv('HOME')

REPOPATH = os.path.split(__file__)[0]
MAPDIR = os.path.join(os.path.dirname(REPOPATH), 'mapdata')
PROJDIR = os.path.join(HOME, NAME)
RUNDIR = os.path.join(PROJDIR, 'output')
DATADIR = os.path.join(PROJDIR, 'data')
