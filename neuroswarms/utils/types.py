"""
Common definitions of data types for matrix arrays.
"""

import numpy as np


BOOL_DTYPE = '?'
BINARY_DTYPE = 'u1'
TILE_INDEX_DTYPE = 'u2'
TILE_DTYPE = 'i2'
POINT_DTYPE = 'i2'
KILOGRAM_DTYPE = 'f'
DISTANCE_DTYPE = 'f'
WEIGHT_DTYPE = 'd'
PHASE_DTYPE = 'd'


def to_points(X):
    """
    Convert floating-point (x,y) coordinates to non-negative whole numbers.
    """
    return np.round(X).astype(POINT_DTYPE)
