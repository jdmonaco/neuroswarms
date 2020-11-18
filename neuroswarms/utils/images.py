"""
Functions for image data.
"""

from PIL import Image
from matplotlib.colors import colorConverter
from numpy import empty


def _fill_rgba(shape, color):
    """
    Create a solid fill rectangular image array with the given color.
    """
    rgba = empty(shape + (4,), 'uint8')
    rgba[:] = uint8color(color)
    return rgba

def uint8color(color):
    """
    Convert Matplotlib color spec to uint8 4-tuple.
    """
    return tuple(int(255*v) for v in colorConverter.to_rgba(color))

def rgba_to_image(rgba, filename):
    """
    Save RGBA color matrix to image file.
    """
    img = Image.fromarray(rgba, 'RGBA')
    img.save(filename)