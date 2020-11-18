"""
Functions to load SVG data for use in the models.
"""

from operator import itemgetter

import numpy as np
import matplotlib as mpl

from .. import MOVIE_DPI


def load_environment(svgfile, dpi=MOVIE_DPI):
    """
    Parse SVG map file for environmental barriers and special locations.

    The map file should contain only line segments and text objects that begin
    with single letters representing cues ('C'), rewards ('R'), and spawn
    locations ('S'). Text object may optionally follow the initial letter with
    numerical digits representing an integer (up to unsigned 16-bit). This
    value is stored as an 'extended' attribute that may be referred to in the
    model. For example, 'S50' may indicate a spawn location with a safe radius
    of 50pts.

    The SVG code should be formatted according to the Tiny 1.x specification.

    :returns: dict with keys:
        :origin: (2,)-tuple of the (top,left) coordinates
        :width: scalar width of environment
        :height: scalar height of environment
        :extent: (4,)-tuple of (left,right,bottom,top) coordinates
        :figsize: (2,)-tuple of equivalent figure image size
        :barriers: (N_B, 4)-matrix of line segments
        :cues: (N_C, 2)-matrix of cue positions
        :rewards: (N_R, 2)-matrix of reward positions
        :spawns: (N_spawns, 2)-matrix of spawn locations
    """
    nodes = _parse_svg(svgfile)
    env = {}

    # Set environmental dimensions
    svg = list(filter(lambda x: x['type'] == 'svg', nodes))[0]
    x = round(svg['x'])
    y = round(svg['y'])
    w = int(np.ceil(svg['width']))
    h = int(np.ceil(svg['height']))
    env['origin'] = x, y
    env['width'] = w
    env['height'] = h
    env['extent'] = x, x + w, y, y + h
    env['figsize'] = w / dpi, h / dpi

    # Set barrier line-segment data
    line_nodes = list(filter(lambda x: x['type'] == 'line', nodes))
    for line in line_nodes:
        line['xmin'] = min(line['x1'], line['x2'])
        line['ymin'] = min(line['y1'], line['y2'])
    lines = sorted(line_nodes, key=itemgetter('ymin', 'xmin'))
    env['barriers'] = np.array([[round(line[k]) for k in ('x1','y1','x2','y2')]
            for line in lines], 'i')

    # Set position data for cues, rewards, and spawn locations
    for key, letter in [('cues', 'c'), ('rewards', 'r'), ('spawns', 's')]:
        points = sorted(list(filter(lambda x: x['type'] == 'text' and \
                str(x['value'][0]).lower() == letter, nodes)),
                key=itemgetter('y', 'x'))
        for pt in points:
            if len(pt['value']) > 1:
                pt['xattr'] = int(pt['value'][1:])
            else:
                pt['xattr'] = 0
        env[key] = np.array([[round(pt[k]) for k in ('x','y','xattr')]
                for pt in points], 'i')

    return env

def _parse_svg(filename):
    with open(filename, 'r') as fd:
        svg = fd.read()

    inopenpair = False
    descrs = []
    i = 0
    while i < len(svg):
        if inopenpair:
            if svg[i] == '<':
                s += '"'
                i = svg.find('>', i)
                descrs.append(s.split())
                inopenpair = False
            else:
                if svg[i] == ' ':
                    s += '_'
                else:
                    s += svg[i]
            i += 1
            continue
        if svg[i] != '<':
            i += 1
            continue
        tag_end = svg.find(' ', i)
        tag = svg[i+1:tag_end]
        if tag[0] in '?!/':
            i += 1
            continue
        s = 'type="%s"' % tag
        i = tag_end
        inquote = False
        found_transform = False
        x = y = 0.0
        while svg[i] not in '/>' or inquote:
            if svg[i] == '"':
                inquote = not inquote
                if inquote and svg.find('matrix(', i+1) == i+1:
                    endquote = svg.find('"', i+1)
                    transform = svg[i+8:endquote-1]
                    x, y = transform.split()[-2:]
                    found_transform = True

            if inquote and svg[i] == ' ':
                s += '_'
            elif not inquote and svg[i] in ':-':
                s += '_'
            else:
                s += svg[i]
            i += 1

        if found_transform:
            s += ' x="%s" y="%s"' % (x, y)

        if svg[i] == '/' or tag == 'svg':
            descrs.append(s.split())
        elif svg[i] == '>':
            inopenpair = True
            s += ' value="'

        i += 1

    nodes = []
    for attrs in descrs:
        node = eval('dict(%s)' % (','.join(attrs)))
        for key in node:
            if node[key].endswith('px'):
                node[key] = node[key][:-2]
            try:
                num = float(node[key])
            except ValueError:
                pass
            else:
                node[key] = num
        nodes.append(node)
    return nodes
