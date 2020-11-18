"""
Functions for importing and processing environmental geometry.
"""

import os
import json
import time
import queue

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from matplotlib.patches import Circle

from .. import MAPDIR, MOVIE_DPI

from .images import uint8color, rgba_to_image, _fill_rgba
from .data import DataStore
from .console import ConsolePrinter

from .svg import load_environment
from .types import *


ALPHA = 10.0
K_H = 20.0
COLORMAP = 'gray_r'
MASK_COLOR = 'cyan'
CUE_COLOR = 'purple'
REWARD_COLOR = 'gold'


def map_index(X):
    """
    Return a tuple index for map matrixes based on a set of position points.
    """
    return tuple(to_points(X).T)


class EnvironmentGeometry(object):

    """
    Import, processing, and data functions on environmental geometry.
    """

    def __init__(self, name, mapdir=None, recompute=False, alpha=ALPHA,
        k_H=K_H):
        """
        Find the named map specification file (.svg) and begin processing.

        :name: Name of environment
        :mapdir: Path to directory containing map data folders
        :recompute: Recompute all geometry regardless of existing data
        :alpha: Barrier repulsion spatial constant (in points)
        """
        self.out = ConsolePrinter(prefix=f'EnvGeom(\'{name}\')',
                prefix_color='green')

        self.name = name
        self.mapdir = MAPDIR if mapdir is None else mapdir
        self.envdir = os.path.join(self.mapdir, name)
        self.svgpath = os.path.join(self.mapdir, f'{name}.svg')
        self.h5path = os.path.join(self.envdir, 'geometry.h5')
        self.datafile = DataStore(self.h5path)
        self.infopath = os.path.join(self.envdir, 'info.json')
        self.backupdir = os.path.join(self.envdir, 'backups')
        self.recompute = recompute

        assert os.path.isdir(self.mapdir), f'not a directory: {mapdir}'

        if os.path.isfile(self.svgpath):
            self.out(self.svgpath, prefix='MapFile')
            self.alpha = alpha
            self.k_H = k_H
            self.process()
        else:
            self.out(f"Missing geometry data or map file for '{name}':" \
                     f"Please save map file to {self.svgpath}.", error=True)

    def __str__(self):
        return f'<{self.__class__.__name__}(\'{self.name}\'): ' \
               f'{self.shape[0]}x{self.shape[1]}, ' \
               f'{self.N_B} barriers, {self.N_C} cues, {self.N_R} rewards, ' \
               f'{len(self.H)} visibility tiles>'

    def __repr__(self):
        return f'{self.__class__.__name__}(\'{self.name}\', ' \
               f'alpha={self.alpha}, k_H={self.k_H})'

    def process(self):
        """
        Load the SVG map file for parsing and processing the environment.
        """
        try:
            env = load_environment(self.svgpath)
        except Exception:
            self.out(self.svgpath, prefix='LoadError', error=True)
            return

        info = self.info = {k:env[k] for k in ('origin','width','height',
            'extent','figsize')}

        self.origin  = info['origin']
        self.width   = info['width']
        self.height  = info['height']
        self.extent  = info['extent']
        self.figsize = info['figsize']

        self.B   = env['barriers']
        self.C   = env['cues'][:,:2]
        self.C_W = env['cues'][:,2]
        self.R   = env['rewards'][:,:2]
        self.R_W = env['rewards'][:,2]
        self.S0  = env['spawns']

        info['N_B']   = self.N_B   = len(self.B)
        info['N_C']   = self.N_C   = len(self.C)
        info['N_R']   = self.N_R   = len(self.R)
        info['N_0']   = self.N_0   = len(self.S0)
        info['shape'] = self.shape = (self.width, self.height)
        info['name']  = self.name
        info['alpha'] = self.alpha
        info['k_H']   = self.k_H

        if not os.path.isdir(self.envdir):
            os.makedirs(self.envdir)

        self._compute_geometry()

        try:
            with open(self.infopath, 'w') as fd:
                json.dump(info, fd, indent=2, separators=(', ', ': '),
                        sort_keys=True)
        except:
            self.out(self.infopath, prefix='SaveError', error=True)
        else:
            self.out(self.infopath, prefix='InfoFile')

    def sample_spawn_points(self, N=1):
        """
        Randomly sample spawn locations from all possible points.

        :N: The number of random samples to draw
        :returns: (N, 2)-matrix of random spawn locations
        """
        N_X0 = len(self.X0)
        if N > N_X0:
            rnd = lambda n: np.random.randint(N_X0, size=n)
        else:
            rnd = lambda n: np.random.permutation(np.arange(N_X0))[:n]

        ix = rnd(N)
        dmin = self.G_PD[map_index(self.X0[ix])]
        while np.any(dmin < self.alpha):
            fix = dmin < self.alpha
            ix[fix] = rnd(fix.sum())
            dmin = self.G_PD[map_index(self.X0[ix])]

        return self.X0[ix]

    def maps(self):
        """
        Return a attribute-key dict of map-like matrix arrays.
        """
        maps = {}
        for k in self.__dict__.keys():
            X = getattr(self, k)
            if isinstance(X, np.ndarray) and X.shape[:2] == self.shape:
                maps[k] = X
        return maps

    def save_all_maps(self, **imagefmt):
        """
        Save images of all environmental map matrixes.
        """
        for name in self.maps().keys():
            self.save_map(name, **imagefmt)

    def save_map(self, name, **imagefmt):
        """
        Save images of all environmental map matrixes.
        """
        M = getattr(self, name)
        if M.ndim == 3:
            for j in range(M.shape[2]):
                self._save_matrix_image(M[...,j], f'{name}_{j:02d}', **imagefmt)
        elif M.ndim == 2:
            self._save_matrix_image(M, name, **imagefmt)

    def plot_all_map_figures(self, **imagefmt):
        """
        Plot all environment maps in new figure windows.
        """
        for name in self.maps().keys():
            self.plot_map(name, **imagefmt)

    def plot_map_figure(self, name, **imagefmt):
        """
        Plot full-bleed figure window(s) of the named map.
        """
        assert name in self.maps().keys(), f'not a map name {name}'
        M = getattr(self, name)
        if M.ndim == 3:
            for j in range(M.shape[2]):
                self.figure(mapname=(name, j), **imagefmt)
        elif M.ndim == 2:
            f, ax = self.figure(mapname=name, **imagefmt)
            return f, ax

    def plot_tile_map(self, cue_color=CUE_COLOR, reward_color=REWARD_COLOR,
        **imagefmt):
        """
        Verify tile map organization by plotting with index numbers.
        """
        cmap = imagefmt.pop('cmap', 'cubehelix')
        f, ax = self.figure(mapname='G_PH', cmap=cmap, **imagefmt)

        # Plot index labels at the center of each grid tile
        dpi = mpl.rcParams['figure.dpi']
        font = dict(fontsize=3.2*(245/dpi), weight='light')
        for i, (x,y) in enumerate(self.H):
            ax.text(x + 0.5, y + 0.5, str(i), fontdict=font, ha='center',
                    va='center', color='hotpink', zorder=0)

        # Draw circles around tiles for each cue
        fmt = dict(fill=False, facecolor=None, alpha=0.9, zorder=10)
        [ax.add_artist(Circle(self.H[self.C_H[c]], radius=self.k_H/2,
            edgecolor=cue_color, linewidth=0.5+0.5*self.C_W[c], **fmt))
                for c in range(self.N_C)]

        # Draw circles around tiles for each reward
        [ax.add_artist(Circle(self.H[self.R_H[r]], radius=self.k_H/2,
            edgecolor=reward_color, linewidth=0.5+0.5*self.R_W[r], **fmt))
                for r in range(self.N_R)]

        plt.draw()

    def plot_visibility(self, which='cue', **imagefmt):
        """
        Plot visibility of cues (which='cue') or rewards (which='reward').
        """
        if which == 'cue':
            P = self.C
            N_P = self.N_C
            C_HP = self.V_HC
        elif which == 'reward':
            P = self.R
            N_P = self.N_R
            C_HP = self.V_HR
        else:
            self.out('Must be cue or reward: {}', which, error=True)
            return

        plt.ioff()
        f, ax = self.figure(clear=True, tag=f'{which}vis', mapname='G_P')

        alpha = 0.5
        ms0 = 2
        lw = 0.5
        cfmt = dict(marker='o', ms=3*ms0, mec='k', mew=lw, alpha=(2+alpha)/3,
                zorder=10)
        vfmt = dict(ls='-', lw=lw, marker='.', ms=ms0, mec='k', mfc='k',
                mew=lw, alpha=alpha, zorder=5)
        cols = [mpl.cm.tab10.colors[c%10] for c in range(N_P)]

        for c, (cx, cy) in enumerate(P):
            Vx, Vy = tuple(map(lambda v: v[np.newaxis,:],
                self.H[C_HP[:,c].nonzero()].T))
            Cx = np.zeros((1,Vx.size), dtype=POINT_DTYPE) + cx
            Cy = np.zeros((1,Vy.size), dtype=POINT_DTYPE) + cy
            X = np.vstack((Cx, Vx))
            Y = np.vstack((Cy, Vy))
            ax.plot([cx], [cy], mfc=cols[c], **cfmt)
            ax.plot(X, Y, c=cols[c], **vfmt)

        plt.ion()
        plt.show()
        plt.draw()

        savepath = os.path.join(self.envdir, f'G_P-{which}-visibility.png')
        plt.savefig(savepath, dpi=mpl.rcParams['savefig.dpi'])
        self.out(f'Saved: {savepath}')

        return f, ax

    def figure(self, clear=True, tag=None, mapname=None, **imagefmt):
        """
        Get a figure window and full-bleed axes for plotting maps.
        """
        wasinteractive = plt.isinteractive()
        if wasinteractive:
            plt.ioff()

        # Name the figure and retrieve background map if specified
        figname = self.name
        if tag is not None:
            figname += f'+{tag}'
        do_mapshow = False
        ix = None
        if mapname is not None:
            if type(mapname) is tuple and len(mapname) == 2:
                mapname, ix = mapname
            if mapname in self.maps():
                figname += f'.{mapname}'
                Mmap = getattr(self, mapname)
                if Mmap.ndim == 3:
                    Mmap = Mmap[...,ix]
                    figname += f'[{ix}]'
                do_mapshow = True
            else:
                self.out(mapname, prefix='InvalidMapName', error=True)

        # Get the figure, clear it, and set the correct size
        f = plt.figure(num=figname, figsize=self.figsize, dpi=MOVIE_DPI)
        if clear:
            f.clear()
        f.set_size_inches(self.figsize, forward=True)

        # Plot the map to full-bleed axes
        ax = plt.axes([0,0,1,1])
        if do_mapshow:
            self.plot(Mmap, ax=ax, clear=clear, **imagefmt)

        if wasinteractive:
            plt.ion()
            plt.show()
        plt.draw()

        return f, ax

    def plot(self, envmap, index=None, ax=None, clear=True, **imagefmt):
        """
        Plot an environment map to an axes object.
        """
        if ax is None:
            ax = plt.gca()
        if clear:
            ax.clear()

        if type(envmap) is str:
            M = getattr(self, envmap)
        elif isinstance(envmap, np.ndarray):
            M = envmap
        if M.ndim == 3:
            if index is None:
                self.out('Dim >2 arrays require index argument', error=True)
                return
            M = M[...,index]

        assert M.shape == self.shape, f'matrix is not a map {Mmap.shape}'

        imagefmt.update(asmap=True, forimshow=True)
        im = ax.imshow(
                self._rgba_matrix_image(M, **imagefmt),
                origin='lower', interpolation='nearest',
                extent=self.extent, zorder=-100)

        ax.axis(self.extent)
        ax.set_axis_off()
        ax.axis('equal')

        return im

    def _save_matrix_image(self, M, name, **imagefmt):
        """
        Save a matrix image to a pre-determined path based on the name.
        """
        if not (M.shape == self.shape or
                (M.ndim == 2 and M.shape[0] == M.shape[1])):
            return

        rgba = self._rgba_matrix_image(M, **imagefmt)
        savepath = os.path.join(self.envdir, f'{name}-matrix.png')
        self._move_to_backup(savepath)
        rgba_to_image(rgba, savepath)
        self.out(f'Saved: {savepath}')

    def _rgba_matrix_image(self, M, asmap=True, forimshow=False,
        mask_color=MASK_COLOR, cmap=COLORMAP, cmin=None, cmax=None):
        """
        Convert a matrix to an RGBA color array for image output.
        """
        if asmap:
            if forimshow:
                M = M.T  # must use origin='lower'
            else:
                M = np.flipud(M.T)

        mask = None
        if np.ma.isMA(M):
            mask = M.mask
            if np.all(M.mask):
                M = np.zeros_like(M.data)
            else:
                vmin = M.min()
                M = M.data.copy()
                M[mask] = vmin

        if M.dtype is np.dtype(bool):
            M = M.astype('f')

        if cmin is None:
            cmin = M.min()
        if cmax is None:
            cmax = M.max()
        np.clip(M, cmin, cmax, out=M)

        cm = plt.get_cmap(cmap)
        if cmin == cmax:
            rgba = _fill_rgba(M.shape, cm(0.0))
        else:
            rgba = cm((M - cmin) / (cmax - cmin), bytes=True)

        if mask is not None:
            rgba[mask] = uint8color(mask_color)

        return rgba

    def _move_to_backup(self, f):
        """
        Move an existing file to the backup directory.
        """
        if not os.path.isfile(f):
            return
        if not os.path.isdir(self.backupdir):
            os.makedirs(self.backupdir)
        head, ext = os.path.splitext(f)
        os.rename(f, os.path.join(self.backupdir, os.path.basename(head) + \
                time.strftime('+%Y-%m-%d-%H%M-%S') + ext))

    def _compute_geometry(self):
        """
        Pipeline script for computing the environmental geometry.
        """
        # Flip all y-values to allow a lower-left origin
        self.B[:,[1,3]] = self.height - self.B[:,[1,3]]
        self.C[:,1]     = self.height - self.C[:,1]
        self.R[:,1]     = self.height - self.R[:,1]
        self.S0[:,1]    = self.height - self.S0[:,1]

        self._rasterize_barriers()
        self._create_environment_mask()
        self._find_closest_barriers()
        self._calculate_cue_reward_distances()
        self._mark_spawn_locations()
        self._construct_visibility_map()
        self._make_visibility_graphs()
        self._compute_tile_maps()

    def _has_data(self, *names):
        """
        Test whether all named objects are stored in the h5 file.
        """
        with self.datafile:
            for name in names:
                if not self.datafile.has_node(f'/{name}'):
                    return False
        return True

    def _remove_arrays(self, *names):
        """
        Remove array data from the h5 file.
        """
        removed = []
        with self.datafile:
            for name in names:
                if self.datafile.has_node(f'/{name}'):
                    self.datafile.remove_node(f'/{name}')
                    delattr(self, name)
                    removed.append(f'{name}')

        self.out(f'Removed: {", ".join(removed)}')

    def _load_arrays(self, *names):
        """
        Read array data from the h5 file into instance attributes.
        """
        loaded = []
        with self.datafile:
            for name in names:
                arr = self.datafile.read_array(f'/{name}')
                setattr(self, name, arr)
                shape = 'x'.join(list(map(str, arr.shape)))
                if np.ma.isMA(arr):
                    loaded.append(f'{name}<{shape}:masked>')
                else:
                    loaded.append(f'{name}<{shape}>')
        self.out(", ".join(loaded), prefix='Loaded')

    def _store_arrays(self, imagefmt={}, **data):
        """
        Save arrays to Array objects in the h5 file.
        """
        saved = []
        with self.datafile:
            for name, arr in data.items():
                setattr(self, name, arr)
                res = self.datafile.new_array('/', name, arr)
                if arr.ndim == 2:
                    self._save_matrix_image(arr, name, **imagefmt)
                elif arr.ndim == 3:
                    for z in range(arr.shape[2]):
                        self._save_matrix_image(arr[...,z], f'{name}_{z:02d}',
                                **imagefmt)
                shape = 'x'.join(list(map(str, arr.shape)))
                if np.ma.isMA(arr):
                    saved.append(f'{name}<{shape}:masked>')
                else:
                    saved.append(f'{name}<{shape}>')
        self.out(f'Stored: {", ".join(saved)}')

    def _meshgrid(self):
        """
        Get a pixel-centered coordinate mesh-grid for the environment.
        """
        x = 0.5 + np.arange(*self.extent[:2])
        y = 0.5 + np.arange(*self.extent[2:])
        return np.array(np.meshgrid(x, y, indexing='ij'), dtype=DISTANCE_DTYPE)

    def _pipeline(self, *names):
        """
        Load data into instance attributes and return True if available and
        recompute is not being forced or step-specific read-only.
        """
        if not self.recompute:
            if self._has_data(*names):
                self._load_arrays(*names)
                return True
        return False

    def _rasterize_barriers(self):
        """
        Rasterize the environment with barriers.
        """
        if self._pipeline('G_B'): return

        B = np.zeros(self.shape, BINARY_DTYPE)

        for x1, y1, x2, y2 in self.B:
            if x1 == x2:
                ymin = min(y1,y2)
                ymax = max(y1,y2)
                B[x1,ymin:ymax+1] = 1
            elif y1 == y2:
                xmin = min(x1,x2)
                xmax = max(x1,x2)
                B[xmin:xmax+1,y1] = 1
            else:
                self.out(f'Non-rectilinear barrier: {(x1,y1,x2,y2)}',
                        error=True)

        self._store_arrays(G_B=B)

    def _scale_factor(self, P_exterior):
        """
        Calculate a radial, adjusted scale factor for the environment that
        loosely represents an inscribed circle if the interior space were
        reconfigured as a square.
        """
        return (np.sqrt(2)/2)*np.sqrt((~P_exterior).sum()/np.pi)

    def _create_environment_mask(self):
        """
        Flood fill the interior to create a mask of occupiable points.
        """
        if self._pipeline('G_P'):
            self.info['G_scale'] = self._scale_factor(self.G_P)
            return

        P = self.G_B.copy()
        target = 0
        barrier = 1
        repl = 2

        # Starting from each of the spawn disc center points, flood-fill the
        # barrier image to mark all interiorly occupiable points
        for x0, y0 in self.S0[:,:2]:
            Q = queue.deque()
            Q.append([x0,y0])
            while Q:
                N = Q.pop()
                W = N.copy()
                E = N.copy()
                y = N[1]
                while W[0] > 0 and P[W[0],y] == target:
                    W[0] -= 1
                while E[0] < self.width and P[E[0],y] == target:
                    E[0] += 1
                for x in range(W[0]+1, E[0]):
                    P[x,y] = repl
                    if P[x,y+1] == target:
                        Q.append([x,y+1])
                    if P[x,y-1] == target:
                        Q.append([x,y-1])

        # Convert values to {0,1} for {valid,masked}
        P[P != repl] = 1
        P[P == repl] = 0

        G_P = P.astype('?')
        self.info['G_scale'] = self._scale_factor(G_P)
        self._store_arrays(G_P=G_P)

    def _find_closest_barriers(self):
        """
        Find the closest barriers and store the interior normal vectors.
        """
        if self._pipeline('G_PD', 'G_PB', 'G_PN'): return

        P = self.G_P.astype('i2')
        PD = np.zeros(self.shape, DISTANCE_DTYPE)
        PB = np.zeros_like(PD)
        PN = np.zeros(self.shape + (2,), DISTANCE_DTYPE)
        halfsq = float(np.sqrt(2)/2)
        W, H, alpha = self.width, self.height, self.alpha
        B = np.hypot(W, H)
        U = np.array([[0       , 1]        ,
                      [0       , -1]       ,
                      [1       , 0]        ,
                      [-1      , 0]        ,
                      [halfsq  , halfsq]   ,
                      [halfsq  , -halfsq]  ,
                      [-halfsq , halfsq]   ,
                      [-halfsq , -halfsq]] , DISTANCE_DTYPE)
        w_d = np.empty_like(U)
        d = np.empty((U.shape[0],1), DISTANCE_DTYPE)
        k = np.empty_like(d)

        def min_normal_vec(P0, x, y):
            n = s = e = w = ne = se = nw = sw = 1

            while (y+n < H) and (P[x,y+n] == P0): n += 1
            if y+n >= H: n = B

            while (y-s >= 0) and (P[x,y-s] == P0): s += 1
            if y-s < 0: s = B

            while (x+e < W) and (P[x+e,y] == P0): e += 1
            if x+e >= W: e = B

            while (x-w >= 0) and (P[x-w,y] == P0): w += 1
            if x-w < 0: w = B

            while (x+ne < W) and (y+ne < H) and (P[x+ne,y+ne] == P0): ne += 1
            if (x+ne >= W) or (y+ne >= H): ne = B

            while (x+se < W) and (y-se >= 0) and (P[x+se,y-se] == P0): se += 1
            if (x+se >= W) or (y-se < 0): se = B

            while (x-nw >= 0) and (y+nw < H) and (P[x-nw,y+nw] == P0): nw += 1
            if (x-nw < 0) or (y+nw >= H): nw = B

            while (x-sw >= 0) and (y-sw >= 0) and (P[x-sw,y-sw] == P0): sw += 1
            if (x-sw < 0) or (y-sw < 0): sw = B

            # Save wall distances and compute the interior barrier coefficients
            d[:] = np.array([n, s, e, w, ne, se, nw, sw])[:,np.newaxis]
            kmax = 1 if P0 else np.exp(-d/alpha).max()

            # Inverse-distance weights in the interior and distance weights in
            # the exterior
            inout = 2*P0 - 1
            w_d[:] = d**inout
            w_d[np.isclose(w_d, B**inout)] = 0.0
            U_avg = np.average(inout*U, weights=w_d, axis=0)

            return (d.min(), kmax, U_avg)

        self.out('Starting barrier search...')
        i = 0
        for x in range(W):
            for y in range(H):
                PD[x,y], PB[x,y], PN[x,y] = min_normal_vec(P[x,y], x, y)
                i += 1
                if i % 1000 == 0:
                    self.out.printf('.')
        self.out.newline()

        # Median-filter the coefficient map and set all exterior points to the
        # maximum coefficient (1)
        k_alpha = int(alpha)
        if k_alpha % 2 == 0: k_alpha += 1
        PB = medfilt2d(PB, kernel_size=k_alpha)
        PB[self.G_P] = 1
        PB -= PB.min()
        PB /= PB.max()

        self._store_arrays(G_PD=PD, G_PB=PB, G_PN=PN)

    def _calculate_cue_reward_distances(self):
        """
        Calculate distances between points and cues/rewards.
        """
        if self._pipeline('D_PC', 'D_PR'): return

        PC = np.zeros(self.shape + (self.N_C,), DISTANCE_DTYPE)
        PR = np.zeros(self.shape + (self.N_R,), DISTANCE_DTYPE)
        XX, YY = self._meshgrid()

        for i, (cx,cy) in enumerate(self.C):
            PC[...,i] = np.hypot(XX - cx, YY - cy)

        for i, (rx,ry) in enumerate(self.R):
            PR[...,i] = np.hypot(XX - rx, YY - ry)

        Cmask = np.empty(PC.shape, '?')
        Cmask[:] = self.G_P[...,np.newaxis]
        PC = np.ma.MaskedArray(data=PC, mask=Cmask)

        Rmask = np.empty(PR.shape, '?')
        Rmask[:] = self.G_P[...,np.newaxis]
        PR = np.ma.MaskedArray(data=PR, mask=Rmask)

        self._store_arrays(D_PC=PC, D_PR=PR)

    def _mark_spawn_locations(self):
        """
        Compute the allowable spawn locations.
        """
        if self._pipeline('G_PS', 'X0'): return

        PS = np.zeros(self.shape, BINARY_DTYPE)
        XX, YY = self._meshgrid()

        for i, (xs, ys, radius) in enumerate(self.S0):
            D = np.hypot(XX - xs, YY - ys)
            PS[D<=radius] = 1

        PS = np.ma.MaskedArray(data=PS, mask=self.G_P)
        X0 = np.array(PS.nonzero()).T

        # Verify that the spawn points match the matrix
        P0 = np.zeros_like(PS)
        P0[tuple(X0.T)] = 1
        assert np.all(P0 == PS), 'spawn point mismatch'

        self._store_arrays(G_PS=PS, X0=X0)

    def _construct_visibility_map(self):
        """
        Construct a coarse hexagonal grid for visibility computations.
        """
        if self._pipeline('H', 'G_H'):
            self.info['N_H'] = self.N_H = self.H.shape[0]
            return

        H = []
        angles = np.linspace(0, 2*np.pi, 7)[:-1]
        Q = queue.deque()
        Q.append(self.origin)

        while Q:
            v = Q.pop()
            existing = False
            for u in H:
                if np.isclose(v[0], u[0]) and np.isclose(v[1], u[1]):
                    existing = True
                    break
            if existing:
                continue
            if not (self.extent[0] <= v[0] < self.extent[1]):
                continue
            if not (self.extent[2] <= v[1] < self.extent[3]):
                continue
            Q.extend([(v[0] + self.k_H*np.cos(a), v[1] + self.k_H*np.sin(a))
                for a in angles])
            H.append(v)
            self.out.printf('.')
        self.out.newline()

        # Mask grid points and sort from top-left to bottom-right
        Hint = np.round(H).astype(TILE_DTYPE)
        Hvalid = Hint[~self.G_P[tuple(Hint.T)]]
        H = Hvalid[np.lexsort(tuple(reversed(tuple(Hvalid.T))))]

        # Store filtered grid points in an image matrix
        G_H = np.zeros(self.shape, BINARY_DTYPE)
        G_H[tuple(H.T)] = 1
        G_H = np.ma.MaskedArray(data=G_H, mask=self.G_P)

        self._store_arrays(H=H, G_H=G_H)

    def _make_visibility_graphs(self):
        """
        Make several visibility graphs for relating objects and locations.
        """
        if self._pipeline('V_HH', 'V_HR', 'V_HC'): return

        N_H = len(self.H)
        HH = np.zeros((N_H, N_H), BOOL_DTYPE)
        HC = np.zeros((N_H, self.N_C), BOOL_DTYPE)
        HR = np.zeros((N_H, self.N_R), BOOL_DTYPE)

        for i, (x0, y0) in enumerate(self.H):
            self.out.printf('.')
            for V, S in [(HH, self.H), (HC, self.C), (HR, self.R)]:
                for j, (x1, y1) in enumerate(S):
                    if (x0 == x1) and (y0 == y1):
                        V[i,j] = True
                        continue
                    theta = np.arctan2(float(y1 - y0), float(x1 - x0))
                    dx, dy = np.cos(theta), np.sin(theta)
                    xgtr = x1 > x0
                    ygtr = y1 > y0
                    xf, yf = float(x0), float(y0)
                    while True:
                        xf += dx
                        yf += dy
                        xri = int(round(xf))
                        yri = int(round(yf))
                        if self.G_P[xri,yri]:
                            break
                        xgtr_ = x1 > xri
                        ygtr_ = y1 > yri
                        if (xgtr_ != xgtr) or (ygtr_ != ygtr):
                            V[i,j] = True
                            break
        self.out.newline()

        self._store_arrays(V_HH=HH, V_HC=HC, V_HR=HR, imagefmt={'asmap':False})

    def _compute_tile_maps(self):
        """
        Create maps of points, cues, and rewards to tile index.
        """
        if self._pipeline('G_PH', 'C_H', 'R_H'): return

        N_H = len(self.H)
        CH = np.empty((self.N_C,), TILE_INDEX_DTYPE)
        RH = np.empty((self.N_R,), TILE_INDEX_DTYPE)

        # Broadcast the point mask between (x,y)-coordinates and tile points
        xy_mesh_tile_shape = (2,) + self.shape + (N_H,)
        VV = np.empty(xy_mesh_tile_shape, '?')
        VV[:] = self.G_P[np.newaxis,...,np.newaxis]

        # Broadcast the meshgrid into tile points
        XY = np.empty(xy_mesh_tile_shape, DISTANCE_DTYPE)
        XY[:] = self._meshgrid()[...,np.newaxis]
        XY = np.ma.MaskedArray(data=XY, mask=VV)

        # Splitcast the tile points through the meshgrid
        HH = np.empty(xy_mesh_tile_shape, DISTANCE_DTYPE)
        HH[:] = self.H.T[:,np.newaxis,np.newaxis,:]
        HH = np.ma.MaskedArray(data=HH, mask=VV)

        # Find indexes of closest tiles to every point in the meshgrid
        D_XH = XY - HH
        PH = np.ma.MaskedArray(
                data=np.argmin(np.hypot(D_XH[0], D_XH[1]), axis=2).astype(
                    TILE_INDEX_DTYPE),
                mask=self.G_P)

        # Directly index the point-tile map for cue/reward tiles
        CH[:] = PH[tuple(self.C.T)]
        RH[:] = PH[tuple(self.R.T)]

        self._store_arrays(G_PH=PH, C_H=CH, R_H=RH,
                imagefmt=dict(cmap='cool', mask_color='k'))
