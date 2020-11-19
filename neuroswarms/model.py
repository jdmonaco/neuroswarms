"""
Neural swarming controller models for multi-agent and single-entity simulations.

Author: Joseph Monaco (jmonaco@jhu.edu)
Affiliation: Johns Hopkins University
Created: 2019-05-12
Updated: 2020-11-16

Related paper:

  Monaco, J.D., Hwang, G.M., Schultz, K.M. et al. Cognitive swarming in complex
      environments with attractor dynamics and oscillatory computing. Biol Cybern
      114, 269–284 (2020). https://doi.org/10.1007/s00422-020-00823-z

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.ph
"""

import os
import json
import platform
import subprocess

import numpy as np
from numpy import (ndarray, array, empty, empty_like, zeros, zeros_like, ones,
        eye, cos, sin, tanh, inf, pi, newaxis as AX, dot, diag, hypot, average,
        arange, broadcast_to, isclose, isfinite, histogram, linspace, exp,
        log1p, sqrt, nextafter, any)
from numpy.random import seed, rand, randn, randint

from matplotlib import pyplot as plt
from matplotlib.colors import colorConverter
from matplotlib.animation import FuncAnimation

from .utils.geometry import EnvironmentGeometry, map_index
from .utils.geometry import CUE_COLOR, REWARD_COLOR
from .utils.context import NeuroswarmsContext
from .utils.types import *

from .matrix import *
from . import MOVIE_FPS, MOVIE_DPI


TWOPI = 2*pi
ONEMEPS = nextafter(1, 0)
ZEROPEPS = nextafter(0, 1)


class NeuroswarmsModel(NeuroswarmsContext):

    """
    Model controller class.
    """

    def set_parameters(self, pfile=None, **params):
        """
        Set model parameters according to file, keywords, or defaults.
        """
        # Import parameters from file before keyword values
        if pfile is not None:
            if not pfile.endswith('.json'):
                pfile += '.json'
            if os.path.isfile(pfile):
                pfilepath = pfile
            else:
                runpfile = os.path.join(self._rundir, pfile)
                projpfile = os.path.join(self._projdir, pfile)
                if os.path.isfile(runpfile):
                    pfilepath = runpfile
                elif os.path.isfile(projpfile):
                    pfilepath = projpfile
                else:
                    self.out(pfile, prefix='MissingParamFile', error=True)
                    raise ValueError('Unable to find parameter file')
            with open(pfilepath, 'r') as fd:
                params_from_file = json.load(fd)
            params_from_file.update(params)
            params = params_from_file
            self.out(pfilepath, prefix='ParameterFile')

        # Set default parameter values
        defaults = dict(
            single_agent = False,
            env          = 'test',
            dt           = 0.01,
            duration     = 10.0,
            rnd_seed     = 'neuroswarms',
            rnd_phase    = True,
            N_S          = 300,
            D_max        = 1.0,
            E_max        = 3e3,
            mu           = 0.9,
            m_multi      = 0.3,
            m_single     = 3.0,
            m_std        = 0.0,
            sigma        = 1.0,
            kappa        = 1.0,
            eta_S        = 1.0,
            eta_R        = 1.0,
            f_0          = 0.0,
            f_I          = 1.0,
            g_C          = 0.2,
            g_R          = 0.3,
            g_S          = 0.5,
            tau_c        = 0.5,
            tau_r        = 0.5,
            tau_q        = 0.1,
            d_rad        = 12.0,
        )

        # Write JSON file with parameter defaults
        kwjson = dict(indent=2, separators=(', ', ': '))
        dfltpath = self.path('defaults.json')
        with open(dfltpath, 'w') as fd:
            json.dump({k:v for k,v in defaults.items() if v is not None},
                    fd, **kwjson)

        # Write JSON file with final parameters
        dparams = defaults.copy()
        dparams.update(params)
        parampath = self.path('params.json')
        with open(parampath, 'w') as fd:
            json.dump({k:v for k,v in dparams.items() if v is not None},
                    fd, **kwjson)

        # Set parameters as global variables for the simulation
        self.out('Model parameters:')
        for name, dflt in defaults.items():
            exec(f'global {name}; {name} = dparams[\'{name}\']')
            val = globals()[name]
            if val == dflt:
                logstr = f' - {name} = {dflt}'
            else:
                logstr = f' * {name} = {val} [default: {dflt}]'
            self.out(logstr, hideprefix=True)

        # Import environmental geometry into the global scope, into the
        # persistent key-value store, and as instance attributes of the
        # simulation object
        global E
        self.hline()
        E = self.E = EnvironmentGeometry(env)
        Ivars = list(sorted(E.info.keys()))
        Avars = list(sorted(filter(lambda k: isinstance(getattr(E, k),
            ndarray), E.__dict__.keys())))
        self.out(repr(E), prefix='Geometry')
        for v in Ivars:
            exec(f'global {v}; self.{v} = {v} = E.info[\'{v}\']')
            self.out(f' - {v} = {getattr(self, v)}', prefix='Geometry', hideprefix=True)
        for k in Avars:
            exec(f'global {k}; self.{k} = {k} = E.{k}')
            self.out(' - {} ({})', k, 'x'.join(list(map(str, getattr(self,
                k).shape))), prefix='Geometry', hideprefix=True)

        # Other parameters - Dependent values
        self.hline()
        self.out('Dependent parameters:')
        depvalues = dict(
            N            = 1 if single_agent else N_S,
            m_mean       = m_single if single_agent else m_multi,
            D_max_scaled = D_max*G_scale,
            sigma_scaled = sigma*G_scale,
            kappa_scaled = kappa*G_scale,
        )
        for name, val in depvalues.items():
            exec(f'global {name}; {name} = {val}')
            self.out(f' - {name} = {val}', hideprefix=True)
        self.hline()

    def simulate(self, tag=None, paramfile=None, **params):
        """
        Run a simulation trial of the NeuroSwarms model with given parameters.

        Arguments
        ---------

        tag       : string tag to be added to the output movie file
        paramfile : absolute path or filename for a parameter fille; for a
                    filename, the project and run directories will be searched

        Keyword arguments provide parameter values that supercede those found
        in a `paramfile`.
        """
        self.set_parameters(paramfile, **params)
        seed(sum(list(map(ord, rnd_seed))))

        # Array allocation of model variables
        X        = empty((N, 2),   DISTANCE_DTYPE)   # actual position
        X_hat    = zeros((N, 2),   DISTANCE_DTYPE)   # intermediate position for calculation
        X_S      = empty((N_S, 2), DISTANCE_DTYPE)   # internal/'mental' field position
        X_S_hat  = zeros((N_S, 2), DISTANCE_DTYPE)   # intermediate field position for calculation
        dX       = zeros((N_S, 2), DISTANCE_DTYPE)   # actual position shift vectors
        dX_norm  = zeros((N_S, 2), DISTANCE_DTYPE)   # actual position shift magnitudes (for barriers)
        dX_S     = zeros((N_S, 2), DISTANCE_DTYPE)   # field position shift vectors
        M        = empty((N, 1),   KILOGRAM_DTYPE)   # agent mass
        v_max    = empty((N, 1),   DISTANCE_DTYPE)   # max velocity
        v        = zeros((N, 2),   DISTANCE_DTYPE)   # actual velocity vectors
        v_S      = zeros((N, 2),   DISTANCE_DTYPE)   # field/particle velocity vectors
        v_m      = zeros((N, 2),   DISTANCE_DTYPE)   # momentum-based velocity vectors
        v_norm   = zeros((N, 1),   DISTANCE_DTYPE)   # linear speed
        v_k      = zeros((N, 2),   DISTANCE_DTYPE)   # kinetic-constrained velocity vectors
        b        = zeros((N, 1),   DISTANCE_DTYPE)   # agent barrier coefficients
        n_b      = zeros((N, 2),   DISTANCE_DTYPE)   # agent barrier-normal vectors
        b_S      = zeros((N_S, 1), DISTANCE_DTYPE)   # field/particle barrier coefficients
        n_bS     = zeros((N_S, 2), DISTANCE_DTYPE)   # field/particle barrier-normal vectors
        H        = empty((N, 1),   TILE_INDEX_DTYPE) # agent visibility tile index
        H_S      = empty((N_S, 1), TILE_INDEX_DTYPE) # field/particle visibility tile index
        Delta    = empty((N_S, 1), DISTANCE_DTYPE)   # single-entity distance to virtual particles
        V_Delta  = empty_like(Delta)                 # single-entity visibility matrix
        D_S      = empty((N_S, N_S), DISTANCE_DTYPE) # inter-agent pairwise distance matrix
        D_S_up   = empty_like(D_S)                   # updated D_S after learning rule update
        V_S      = empty_like(D_S)                   # inter-agent visibility matrix
        V_eye    = 1 - eye(N_S)                      # identity visibility mask
        V_C_S    = empty((N_S, N_C), DISTANCE_DTYPE) # visible cue preferences
        V_C      = empty_like(V_C_S)                 # agent-cue visibility matrix
        D_C      = empty((N_S, N_C), DISTANCE_DTYPE) # agent-cue distance matrix
        D_R      = empty((N_S, N_R), DISTANCE_DTYPE) # agent-reward distance matrix
        D_R_up   = empty_like(D_R)                   # updated D_R after learning rule update
        V_R      = empty_like(D_R)                   # agent-reward visibility matrix
        V_R_capt = zeros_like(V_R, dtype=bool)       # boolean flag indicating "captured" rewards
        W_S      = empty((N_S, N_S), WEIGHT_DTYPE)   # recurrent swarming weight matrix
        W_R      = empty((N_S, N_R), WEIGHT_DTYPE)   # feedforward reward-approach weight matrix
        dX_R     = zeros((N_S, 2),   DISTANCE_DTYPE) # reward-related position shift vectors
        W_dX     = empty((N_S, 2),   DISTANCE_DTYPE) # single-entity weighting of virtual particles
        Theta    = empty((N_S, 1),   PHASE_DTYPE)    # agent phase state
        dTheta   = empty((N_S, N_S), PHASE_DTYPE)    # inter-agent pairwaise phase differences
        I_C      = zeros((N_S, 1),   WEIGHT_DTYPE)   # total cue-based input
        I_R      = zeros((N_S, 1),   WEIGHT_DTYPE)   # total reward-based input
        I_S      = zeros((N_S, 1),   WEIGHT_DTYPE)   # total swarming-based input
        p        = zeros((N_S, 1),   WEIGHT_DTYPE)   # net neural activation
        c        = zeros((N_S, N_C), WEIGHT_DTYPE)   # cue-based input matrix
        r        = zeros((N_S, N_R), WEIGHT_DTYPE)   # reward-based input matrix
        q        = zeros((N_S, N_S), WEIGHT_DTYPE)   # swarming-based input matrix

        # Nonzero initialization of model variables
        M[:]           = m_mean + m_std*randn(N, 1)
        v_max[:]       = sqrt((2*E_max) / M)
        X[:]           = E.sample_spawn_points(N=N)
        if single_agent:
            X_S[:]     = E.sample_spawn_points(N=N_S)
        else:
            X_S[:]     = X
        H[:]           = G_PH[map_index(X)][:,AX]
        H_S[:]         = G_PH[map_index(X_S)][:,AX]
        Delta[:]       = distances(X, X_S)
        if single_agent:
            V_Delta[:] = V_HH[tile_index(H, H_S)]
            D_S[:]     = pairwise_distances(X_S, X_S)
            V_S[:]     = V_eye*V_HH[pairwise_tile_index(H_S, H_S)]
            V_C[:]     = V_HC[H_S[:,0]]  # TODO: nonlocal
            D_C[:]     = D_PC[map_index(X_S)]
            D_R[:]     = D_PR[map_index(X_S)]
            V_R[:]     = V_HR[H_S[:,0]]  # TODO: nonlocal
        else:
            V_Delta[:] = 1.0
            D_S[:]     = pairwise_distances(X, X)
            V_S[:]     = V_eye*(D_S <= D_max_scaled)* \
                            V_HH[pairwise_tile_index(H, H)]
            V_C[:]     = V_HC[H[:,0]]
            D_C[:]     = D_PC[map_index(X)]
            D_R[:]     = D_PR[map_index(X)]
            V_R[:]     = V_HR[H[:,0]]
        W_S[:]         = V_S*exp(-D_S**2/(2*sigma_scaled**2))  # Eqn. 3
        D_S_up[:]      = inf
        V_C_S[:]       = V_HC[randint(N_H, size=N_S)]
        W_R[:]         = V_R*exp(-D_R/kappa_scaled)  # Eqn. 4
        D_R_up[:]      = inf
        if rnd_phase:
            Theta[:]   = TWOPI*rand(N_S, 1)
        else:
            Theta[:]   = 0.0
        dTheta[:]      = pairwise_phasediffs(Theta, Theta)
        c[:]           = V_C*V_C_S
        r[:]           = V_R
        q[:]           = V_S*cos(dTheta)

        # Tracking simulation timesteps and recording data for each movie frames
        _timesteps = arange(0, duration+dt, dt)
        _nframes   = len(_timesteps)
        _X         = zeros((_nframes,) + X.shape, X.dtype)
        _X_S       = zeros((_nframes,) + X_S.shape, X_S.dtype)
        _Theta     = zeros((_nframes,) + Theta.shape, Theta.dtype)

        # Plot formatting parameters
        lw = 0.5
        agent_sz = 6
        cue_markers = 'v^<>12348sphHDd'
        reward_marker = '*'
        reward_color = {False:REWARD_COLOR, True:'w'}  # captured rewards turn white
        cr_fmt = dict(linewidths=lw, edgecolors='k', alpha=0.7, zorder=0)
        x_fmt = dict(marker='o', s=agent_sz, linewidths=lw, edgecolors='k',
                alpha=0.5, zorder=10)
        xs_fmt = x_fmt.copy()
        xs_fmt.update(marker='.', s=3, c='k', linewidths=0, alpha=1, zorder=5)
        text_xy = (0.5*width, 0.05*height)
        text_fmt = dict(va='top', ha='center', color='gray',
                fontsize='xx-small', fontweight='normal')

        # Color array and colormap for visualizing agent phase state
        swarm_colors = zeros((N_S, 4))
        X_cmap = plt.get_cmap('hsv')

        # Function to initialize plotting window for figure animation
        plt.ioff()
        _fig, _ax = E.figure(tag=tag, mapname='G_P', clear=True)
        self.artists = []
        self.cues = []
        def init():
            for i, _ in enumerate(C):
                self.cues.append(plt.scatter([], [], s=[],
                        marker=cue_markers[randint(len(cue_markers))],
                        c=CUE_COLOR, **cr_fmt))
            self.rewards = plt.scatter([], [], s=[], marker=reward_marker,
                    c=REWARD_COLOR, **cr_fmt)

            self._X_scatter = _ax.scatter([], [], cmap='hsv', vmin=0.0,
                    vmax=TWOPI, **x_fmt)
            self._X_S_scatter = _ax.scatter([], [], **xs_fmt)

            self._timestamp_text = _ax.text(text_xy[0], text_xy[1], '',
                    **text_fmt)

            self.artists.extend(self.cues)
            self.artists.extend([self.rewards, self._timestamp_text,
                self._X_scatter, self._X_S_scatter])

            return self.artists

        #
        # Update loop for Matplotlib figure animation
        #

        def update(n):
            t = _timesteps[n]

            # Draw the cues and rewards on the first timestep
            if n == 0:
                for i, xy in enumerate(C):
                    cue = self.cues[i]
                    cue.set_offsets(xy[AX])
                    cue.set_sizes([(k_H/3*(1+C_W[i]/3))**2])
                self.rewards.set_offsets(R)
                self.rewards.set_sizes((k_H/3*(1+R_W/3))**2)
            elif single_agent:
                rcols = [reward_color[capt] for capt in V_R_capt[0]]
                self.rewards.set_facecolors(rcols)

            # Update the progress bar
            if _nframes >= 50 and n % int(_nframes/50) == 0:
                self.box(filled=False, color='purple')

            # Update the animated timestamp text string
            self._timestamp_text.set_text(f't = {t:0.3f} s')

            # Update the scatter plots
            self._X_scatter.set_offsets(X)
            self._X_S_scatter.set_offsets(X_S)
            swarm_colors[:] = X_cmap(Theta.squeeze()/TWOPI)

            # Color the virtual particles (single-entity) or swarm (multi-agent)
            if single_agent:
                if n == 0:
                    self._X_scatter.set_facecolor('limegreen')
                self._X_S_scatter.set_color(swarm_colors)
                self._X_S_scatter.set_sizes(1 + 5*p[:,0])  # size ~ activation
            else:
                self._X_scatter.set_facecolor(swarm_colors)

            # Pre-synaptic input updates (Eqns. 5-7 in the paper)
            c[:] += dt/tau_c*(V_C*V_C_S - c)
            r[:] += dt/tau_r*(V_R - r)
            q[:] += dt/tau_q*(V_S*cos(dTheta) - q)

            # Net input currents with count normalization and exclusion of
            # zero-count units (Eqns. 8-10)
            N_VC = V_C.sum(axis=1)[:,AX]  # no. visible cues
            N_VR = V_R.sum(axis=1)[:,AX]  # no. visible rewards
            N_VS = V_S.sum(axis=1)[:,AX]  # no. visible agents
            cnz = N_VC.nonzero()[0]       # >0 visible cues
            rnz = N_VR.nonzero()[0]       # >0 visible rewards
            snz = N_VS.nonzero()[0]       # >0 visible neighbors
            I_C[cnz] = g_C/N_VC[cnz]*c[cnz].sum(axis=1)[:,AX]
            I_R[rnz] = g_R/N_VR[rnz]*(W_R[rnz]*r[rnz]).sum(axis=1)[:,AX]
            I_S[snz] = g_S/N_VS[snz]*(W_S[snz]*q[snz]).sum(axis=1)[:,AX]

            # Post-synaptic activation updates (Eqns. 11-12)
            p[:]     = I_S + I_C + I_R
            p[p<0]   = 0.0
            Theta[:] += dt*TWOPI*(f_I*p + f_0)

            # Synaptic weight updates for visible connections (Eqns. 13-14)
            W_R[:] += V_Delta*V_R*dt*eta_R*p*(r - p*W_R)
            W_S[:] += V_Delta*V_S*dt*eta_S*p*(q - p*W_S)

            # For precision and masking, convert non-zero weights to distances
            # via log1p (thus requiring subtraction of 1, since we are not
            # using expm1 to create weights because we do not want learning to
            # operate on negative weights). Eqns. 15-16.
            W_R.clip(0, ONEMEPS, out=W_R)
            W_S.clip(0, ONEMEPS, out=W_S)
            W_R_nz         = W_R != 0
            W_S_nz         = W_S != 0
            D_R_up[W_R_nz] = -kappa_scaled*log1p(W_R[W_R_nz] - 1)
            D_S_up[W_S_nz] = sqrt(-(2*sigma_scaled**2)* \
                                log1p(W_S[W_S_nz] - 1))

            # Compute the reward and somatic motion updates (Eqns. 17-19)
            dX_R[:] = reward_motion_update(D_R_up, D_R, X_S, R, V_R)
            if single_agent:
                dX_S[:] = somatic_motion_update(D_S_up, D_S, X_S, V_S)
            else:
                dX_S[:] = somatic_motion_update(D_S_up, D_S, X, V_S)
            dX[:] = 0.5 * (dX_S + dX_R)

            # Motion update: Geometric constraints for internal fields
            X_S_hat[:] = X_S + dX
            b_S[:]     = G_PB[map_index(X_S_hat)][:,AX]
            n_bS[:]    = G_PN[map_index(X_S_hat)]
            dX_norm[:] = hypot(dX[:,0], dX[:,1])[:,AX]
            dX[:]      = (1-b_S)*dX + b_S*dX_norm*n_bS  # Eqn. 20
            X_S[:]     = X_S + dX

            # Motion update: Agent velocity guided by internal fields
            if single_agent:

                # Effective velocity ~ p-cubed-weighted average of approaching
                # vectors to the visible internal field centers (Eqn. 25)
                W_dX[:] = V_Delta*p**3
                if any(W_dX):
                    v_S[:] = average(X_S - X, weights=W_dX, axis=0) / dt
                else:
                    v_S[:] = 0.0

            else:

                # Each swarm agent simply approaches its internal place field
                v_S[:] = (X_S - X)/dt

            # Motion update: Energetic constraints on momentums (Eqns. 21-23)
            v_m[:]    = mu*v + (1-mu)*v_S
            v_norm[:] = hypot(v_m[:,0], v_m[:,1])[:,AX]
            v_nz      = v_norm.nonzero()[0]
            v_k[v_nz] = v_max[v_nz]*tanh(v_norm[v_nz]/v_max[v_nz])*(
                                v_m[v_nz]/v_norm[v_nz])

            # Motion update: Geometric (barrier) constraints (Eqn. 24)
            X_hat[:]  = X + v_k*dt
            b[:]      = G_PB[map_index(X_hat)][:,AX]
            n_b[:]    = G_PN[map_index(X_hat)]
            v[:]      = (1-b)*v_k + b*hypot(v_k[:,0], v_k[:,1])[:,AX]*n_b
            X[:]     += v*dt

            # Record current values of output variables
            _X[n]     = X
            _X_S[n]   = X_S
            _Theta[n] = Theta

            # Geometry update: Distances and visibility
            v_k[:]          = 0.0
            H[:]            = G_PH[map_index(X)][:,AX]
            H_S[:]          = G_PH[map_index(X_S)][:,AX]
            Delta[:]        = distances(X, X_S)
            if single_agent:
                V_Delta[:]  = V_HH[tile_index(H, H_S)]
                D_S[:]      = pairwise_distances(X_S, X_S)
                V_S[:]      = V_eye*V_HH[pairwise_tile_index(H_S, H_S)]
                V_C[:]      = V_HC[H_S[:,0]]  # TODO: nonlocal
                D_C[:]      = D_PC[map_index(X_S)]
                D_R[:]      = D_PR[map_index(X_S)]
                V_R[:]      = V_HR[H_S[:,0]]  # TODO: nonlocal
                V_R_capt[:] |= (D_PR[map_index(X)] < d_rad)  # capture by single-entity
            else:
                D_S[:]      = pairwise_distances(X, X)
                V_S[:]      = V_eye*(D_S <= D_max_scaled)* \
                                 V_HH[pairwise_tile_index(H, H)]
                V_C[:]      = V_HC[H[:,0]]
                D_C[:]      = D_PC[map_index(X)]
                D_R[:]      = D_PR[map_index(X)]
                V_R[:]      = V_HR[H[:,0]]
                V_R_capt[:] |= (D_R < d_rad)  # capture by individual agents

            # Make captured rewards invisible
            V_R[V_R_capt] = 0

            # Model update: Internal state variables
            dX[:] = 0.0
            Theta[:] %= TWOPI
            dTheta[:] = pairwise_phasediffs(Theta, Theta)

            # Reset intermediate variables
            I_C[:]    = 0.0
            I_R[:]    = 0.0
            I_S[:]    = 0.0
            D_R_up[:] = inf
            D_S_up[:] = inf

            # Reset weights to visibility- & distance-based values (Eqns. 3-4)
            W_R[:] = V_R*exp(-D_R/kappa_scaled)
            W_S[:] = V_S*exp(-D_S**2/(2*sigma_scaled**2))

            # For matplolib.animation.FuncAnimation:
            return self.artists

        # Create the Matplotlib figure animation object
        anim = FuncAnimation(fig=_fig, func=update, frames=range(_nframes),
               init_func=init, interval=10, repeat=False, blit=True)

        # Run the simulation and save the animation to a movie file
        if tag: fn = '{}+{}.mp4'.format(self._name, tag)
        else:   fn = '{}.mp4'.format(self._name)
        self.savepath = self.path(fn)
        anim.save(self.savepath, fps=MOVIE_FPS, dpi=MOVIE_DPI)
        self.hline()
        plt.close(_fig)
        plt.ion()

        # Save simulation data to the datafile run root
        self.save_array(_timesteps, 't')
        self.save_array(_X, 'X')
        self.save_array(_X_S, 'X_S')
        self.save_array(_Theta, 'Theta')

        # Play the movie if it was successfully saved
        if os.path.isfile(self.savepath):
            self.out(self.savepath, prefix='SavedMovie')
            if platform.system() != "Windows":
                self.play_movie(self.savepath)

    def play_movie(self, movie_path):
        """
        Play the movie file located at the given path.

        Note: This code will call the command-line movie player `mpv` if it is
        installed and available on the local search path. On macOS systems, it
        can be installed with homebrew via `brew install mpv`. On linux,
        it can be installed in standard ways, such as `sudo apt install mpv`.
        """
        dv = subprocess.DEVNULL
        devnull = dict(stdout=dv, stderr=dv)
        p = subprocess.run(['which', 'mpv'], **devnull)
        if p.returncode != 0:
            self.out('Player \'mpv\' is missing', error=True)
            return

        mpv_cmd = ['mpv', '--loop=yes', '--ontop=yes']
        mpv_cmd.append(movie_path)
        subprocess.run(mpv_cmd, **devnull)
