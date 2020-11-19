"""
Matrix operations for neuroswarms models.

Author: Joseph Monaco (jmonaco@jhu.edu)
Affiliation: Johns Hopkins University
Created: 2019-05-12
Updated: 2020-11-16

Related paper:

  Monaco, J.D., Hwang, G.M., Schultz, K.M. et al. Cognitive swarming in complex
      environments with attractor dynamics and oscillatory computing. Biol Cybern
      114, 269–284 (2020). https://doi.org/10.1007/s00422-020-00823-z

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php
"""

__all__ = ('tile_index', 'pairwise_tile_index', 'pairwise_distances',
           'distances', 'pairwise_phasediffs', 'pairwise_unit_diffs',
           'somatic_motion_update', 'reward_motion_update')

from numpy import (empty, zeros, newaxis as AX, swapaxes, hypot, sin, inf,
        broadcast_arrays, broadcast_to)

from .utils.types import *


DEBUGGING = False


def _check_ndim(Mstr, M, ndim):
    assert M.ndim == ndim, f'{Mstr}.ndim != {ndim}'

def _check_shape(Mstr, M, shape, axis=None):
    if axis is None:
        assert M.shape == shape, f'{Mstr}.shape != {shape}'
    else:
        assert M.shape[axis] == shape, f'{Mstr}.shape[{axis}] != {shape}'


def tile_index(A, B):
    """
    Entrywise comparison index of tile index (column) vectors.
    """
    AA, BB = broadcast_arrays(A, B)
    if DEBUGGING:
        shape = (max(A.shape[0], B.shape[0]), 1)
        _check_shape('AA', AA, shape)
        _check_shape('BB', BB, shape)

    return (AA, BB)

def pairwise_tile_index(A, B):
    """
    Pairwise comparison index of tile index (column) vectors.
    """
    AA, BB = broadcast_arrays(A, B.T)
    if DEBUGGING:
        shape = (len(A), len(B))
        _check_shape('AA', AA, shape)
        _check_shape('BB', BB, shape)

    return (AA, BB)

def pairwise_phasediffs(A, B):
    """
    Compute synchronizing phase differences between phase pairs.
    """
    N_A = len(A)
    N_B = len(B)
    DD_shape = (N_A, N_B)
    if DEBUGGING:
        _check_ndim('A', A, 2)
        _check_ndim('B', B, 2)
        _check_shape('A', A, 1, axis=1)
        _check_shape('B', B, 1, axis=1)

    return B.T - A

def distances(A, B):
    """
    Compute distances between points in entrywise order.
    """
    AA, BB = broadcast_arrays(A, B)
    shape = AA.shape
    if DEBUGGING:
        _check_ndim('AA', AA, 2)
        _check_ndim('BB', BB, 2)
        _check_shape('AA', AA, 2, axis=1)
        _check_shape('BB', BB, 2, axis=1)

    return hypot(AA[:,0] - BB[:,0], AA[:,1] - BB[:,1])[:,AX]

def pairwise_unit_diffs(A, B):
    """
    Compute attracting unit-vector differences between pairs of points.
    """
    DD = pairwise_position_deltas(A, B)
    D_norm = hypot(DD[...,0], DD[...,1])
    nz = D_norm.nonzero()
    DD[nz] /= D_norm[nz][...,AX]
    return DD

def pairwise_distances(A, B):
    """
    Compute distances between pairs of points.
    """
    DD = pairwise_position_deltas(A, B)
    return hypot(DD[...,0], DD[...,1])

def pairwise_position_deltas(A, B):
    """
    Compute attracting component deltas between pairs of points.
    """
    N_A = len(A)
    N_B = len(B)
    if DEBUGGING:
        _check_ndim('A', A, 2)
        _check_ndim('B', B, 2)
        _check_shape('A', A, 2, axis=1)
        _check_shape('B', B, 2, axis=1)

    # Broadcast the first position matrix
    AA = empty((N_A,N_B,2), DISTANCE_DTYPE)
    AA[:] = A[:,AX,:]

    return B[AX,...] - AA

def somatic_motion_update(D_up, D_cur, X, V):
    """
    Compute updated positions by averaging pairwise difference vectors for
    mutually visible pairs with equal bidirectional adjustments within each
    pair. The updated distance matrix does not need to be symmetric; it
    represents 'desired' updates based on recurrent learning.

    :D_up: R(N,N)-matrix of updated distances
    :D_cur: R(N,N)-matrix of current distances
    :X: R(N,2)-matrix of current positions
    :V: {0,1}(N,2)-matrix of current agent visibility
    :returns: R(N,2)-matrix of updated positions
    """
    N = len(X)
    D_shape = (N, N)
    if DEBUGGING:
        _check_ndim('X', X, 2)
        _check_shape('X', X, 2, axis=1)
        _check_shape('D_up', D_up, D_shape)
        _check_shape('D_cur', D_cur, D_shape)
        _check_shape('V', V, D_shape)

    # Broadcast field position matrix and its transpose
    XX = empty((N,N,2))
    XX[:] = X[:,AX,:]
    XT = swapaxes(XX, 0, 1)

    # Find visible & valid values (i.e., corresponding to non-zero weights)
    #
    # NOTE: The normalizing factor is divided by 2 because the somatic update
    # represents one half of the change in distance between a pair of units.
    D_inf = D_up == inf
    norm = V * ~D_inf
    N = norm.sum(axis=1)
    valid = N.nonzero()[0]
    norm[valid] /= 2*N[valid,AX]

    # Zero out the inf elements of the updated distance matrix and corresponding
    # elements in the current distance matrix
    D_up[D_inf] = D_cur[D_inf] = 0.0

    # Construct the agent-agent avoidant unit vectors
    DX = XX - XT
    DX_norm = hypot(DX[...,0], DX[...,1])
    valid = DX_norm.nonzero()
    DX[valid] /= DX_norm[valid][:,AX]

    return (norm[...,AX]*(D_up - D_cur)[...,AX]*DX).sum(axis=1)

def reward_motion_update(D_up, D_cur, X, R, V):
    """
    Compute updated positions by averaging reward-based unit vectors for
    adjustments of the point only. The updated distance matrix represents
    'desired' updates based on reward learning.

    :D_up: R(N,N_R)-matrix of updated distances between points and rewards
    :D_cur: R(N,N_R)-matrix of current distances between points and rewards
    :X: R(N,2)-matrix of current point positions
    :R: R(N_R,2)-matrix of current reward positions
    :V: {0,1}(N_R,2)-matrix of current agent-reward visibility
    :returns: R(N,2)-matrix of updated positions
    """
    N = len(X)
    N_R = len(R)
    D_shape = (N, N_R)
    if DEBUGGING:
        _check_ndim('X', X, 2)
        _check_ndim('R', R, 2)
        _check_shape('X', X, 2, axis=1)
        _check_shape('R', R, 2, axis=1)
        _check_shape('D_up', D_up, D_shape)
        _check_shape('D_cur', D_cur, D_shape)
        _check_shape('V', V, D_shape)

    # Broadcast field position matrix
    XX = empty((N,N_R,2))
    XX[:] = X[:,AX,:]

    # Find valid values (i.e., corresponding to non-zero weights)
    D_inf = D_up == inf
    norm = V * ~D_inf
    N = norm.sum(axis=1)
    valid = N.nonzero()[0]
    norm[valid] /= N[valid,AX]

    # Zero out the inf elements of the updated distance matrix and corresponding
    # elements in the current distance matrix
    D_up[D_inf] = D_cur[D_inf] = 0.0

    # Construct the agent-reward avoidant unit vectors
    DR = XX - R[AX]
    DR_norm = hypot(DR[...,0], DR[...,1])
    valid = DR_norm.nonzero()
    DR[valid] /= DR_norm[valid][:,AX]

    return (norm[...,AX]*(D_up - D_cur)[...,AX]*DR).sum(axis=1)
