"""Anchor-local linear residual models with explicit input availability."""
from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

LOG = logging.getLogger(__name__)


def lag_features(x, positions, offsets):
    """Return lag-major features; normalized zero padding resets each anchor."""
    x = np.asarray(x, dtype=np.float64)
    positions, offsets = np.asarray(positions), np.asarray(offsets)
    index = positions[:, None] + offsets[None, :]
    valid = (index >= 0) & (index < x.shape[1])
    values = x[:, np.clip(index, 0, x.shape[1]-1), :]
    return (values * valid[None, :, :, None]).reshape(len(x), len(positions), -1)


def sufficient_statistics(x, y, *, center, block_size, offsets, max_ar_order,
                          batch_anchors=96, device='cpu'):
    """Accumulate only supervised targets, retaining TRAIN output lags for ARX."""
    channels = x.shape[-1]
    width = len(offsets)*channels + max_ar_order*channels
    gram = torch.zeros((width, width), dtype=torch.float64, device=device)
    rhs = torch.zeros((width, channels), dtype=torch.float64, device=device)
    sx = torch.zeros(width, dtype=torch.float64, device=device)
    sy = torch.zeros(channels, dtype=torch.float64, device=device)
    positions = np.arange(center, center+block_size)
    for start in range(0, len(x), batch_anchors):
        stop = min(start+batch_anchors, len(x))
        design = lag_features(x[start:stop], positions, offsets)
        if max_ar_order:
            previous = lag_features(y[start:stop], positions, -np.arange(1, max_ar_order+1))
            design = np.concatenate([design, previous], axis=-1)
        tx = torch.as_tensor(design.reshape(-1, width), dtype=torch.float64, device=device)
        ty = torch.as_tensor(y[start:stop, center:center+block_size].reshape(-1, channels),
                             dtype=torch.float64, device=device)
        gram.addmm_(tx.T, tx)
        rhs.addmm_(tx.T, ty)
        sx += tx.sum(0)
        sy += ty.sum(0)
        if start == 0 or stop == len(x) or start % (batch_anchors*20) == 0:
            LOG.info('FIT sufficient statistics anchors %d/%d, features=%d', stop, len(x), width)
    return dict(gram=gram.cpu().numpy(), rhs=rhs.cpu().numpy(), sum_x=sx.cpu().numpy(),
                sum_y=sy.cpu().numpy(), count=len(x)*block_size)


def add_delta_statistics(stats, offsets, channels):
    """Append the historical LOW-current minus LOW-previous linear feature."""
    offsets = list(offsets)
    a = offsets.index(0)*channels + np.arange(channels)
    b = offsets.index(-1)*channels + np.arange(channels)
    g = stats['gram']
    cross = g[:, a] - g[:, b]
    delta = g[np.ix_(a, a)] - g[np.ix_(a, b)] - g[np.ix_(b, a)] + g[np.ix_(b, b)]
    return dict(gram=np.block([[g, cross], [cross.T, delta]]),
                rhs=np.concatenate([stats['rhs'], stats['rhs'][a]-stats['rhs'][b]]),
                sum_x=np.concatenate([stats['sum_x'], stats['sum_x'][a]-stats['sum_x'][b]]),
                sum_y=stats['sum_y'], count=stats['count'])


def fit_ridge(stats, indices, alphas):
    """Solve centered normal equations with an unpenalized intercept."""
    indices = np.asarray(indices, dtype=int)
    n = stats['count']
    mean_x = stats['sum_x'][indices]/n
    mean_y = stats['sum_y']/n
    gram = stats['gram'][np.ix_(indices, indices)] - n*np.outer(mean_x, mean_x)
    gram = (gram+gram.T)*.5
    rhs = stats['rhs'][indices] - n*np.outer(mean_x, mean_y)
    results = {}
    for alpha in alphas:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('Ridge alpha must be finite and positive')
        penalized = gram.copy()
        penalized.flat[::len(penalized)+1] += alpha
        weights = cho_solve(cho_factor(penalized, lower=True, check_finite=False), rhs,
                            check_finite=False)
        results[float(alpha)] = (weights, mean_y-mean_x@weights)
    return results


def spectral_radius(ar_weights):
    order, channels, _ = ar_weights.shape
    if not order:
        return 0.0
    companion = np.zeros((order*channels, order*channels))
    companion[:channels] = np.concatenate([w.T for w in ar_weights], axis=1)
    if order > 1:
        companion[channels:, :-channels] = np.eye((order-1)*channels)
    return float(np.abs(np.linalg.eigvals(companion)).max())


def predict(state, x, *, center, block_size, batch_anchors=48):
    """Free-run on LOW alone. Past observed HIGH/residuals are never arguments."""
    x = np.asarray(x)
    if not np.isfinite(x).all():
        raise ValueError('Prediction input must be finite')
    order = len(state['ar_weights'])
    positions = np.arange(center+block_size) if order else np.arange(center, center+block_size)
    result = []
    for start in range(0, len(x), batch_anchors):
        batch = x[start:start+batch_anchors]
        design = lag_features(batch, positions, state['offsets'])
        if state['delta']:
            current = lag_features(batch, positions, np.array([0]))
            previous = lag_features(batch, positions, np.array([-1]))
            design = np.concatenate([design, current-previous], axis=-1)
        feedforward = (design.reshape(-1, design.shape[-1])@state['weights'] +
                       state['bias']).reshape(len(batch), len(positions), -1)
        if order:
            for t in range(len(positions)):
                for k in range(1, min(order, t)+1):
                    feedforward[:, t] += feedforward[:, t-k]@state['ar_weights'][k-1]
            feedforward = feedforward[:, center:center+block_size]
        if not np.isfinite(feedforward).all():
            raise ValueError('Free-running predictions must be finite')
        result.append(feedforward)
    return np.concatenate(result)


def make_state(weights, bias, offsets, channels, *, delta=False, ar_order=0):
    input_width = len(offsets)*channels + int(delta)*channels
    ar = weights[input_width:].reshape(ar_order, channels, channels)
    return dict(weights=weights[:input_width], bias=bias, offsets=np.asarray(offsets, dtype=int),
                delta=bool(delta), ar_weights=ar)


def save_state(path, state):
    np.savez_compressed(path, **state)


def load_state(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(weights=data['weights'], bias=data['bias'], offsets=data['offsets'],
                    delta=bool(data['delta']), ar_weights=data['ar_weights'])
