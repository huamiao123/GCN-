import numpy as np


def numpy_v2(h, w, bias, a_off, dtype=np.float64):
    h, w, a_off = (np.asarray(v, dtype=dtype) for v in (h, w, a_off))
    bias = np.asarray(bias, dtype=dtype)
    s = 1.0 / np.sqrt(a_off.sum(axis=1) + 1.0)
    hs = s[:, None] * h
    p = s[:, None] * ((a_off + np.eye(a_off.shape[0], dtype=dtype)) @ hs @ w) + bias
    return p, s, hs


def numpy_backward_v2(h, w, a_off, grad):
    p, s, hs = numpy_v2(h, w, np.zeros(w.shape[1]), a_off)
    del p
    grad_scaled = s[:, None] * grad
    ybar = (a_off.T + np.eye(a_off.shape[0])) @ grad_scaled
    return hs.T @ ybar, s[:, None] * (ybar @ w.T), ybar


def torch_v2(h, w, bias, a_off):
    import torch
    s = torch.rsqrt(a_off.sum(dim=1) + 1.0)
    hs = s[:, None] * h
    p = s[:, None] * ((a_off + torch.eye(a_off.shape[0], dtype=h.dtype, device=h.device)) @ hs @ w) + bias
    return p, s, hs
