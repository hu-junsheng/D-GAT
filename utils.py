import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import pickle
import torch.nn.functional as F
import collections

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def sorted_choices(p,qs,k,p_sys,q_sys=None):
    if q_sys is None:
        q_sys = p_sys
    pc = p_sys[p]
    d = {}
    for q in qs:
        dist = np.sqrt(np.absolute(q_sys[q] - p_sys[p]).sum())
        if dist in d.keys():
            dist += 1e-2 * np.random.random()
        d[dist] = q
    n = 0
    out = []
    od = dict(sorted(d.items()))
    for key, val in od.items():
        if n == k:
            break
        out.append(val)
        n += 1
    return out

def logmag_phase(ir, n_fft=256):
    spec = torch.fft.rfft(ir, n_fft)
    mag = spec.abs().clamp(min=1e-5)
    phs = spec.angle()
    # phs = spec / mag
    # phs = torch.cat([phs.real.unsqueeze(-1),phs.imag.unsqueeze(-1)], dim=-1)
    logmag = 20 * mag.log10()
    return logmag, phs