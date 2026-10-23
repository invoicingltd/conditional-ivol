"""Formation-date eligibility on the cached CRSP arrays."""
from __future__ import annotations
import json
import numpy as np
from civol.core.paths import CACHE


class Cache:
    def __init__(self, folder=CACHE):
        folder = folder
        self.dates = np.load(folder / 'dates.npy')
        self.permnos = np.load(folder / 'permnos.npy')
        self.ret = np.load(folder / 'ret.npy', mmap_mode='r')
        self.valid = np.load(folder / 'valid.npy', mmap_mode='r')
        self.elig = np.load(folder / 'elig.npy', mmap_mode='r')
        self.mktcap = np.load(folder / 'mktcap.npy', mmap_mode='r')
        self.permco = np.load(folder / 'permco.npy', mmap_mode='r')
        self.ff = np.load(folder / 'ff.npy')        # sessions x 4: Mkt-RF, SMB, HML, RF
        self.manifest = json.loads((folder / 'MANIFEST.json').read_text())

    def __len__(self):
        return len(self.dates)


def eligible_universe(cache, t_idx, T):
    """Indices (into cache.permnos) of issuers eligible at session t_idx."""
    if t_idx - T + 1 < 0:
        raise ValueError('Insufficient history')
    rows = slice(t_idx - T + 1, t_idx + 1)
    elig_t = np.asarray(cache.elig[t_idx])
    valid_window = np.asarray(cache.valid[rows]).all(0)
    permco_window = np.asarray(cache.permco[rows])
    permco_t = permco_window[-1]
    permco_const = (permco_window == permco_t[None, :]).all(0) & (permco_t > 0)
    cap = np.asarray(cache.mktcap[t_idx], dtype=np.float64)
    candidates = elig_t & valid_window & permco_const & np.isfinite(cap) & (cap > 0)
    idx = np.flatnonzero(candidates)
    # one security per PERMCO: largest capitalization, ties to the lowest PERMNO
    order = np.lexsort((cache.permnos[idx], -cap[idx]))
    idx = idx[order]
    _, first = np.unique(permco_t[idx], return_index=True)
    idx = np.sort(idx[first])
    x = np.asarray(cache.ret[rows][:, idx], dtype=np.float64)
    var = x.var(0)
    keep = np.isfinite(x).all(0) & (var > 1e-14)
    idx = idx[keep]
    info = dict(flag_eligible=int(elig_t.sum()), complete_history=int((elig_t & valid_window).sum()),
                identity_constant=int((elig_t & valid_window & permco_const).sum()), with_capitalization=int(candidates.sum()),
                one_per_permco=int(len(first)), selected=int(len(idx)),
                eligible_cap_total=float(np.nansum(np.where(elig_t, cap, 0.))), selected_cap_total=float(cap[idx].sum()))
    return idx, info
