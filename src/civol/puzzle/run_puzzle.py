"""IVOL-return relation under the standard measure and cIVOL.

Builds the issuer-date panel from the production and rank-rule records
plus the CRSP cache (21-session compounded outcome returns, formation controls),
records the current analysis source hash, then runs the sorts, double sorts
and Fama-MacBeth regressions and writes the results directory.
"""
from __future__ import annotations
import hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from civol.core.paths import records_path, output_path
import pandas as pd

from civol.core.paths import RECORDS
from civol.core.universe import Cache

RANK_RECORDS = records_path('rank_rule')
OUT = output_path('puzzle', 'results')
H = 21
MEASURES = {'ff3_1m': 'FF3-1m (standard)', 'civolA': 'cIVOL rule A', 'civolk10': 'cIVOL K=10', 'civol': 'cIVOL Bai-Ng'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def newey_west(x, lag=3):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 5:
        return np.nan, np.nan, n
    m = x.mean(); u = x - m
    s = np.sum(u * u) / n
    for L in range(1, lag + 1):
        s += 2 * (1 - L / (lag + 1)) * np.sum(u[L:] * u[:-L]) / n
    se = np.sqrt(s / n)
    return float(m), float(m / se) if se > 0 else np.nan, n


def ts_alpha(y, X, lag=3):
    """Time-series regression of y on factors X with intercept; Newey-West t for alpha."""
    y = np.asarray(y, float); X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    ok = np.isfinite(y) & np.isfinite(X).all(1); y = y[ok]; X = X[ok]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ beta; n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return float(beta[0]), float(beta[0] / np.sqrt(V[0, 0])), beta[1:].tolist(), n


def build_panel(cache):
    frames = []; factor_rows = []
    for js in sorted(RECORDS.glob('*.json')):
        if js.name.endswith('.FAILED.json') or js.name.startswith('RUN'):
            continue
        s = json.loads(js.read_text())
        if not s.get('outcome_available'):
            continue
        z = np.load(js.with_suffix('.npz')); t = int(s['t_idx'])
        rk = RANK_RECORDS / js.name
        if not rk.exists():
            raise FileNotFoundError(rk)
        zr = np.load(rk.with_suffix('.npz'))
        if not np.array_equal(zr['permno'], z['permno']):
            raise ValueError('issuer sets differ: ' + js.name)
        cols = np.searchsorted(cache.permnos, z['permno'])
        out = np.asarray(cache.ret[t + 1:t + 1 + H][:, cols], dtype=np.float64)
        form = np.asarray(cache.ret[t - 20:t + 1][:, cols], dtype=np.float64)
        ff = np.asarray(cache.ff[t + 1:t + 1 + H], dtype=np.float64)
        rf21 = np.prod(1 + ff[:, 3]) - 1
        ret21 = np.expm1(np.nansum(out, axis=0))
        past21 = np.expm1(np.nansum(form, axis=0))
        nz_form = np.sum(np.isfinite(form) & (np.abs(form) > 0), axis=0)
        scored = z['scored']
        f = pd.DataFrame(dict(date=pd.Timestamp(s['date']), permno=z['permno'], mktcap=np.asarray(z['mktcap'], float), scored=scored,
                              ret21=ret21, exret21=ret21 - rf21, past21=past21, nonzero_formation=nz_form,
                              ff3_1m=np.sqrt(np.maximum(z['ff3_1m_simple'], 0)), civolA=np.sqrt(zr['civolA_q']),
                              civolk10=np.sqrt(z['civolk10_q']), civol=np.sqrt(z['civol_q'])))
        frames.append(f[scored])
        factor_rows.append(dict(date=pd.Timestamp(s['date']), mkt=np.prod(1 + ff[:, 0] + ff[:, 3]) - np.prod(1 + ff[:, 3]), smb=ff[:, 1].sum(), hml=ff[:, 2].sum(), rf=rf21))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(factor_rows).set_index('date')


def quintile_sorts(panel, measure, weight):
    rows = []
    for date, g in panel.groupby('date', sort=True):
        q = pd.qcut(g[measure].rank(method='first'), 5, labels=False)
        rec = dict(date=date)
        for k in range(5):
            gg = g[q == k]
            w = gg.mktcap if weight == 'vw' else pd.Series(1., index=gg.index)
            rec[f'q{k + 1}'] = float(np.average(gg.exret21, weights=w))
        rec['spread'] = rec['q5'] - rec['q1']
        rows.append(rec)
    return pd.DataFrame(rows).set_index('date')


def double_sort(panel, first, second, weight):
    rows = []
    for date, g in panel.groupby('date', sort=True):
        q1 = pd.qcut(g[first].rank(method='first'), 5, labels=False)
        spreads = []
        for k in range(5):
            gg = g[q1 == k]
            q2 = pd.qcut(gg[second].rank(method='first'), 5, labels=False)
            w = gg.mktcap if weight == 'vw' else pd.Series(1., index=gg.index)
            top = gg[q2 == 4]; bot = gg[q2 == 0]
            spreads.append(np.average(top.exret21, weights=w[top.index]) - np.average(bot.exret21, weights=w[bot.index]))
        rows.append(dict(date=date, avg_spread=float(np.mean(spreads)), **{f'within_q{k + 1}': spreads[k] for k in range(5)}))
    return pd.DataFrame(rows).set_index('date')


def fama_macbeth(panel, regressors):
    rows = []
    for date, g in panel.groupby('date', sort=True):
        X = np.column_stack([np.ones(len(g))] + [g[r].to_numpy(float) for r in regressors]); y = g.exret21.to_numpy(float)
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        rows.append(dict(date=date, n=int(ok.sum()), **{r: beta[i + 1] for i, r in enumerate(regressors)}, intercept=beta[0]))
    return pd.DataFrame(rows).set_index('date')


def summarize_sorts(table, factors):
    out = {}
    for col in [f'q{k}' for k in range(1, 6)] + ['spread']:
        m, t, n = newey_west(table[col]); out[col] = dict(mean=m, t=t)
    a, ta, b, n = ts_alpha(table['spread'], factors.loc[table.index, ['mkt', 'smb', 'hml']])
    out['spread_ff3_alpha'] = dict(alpha=a, t=ta, betas=b, dates=n)
    return out


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'RUN_MANIFEST.json').write_text(json.dumps(dict(
        utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
        statement='Source hash and timestamp for this analysis run.'), indent=1))
    cache = Cache()
    panel, factors = build_panel(cache)
    panel = panel[panel.nonzero_formation >= 5].copy()
    panel = panel[(panel.ff3_1m > 0) & np.isfinite(panel.exret21)].copy()
    panel['log_mktcap'] = np.log(panel.mktcap)
    for m in MEASURES:
        panel[f'log_{m}'] = np.log(panel[m])
    panel['surprise_A'] = panel['log_ff3_1m'] - panel['log_civolA']        # realized one-month IVOL relative to its conditional expectation
    panel['surprise_K10'] = panel['log_ff3_1m'] - panel['log_civolk10']
    cut = panel.groupby('date').mktcap.transform(lambda v: v.quantile(.2))
    samples = {'all': panel, 'ex_microcap': panel[panel.mktcap >= cut]}
    results = dict(dates=int(panel.date.nunique()), issuer_dates=int(len(panel)), ex_microcap_issuer_dates=int(len(samples['ex_microcap'])),
                   first=str(panel.date.min().date()), last=str(panel.date.max().date()),
                   spearman_civolA_ff3_1m_median=float(np.median([g.civolA.corr(g.ff3_1m, method='spearman') for _, g in panel.groupby('date')])))
    for sname, sp in samples.items():
        block = dict(univariate={}, double={}, fama_macbeth={}, transition={})
        for m in MEASURES:
            for w in ('vw', 'ew'):
                tab = quintile_sorts(sp, m, w); tab.to_csv(OUT / f'sort_{sname}_{m}_{w}.csv')
                block['univariate'][f'{m}_{w}'] = summarize_sorts(tab, factors)
        for first, second in (('ff3_1m', 'civolA'), ('civolA', 'ff3_1m'), ('ff3_1m', 'civolk10'), ('civolk10', 'ff3_1m')):
            for w in ('vw', 'ew'):
                tab = double_sort(sp, first, second, w); tab.to_csv(OUT / f'double_{sname}_{first}_then_{second}_{w}.csv')
                m, t, n = newey_west(tab.avg_spread)
                block['double'][f'{first}_then_{second}_{w}'] = dict(avg_spread=m, t=t, within={f'q{k + 1}': newey_west(tab[f'within_q{k + 1}'])[:2] for k in range(5)})
        specs = {'ff3_1m': ['log_ff3_1m', 'log_mktcap', 'past21'], 'civolA': ['log_civolA', 'log_mktcap', 'past21'], 'civolk10': ['log_civolk10', 'log_mktcap', 'past21'],
                 'civol': ['log_civol', 'log_mktcap', 'past21'], 'joint_A': ['log_ff3_1m', 'log_civolA', 'log_mktcap', 'past21'], 'joint_K10': ['log_ff3_1m', 'log_civolk10', 'log_mktcap', 'past21'],
                 'ff3_1m_no_controls': ['log_ff3_1m'], 'civolA_no_controls': ['log_civolA'],
                 'expected_plus_surprise_A': ['log_civolA', 'surprise_A', 'log_mktcap', 'past21'],
                 'expected_plus_surprise_K10': ['log_civolk10', 'surprise_K10', 'log_mktcap', 'past21'],
                 'expected_plus_surprise_A_no_controls': ['log_civolA', 'surprise_A']}
        for name, regs in specs.items():
            fm = fama_macbeth(sp, regs); fm.to_csv(OUT / f'fm_{sname}_{name}.csv')
            block['fama_macbeth'][name] = {r: dict(mean=newey_west(fm[r])[0], t=newey_west(fm[r])[1]) for r in regs}
            block['fama_macbeth'][name]['dates'] = int(len(fm)); block['fama_macbeth'][name]['mean_n'] = float(fm.n.mean())
        # quintile persistence
        for m in MEASURES:
            q = sp.groupby('date')[m].transform(lambda v: pd.qcut(v.rank(method='first'), 5, labels=False))
            tmp = sp[['date', 'permno']].copy(); tmp['q'] = q.values
            dates = sorted(tmp.date.unique()); stay = []
            for a, b in zip(dates[:-1], dates[1:]):
                x = tmp[tmp.date == a].set_index('permno').q; y = tmp[tmp.date == b].set_index('permno').q
                common = x.index.intersection(y.index)
                if len(common) > 100:
                    stay.append(float((x.loc[common] == y.loc[common]).mean()))
            block['transition'][m] = dict(same_quintile_share_median=float(np.median(stay)))
        # measure levels within each other's quintiles
        qf = sp.groupby('date').ff3_1m.transform(lambda v: pd.qcut(v.rank(method='first'), 5, labels=False))
        block['civolA_within_ff3_quintiles'] = sp.groupby(qf).civolA.median().round(4).tolist()
        block['ff3_within_civolA_quintiles'] = sp.groupby(sp.groupby('date').civolA.transform(lambda v: pd.qcut(v.rank(method='first'), 5, labels=False))).ff3_1m.median().round(4).tolist()
        results[sname] = block
    results['seconds'] = round(time.perf_counter() - started, 1)
    (OUT / 'SUMMARY.json').write_text(json.dumps(results, indent=1, default=float))
    # console summary
    for sname in samples:
        b = results[sname]
        print(f'\n===== sample: {sname} =====')
        for m, label in MEASURES.items():
            for w in ('vw', 'ew'):
                u = b['univariate'][f'{m}_{w}']
                print(f"{label:18s} {w}: q1..q5 = " + ' '.join(f"{u[f'q{k}']['mean'] * 100:6.3f}" for k in range(1, 6)) +
                      f" | 5-1 = {u['spread']['mean'] * 100:6.3f}% (t {u['spread']['t']:5.2f}) | FF3 alpha {u['spread_ff3_alpha']['alpha'] * 100:6.3f}% (t {u['spread_ff3_alpha']['t']:5.2f})")
        for key, d in b['double'].items():
            print(f"double {key:28s}: avg 5-1 = {d['avg_spread'] * 100:6.3f}% (t {d['t']:5.2f})")
        for name, d in b['fama_macbeth'].items():
            print(f"FM {name:20s}: " + ', '.join(f"{r}={d[r]['mean']:.4f} (t {d[r]['t']:.2f})" for r in d if r not in ('dates', 'mean_n')))
        print('transition', b['transition'])
    print('seconds', results['seconds'])


if __name__ == '__main__':
    main()
