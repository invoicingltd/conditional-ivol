"""Forecast encompassing and combination tests."""
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
OUT = output_path('encompassing', 'results')
H = 21
MIN_PRIOR_DATES = 24
FLOOR = 1e-9
FORECASTS = {'civolA': 'cIVOL rule A', 'civol': 'cIVOL Bai-Ng', 'civolk10': 'cIVOL K=10', 'pca_garch': 'PCA-GARCH', 'ff3_garch': 'FF3-GARCH',
             'har_ff3': 'HAR-FF3', 'pca_ewma': 'PCA-EWMA', 'ff3_ewma': 'FF3-EWMA', 'ff3_1m': 'FF3-1m', 'pca_1m': 'PCA-1m', 'ff3_12m': 'FF3-12m'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def newey_west(x, lag=3):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 5:
        return np.nan, np.nan, n
    m = x.mean(); u = x - m; s = np.sum(u * u) / n
    for L in range(1, lag + 1):
        s += 2 * (1 - L / (lag + 1)) * np.sum(u[L:] * u[:-L]) / n
    return float(m), float(m / np.sqrt(s / n)) if s > 0 else np.nan, n


def qlike(rv, f):
    r = rv / f
    return r - np.log(r) - 1


def build_panel(cache):
    frames = []
    for js in sorted(RECORDS.glob('*.json')):
        if js.name.endswith('.FAILED.json') or js.name.startswith('RUN'):
            continue
        s = json.loads(js.read_text())
        if not s.get('outcome_available'):
            continue
        z = np.load(js.with_suffix('.npz')); t = int(s['t_idx'])
        zr = np.load((RANK_RECORDS / js.name).with_suffix('.npz'))
        if not np.array_equal(zr['permno'], z['permno']):
            raise ValueError(js.name)
        cols = np.searchsorted(cache.permnos, z['permno'])
        form = np.asarray(cache.ret[t - 20:t + 1][:, cols], dtype=np.float64)
        out = np.asarray(cache.ret[t + 1:t + 1 + H][:, cols], dtype=np.float64)
        nz_form = np.sum(np.isfinite(form) & (np.abs(form) > 0), axis=0); nz_out = np.sum(np.isfinite(out) & (np.abs(out) > 0), axis=0)
        # HAR regressors: FF3 residual squares over the last 21 sessions with the record's formation betas
        ff = np.asarray(cache.ff[t - 20:t + 1], dtype=np.float64)
        excess = form - np.log1p(ff[:, 3])[:, None]
        resid = excess - z['issuer_ff3_alpha'][None, :] - ff[:, :3] @ z['issuer_ff3_betas'].T
        r2 = resid * resid
        f = pd.DataFrame(dict(date=pd.Timestamp(s['date']), k=len(frames), permno=z['permno'], mktcap=np.asarray(z['mktcap'], float), scored=z['scored'],
                              nonzero_formation=nz_form, nonzero_outcome=nz_out, rv_ff3=z['rv_ff3'], rv_broad=z['rv_broad'],
                              civolA=zr['civolA_ipv'], civol=z['civol_ipv'], civolk10=z['civolk10_ipv'], pca_garch=z['pca_garch'], ff3_garch=z['ff3_garch'],
                              pca_ewma=z['pca_ewma'], ff3_ewma=z['ff3_ewma'], ff3_1m=z['ff3_1m'], pca_1m=z['pca_1m'], ff3_12m=z['ff3_12m'],
                              rv1=H * r2[-1], rv5=H * r2[-5:].mean(0), rv21=H * r2.mean(0)))
        frames.append(f[z['scored']])
    return pd.concat(frames, ignore_index=True)


def pooled_log_fit(train, y, xcols):
    X = np.column_stack([np.ones(len(train))] + [np.log(np.maximum(train[c].to_numpy(float), FLOOR)) for c in xcols])
    Y = np.log(np.maximum(train[y].to_numpy(float), FLOOR))
    ok = np.isfinite(X).all(1) & np.isfinite(Y)
    # Solve normal equations for the reported final weights.
    # Expanding forecasts are fitted separately in expanding_forecast.
    Xo = X[ok]; Yo = Y[ok]
    if len(Yo) < 10 * X.shape[1]:
        raise FloatingPointError(f'pooled log fit on {xcols}: only {len(Yo)} rows')
    beta = _solve_normal_equations(Xo, Yo)
    smear = float(np.mean(np.exp(Yo - _predict(Xo, beta))))
    if not (np.all(np.isfinite(beta)) and np.isfinite(smear)):
        raise FloatingPointError(f'pooled log fit on {xcols} produced invalid coefficients')
    return beta, smear


def _solve_normal_equations(X, Y):
    gram = np.einsum('ij,ik->jk', X, X); rhs = np.einsum('ij,i->j', X, Y)
    return np.linalg.solve(gram, rhs)


def _predict(X, beta):
    return np.einsum('ij,j->i', X, beta)


def apply_log_fit(frame, beta, smear, xcols):
    X = np.column_stack([np.ones(len(frame))] + [np.log(np.maximum(frame[c].to_numpy(float), FLOOR)) for c in xcols])
    return np.exp(_predict(X, beta)) * smear


def expanding_forecast(panel, name, xcols, y='rv_ff3'):
    """Fit on all issuer-dates of prior formation dates (expanding), forecast the current date.

    Copy the inputs to NumPy arrays before iteration to keep date masks and
    output assignment stable across pandas operations.
    """
    kcol = panel['k'].to_numpy(dtype=np.int64, copy=True)
    X = np.column_stack([np.ones(len(panel))] + [np.log(np.maximum(panel[c].to_numpy(float), FLOOR)) for c in xcols])
    Y = np.log(np.maximum(panel[y].to_numpy(float), FLOOR))
    valid = np.isfinite(X).all(1) & np.isfinite(Y)
    out = np.full(len(panel), np.nan)
    started = False
    for k in np.unique(kcol):
        if k < MIN_PRIOR_DATES:
            continue
        ok = (kcol < k) & valid
        if ok.sum() < 10 * X.shape[1]:
            if started:
                raise FloatingPointError(f'{name}: training set at formation date index {k} has {ok.sum()} rows')
            continue            # inputs that are themselves expanding forecasts (HAR) exist only from MIN_PRIOR_DATES on
        started = True
        beta = _solve_normal_equations(X[ok], Y[ok])
        smear = float(np.mean(np.exp(Y[ok] - _predict(X[ok], beta))))
        cur = kcol == k
        f = np.exp(_predict(X[cur], beta)) * smear
        if not (np.all(np.isfinite(beta)) and np.isfinite(smear)) or np.mean(np.isfinite(f) & (f > 0)) < .5:
            raise FloatingPointError(f'{name}: invalid forecasts at formation date index {k}')
        out[cur] = f
    panel[name] = out
    return panel


def fm_log_regression(panel, y, xcols):
    rows = []
    for k, g in panel.groupby('k', sort=True):
        X = np.column_stack([np.ones(len(g))] + [np.log(np.maximum(g[c].to_numpy(float), FLOOR)) for c in xcols]); Y = np.log(np.maximum(g[y].to_numpy(float), FLOOR))
        ok = np.isfinite(X).all(1) & np.isfinite(Y)
        beta, *_ = np.linalg.lstsq(X[ok], Y[ok], rcond=None)
        res = Y[ok] - X[ok] @ beta
        rows.append(dict(k=k, r2=1 - res.var() / Y[ok].var(), **{c: beta[i + 1] for i, c in enumerate(xcols)}))
    fm = pd.DataFrame(rows).set_index('k')
    out = {c: dict(zip(('mean', 't'), newey_west(fm[c])[:2])) for c in xcols}
    out['r2'] = float(fm.r2.mean()); out['dates'] = int(len(fm))
    return out


def date_losses(panel, y, cols):
    rows = []
    for k, g in panel.groupby('k', sort=True):
        rv = g[y].to_numpy(float); rec = dict(k=k, date=g.date.iloc[0])
        for c in cols:
            f = g[c].to_numpy(float); ok = np.isfinite(f) & (f > 0) & np.isfinite(rv) & (rv > 0)
            l = qlike(rv[ok], f[ok]); rec[f'{c}_mean'] = float(l.mean()); rec[f'{c}_median'] = float(np.median(l))
            rec[f'{c}_mselog'] = float(np.mean((np.log(rv[ok]) - np.log(f[ok])) ** 2))
        rows.append(rec)
    return pd.DataFrame(rows).set_index('k')


def dm(dl, a, b, stat='mean'):
    d = (dl[f'{a}_{stat}'] - dl[f'{b}_{stat}']).to_numpy(); m, t, n = newey_west(d)
    return dict(mean_difference=m, t=t, dates=n, a_wins=float(np.mean(d < 0)))


def main():
    started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'RUN_MANIFEST.json').write_text(json.dumps(dict(
        utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
        statement='Source hash and timestamp for this analysis run.'), indent=1))
    cache = Cache()
    panel = build_panel(cache)
    panel = panel[(panel.nonzero_formation >= 5) & (panel.nonzero_outcome >= 5) & np.isfinite(panel.rv_ff3) & (panel.rv_ff3 > 0)].copy()
    # HAR-FF3 and the fitted combinations, expanding window on prior dates
    panel = expanding_forecast(panel, 'har_ff3', ['rv1', 'rv5', 'rv21'])
    combos = {'comb_civolA_garch_fit': ['civolA', 'pca_garch'], 'comb_civolA_har_fit': ['civolA', 'har_ff3'],
              'comb_civolA_ff3garch_fit': ['civolA', 'ff3_garch'], 'comb_all4_fit': ['civolA', 'pca_garch', 'har_ff3', 'pca_ewma']}
    for name, cols in combos.items():
        panel = expanding_forecast(panel, name, cols)
    for name, cols in {'comb_civolA_garch_eq': ['civolA', 'pca_garch'], 'comb_civolA_har_eq': ['civolA', 'har_ff3'], 'comb_all4_eq': ['civolA', 'pca_garch', 'har_ff3', 'pca_ewma']}.items():
        panel[name] = np.exp(np.mean([np.log(np.maximum(panel[c].to_numpy(float), FLOOR)) for c in cols], axis=0))
    panel = panel[panel.k >= MIN_PRIOR_DATES].copy()
    results = dict(dates=int(panel.k.nunique()), issuer_dates=int(len(panel)), first=str(panel.date.min().date()), last=str(panel.date.max().date()))
    # 1. encompassing regressions
    sets = {'civolA_alone': ['civolA'], 'pca_garch_alone': ['pca_garch'], 'ff3_garch_alone': ['ff3_garch'], 'har_alone': ['har_ff3'], 'pca_ewma_alone': ['pca_ewma'], 'ff3_1m_alone': ['ff3_1m'],
            'civolA+pca_garch': ['civolA', 'pca_garch'], 'civolA+ff3_garch': ['civolA', 'ff3_garch'], 'civolA+har': ['civolA', 'har_ff3'], 'civolA+pca_ewma': ['civolA', 'pca_ewma'],
            'civolA+ff3_1m': ['civolA', 'ff3_1m'], 'civolA+garch+har+ewma': ['civolA', 'pca_garch', 'har_ff3', 'pca_ewma'], 'civolk10+pca_garch': ['civolk10', 'pca_garch'],
            'civol+pca_garch': ['civol', 'pca_garch'], 'pca_garch+har': ['pca_garch', 'har_ff3']}
    results['encompassing'] = {y: {name: fm_log_regression(panel, y, cols) for name, cols in sets.items()} for y in ('rv_ff3', 'rv_broad')}
    # 2 and 3. losses, combinations, head-to-head
    cols = list(FORECASTS) + list(combos) + ['comb_civolA_garch_eq', 'comb_civolA_har_eq', 'comb_all4_eq']
    dl = date_losses(panel, 'rv_ff3', cols); dl.to_csv(OUT / 'date_losses_rv_ff3.csv')
    results['losses_rv_ff3'] = {c: dict(mean=float(dl[f'{c}_mean'].mean()), median=float(dl[f'{c}_median'].mean()), mselog=float(dl[f'{c}_mselog'].mean())) for c in cols}
    pairs = [('civolA', 'pca_garch'), ('civolA', 'ff3_garch'), ('civolA', 'har_ff3'), ('civolA', 'pca_ewma'), ('civolA', 'ff3_1m'), ('pca_garch', 'har_ff3'),
             ('comb_civolA_garch_fit', 'civolA'), ('comb_civolA_garch_fit', 'pca_garch'), ('comb_civolA_garch_eq', 'civolA'), ('comb_civolA_garch_eq', 'pca_garch'),
             ('comb_civolA_har_fit', 'civolA'), ('comb_civolA_har_fit', 'har_ff3'), ('comb_civolA_har_eq', 'civolA'), ('comb_civolA_har_eq', 'har_ff3'),
             ('comb_civolA_ff3garch_fit', 'civolA'), ('comb_civolA_ff3garch_fit', 'ff3_garch'),
             ('comb_all4_fit', 'civolA'), ('comb_all4_fit', 'pca_garch'), ('comb_all4_fit', 'har_ff3'), ('comb_all4_eq', 'civolA')]
    results['dm_rv_ff3'] = {f'{a}_minus_{b}': {stat: dm(dl, a, b, stat) for stat in ('mean', 'median', 'mselog')} for a, b in pairs}
    dl['decade'] = (pd.to_datetime(dl.date).dt.year // 10) * 10
    results['by_decade'] = []
    for dec, g in dl.groupby('decade'):
        rec = dict(decade=int(dec), dates=int(len(g)))
        for c in ('civolA', 'pca_garch', 'har_ff3', 'comb_civolA_garch_fit', 'comb_civolA_har_fit', 'comb_all4_fit'):
            rec[c] = float(g[f'{c}_mean'].mean())
        for a, b in (('civolA', 'har_ff3'), ('civolA', 'pca_garch'), ('comb_civolA_garch_fit', 'civolA'), ('comb_all4_fit', 'civolA')):
            d = (g[f'{a}_mean'] - g[f'{b}_mean']).to_numpy(); m, t, n = newey_west(d); rec[f'{a}_minus_{b}'] = m; rec[f't_{a}_minus_{b}'] = t
        results['by_decade'].append(rec)
    dlb = date_losses(panel, 'rv_broad', ['civolA', 'pca_garch', 'har_ff3', 'comb_civolA_garch_fit', 'comb_civolA_har_fit', 'comb_all4_fit'])
    results['losses_rv_broad'] = {c: float(dlb[f'{c}_mean'].mean()) for c in ('civolA', 'pca_garch', 'har_ff3', 'comb_civolA_garch_fit', 'comb_civolA_har_fit', 'comb_all4_fit')}
    results['dm_rv_broad'] = {f'{a}_minus_{b}': dm(dlb, a, b) for a, b in (('civolA', 'pca_garch'), ('civolA', 'har_ff3'), ('comb_civolA_garch_fit', 'civolA'), ('comb_all4_fit', 'civolA'))}
    # fitted weights at the last date, for reporting
    last = panel.k.max(); train = panel[panel.k < last]
    results['final_weights'] = {name: dict(zip(['intercept'] + cols, pooled_log_fit(train, 'rv_ff3', cols)[0].tolist())) for name, cols in combos.items()}
    results['final_weights']['har_ff3'] = dict(zip(['intercept', 'rv1', 'rv5', 'rv21'], pooled_log_fit(train, 'rv_ff3', ['rv1', 'rv5', 'rv21'])[0].tolist()))
    results['seconds'] = round(time.perf_counter() - started, 1)
    (OUT / 'SUMMARY.json').write_text(json.dumps(results, indent=1, default=float))
    # console
    print('dates', results['dates'], 'issuer-dates', results['issuer_dates'], results['first'], results['last'])
    print('\n== encompassing regressions (log RV-FF3 on log forecasts; Fama-MacBeth over dates, NW t) ==')
    for name, r in results['encompassing']['rv_ff3'].items():
        print(f"{name:26s} " + ', '.join(f"{c}={r[c]['mean']:.3f} (t {r[c]['t']:.1f})" for c in r if c not in ('r2', 'dates')) + f"  R2 {r['r2']:.3f}")
    print('\n== mean | median QLIKE | MSE log against RV-FF3 ==')
    for c, v in sorted(results['losses_rv_ff3'].items(), key=lambda kv: kv[1]['mean']):
        print(f"{c:28s} {v['mean']:.4f} {v['median']:.4f} {v['mselog']:.4f}")
    print('\n== DM (mean QLIKE): a minus b, negative favours a ==')
    for key, v in results['dm_rv_ff3'].items():
        print(f"{key:44s} {v['mean']['mean_difference']:+.4f} (t {v['mean']['t']:.2f}, a wins {v['mean']['a_wins']:.2f}) | median {v['median']['mean_difference']:+.4f} (t {v['median']['t']:.2f}) | mselog {v['mselog']['mean_difference']:+.4f} (t {v['mselog']['t']:.2f})")
    print('\n== by decade (mean QLIKE) =='); print(pd.DataFrame(results['by_decade']).round(4).to_string())
    print('\n== rv_broad ==', {k: round(v, 4) for k, v in results['losses_rv_broad'].items()}, {k: (round(v['mean_difference'], 4), round(v['t'], 2)) for k, v in results['dm_rv_broad'].items()})
    print('\n== final fitted weights ==', json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in results['final_weights'].items()}))
    print('seconds', results['seconds'])


if __name__ == '__main__':
    main()
