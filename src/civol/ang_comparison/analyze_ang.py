"""Apples-to-apples comparison with Ang, Hodrick, Xing and Zhang (2006).

Stage 'anchor': Ang IVOL (daily FF3 residual standard deviation over the formation calendar
month, >= 17 days) for every eligible stock at every month end from the cache, next-month
excess returns, value- and equal-weighted quintile sorts, FF3 alphas (full universe).
Stage 'compare': the same on the measurement universe, then cIVOL at the same month ends,
the top-quintile membership decomposition and Fama-MacBeth regressions.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from civol.core.paths import records_path, output_path
import pandas as pd

HERE = Path(__file__).resolve().parent
from civol.core.universe import Cache

RECORDS = records_path('ang_comparison')
OUT = output_path('ang_comparison', 'results')
MIN_DAYS = 17


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def newey_west(x, lag=6):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 10:
        return np.nan, np.nan, n
    m = x.mean(); u = x - m; s = np.sum(u * u) / n
    for L in range(1, lag + 1):
        s += 2 * (1 - L / (lag + 1)) * np.sum(u[L:] * u[:-L]) / n
    return float(m), float(m / np.sqrt(s / n)) if s > 0 else np.nan, n


def ts_alpha(y, X, lag=6):
    y = np.asarray(y, float); X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    ok = np.isfinite(y) & np.isfinite(X).all(1); y = y[ok]; X = X[ok]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X); S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lag + 1):
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None]); S += (1 - L / (lag + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return float(beta[0]), float(beta[0] / np.sqrt(V[0, 0])), int(len(y))


def month_blocks(dates):
    d = pd.DatetimeIndex(dates)
    key = d.year * 100 + d.month
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:] - 1, len(d) - 1]
    return [(int(k), int(a), int(b)) for k, a, b in zip(key[starts], starts, ends)]


def ang_ivol_month(cache, a, b):
    """Ang IVOL for all permnos with >= MIN_DAYS valid days in sessions a..b; also MAX, past-month return, extreme-day count."""
    logr = np.asarray(cache.ret[a:b + 1], dtype=np.float64)                 # days x permnos (NaN invalid)
    valid = np.isfinite(logr)
    n_valid = valid.sum(0)
    cols = np.flatnonzero(n_valid >= MIN_DAYS)
    if len(cols) == 0:
        return cols, np.array([]), np.array([]), np.array([]), np.array([])
    ff = np.asarray(cache.ff[a:b + 1], dtype=np.float64)
    simple = np.expm1(logr[:, cols]); vmask = valid[:, cols]
    excess = simple - ff[:, 3][:, None]
    X = np.column_stack([np.ones(len(ff)), ff[:, :3]])
    ivol = np.empty(len(cols)); mx = np.empty(len(cols)); past = np.empty(len(cols)); extreme = np.empty(len(cols))
    full = vmask.all(0)
    if full.any():
        y = excess[:, full]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ beta
        ivol[full] = np.sqrt(np.sum(res * res, 0) / (len(ff) - 4))
    for j in np.flatnonzero(~full):
        m = vmask[:, j]; y = excess[m, j]; Xm = X[m]
        beta, *_ = np.linalg.lstsq(Xm, y, rcond=None); res = y - Xm @ beta
        ivol[j] = np.sqrt(np.sum(res * res) / (m.sum() - 4))
    s0 = np.where(vmask, simple, np.nan)
    mx[:] = np.nanmax(s0, axis=0)
    past[:] = np.exp(np.nansum(np.where(vmask, logr[:, cols], 0.), axis=0)) - 1
    sd = np.nanstd(s0, axis=0)
    extreme[:] = np.nansum(np.abs(s0 - np.nanmean(s0, axis=0)) > 3 * sd, axis=0)
    return cols, ivol, mx, past, extreme


def next_month_return(cache, a, b):
    """Compounded simple return and excess return over sessions a..b; complete flag."""
    logr = np.asarray(cache.ret[a:b + 1], dtype=np.float64); valid = np.isfinite(logr)
    ff = np.asarray(cache.ff[a:b + 1], dtype=np.float64)
    ret = np.exp(np.nansum(np.where(valid, logr, 0.), axis=0)) - 1
    rf = np.prod(1 + ff[:, 3]) - 1
    return ret, ret - rf, valid.all(0), valid.any(0), dict(mkt=np.prod(1 + ff[:, 0] + ff[:, 3]) - np.prod(1 + ff[:, 3]), smb=ff[:, 1].sum(), hml=ff[:, 2].sum(), rf=rf)


def build_anchor(cache, start='1988-01', end='2025-11'):
    blocks = month_blocks(cache.dates)
    rows = []; factors = []
    for i in range(len(blocks) - 1):
        key, a, b = blocks[i]; nkey, na, nb = blocks[i + 1]
        ym = f'{key // 100}-{key % 100:02d}'
        if ym < start or ym > end:
            continue
        elig = np.asarray(cache.elig[b])
        cols, ivol, mx, past, extreme = ang_ivol_month(cache, a, b)
        keep = elig[cols]
        cols, ivol, mx, past, extreme = cols[keep], ivol[keep], mx[keep], past[keep], extreme[keep]
        ret, exret, complete, anyv, fac = next_month_return(cache, na, nb)
        cap = np.asarray(cache.mktcap[b], dtype=np.float64)[cols]
        pc = np.asarray(cache.permco[b])[cols]
        f = pd.DataFrame(dict(month=ym, t_idx=b, permno=cache.permnos[cols], permco=pc, mktcap=cap, ang_ivol=ivol, max_ret=mx, past_ret=past,
                              extreme_days=extreme, ret_next=ret[cols], exret_next=exret[cols], complete_next=complete[cols], any_next=anyv[cols]))
        f = f[np.isfinite(f.mktcap) & (f.mktcap > 0) & (f.ang_ivol > 0)]
        # one security per PERMCO (largest capitalization), as in the measurement universe
        f = f.sort_values(['permco', 'mktcap'], ascending=[True, False]).drop_duplicates('permco')
        rows.append(f); factors.append(dict(month=ym, **fac))
    panel = pd.concat(rows, ignore_index=True)
    return panel, pd.DataFrame(factors).set_index('month')


def sorts(panel, measure, weight, factors, by='month'):
    rows = []
    for m, g in panel.groupby(by, sort=True):
        q = pd.qcut(g[measure].rank(method='first'), 5, labels=False)
        rec = dict(month=m)
        for k in range(5):
            gg = g[q == k]; w = gg.mktcap if weight == 'vw' else pd.Series(1., index=gg.index)
            rec[f'q{k + 1}'] = float(np.average(gg.exret_next, weights=w))
        rec['spread'] = rec['q5'] - rec['q1']; rows.append(rec)
    tab = pd.DataFrame(rows).set_index('month')
    out = {c: dict(zip(('mean', 't'), newey_west(tab[c])[:2])) for c in list(tab.columns)}
    a, t, n = ts_alpha(tab['spread'], factors.loc[tab.index, ['mkt', 'smb', 'hml']])
    out['alpha'] = dict(alpha=a, t=t, months=n)
    return tab, out


def fama_macbeth(panel, regs):
    rows = []
    for m, g in panel.groupby('month', sort=True):
        X = np.column_stack([np.ones(len(g))] + [g[r].to_numpy(float) for r in regs]); y = g.exret_next.to_numpy(float)
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        if ok.sum() < len(regs) + 10:
            continue
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        rows.append(dict(month=m, **{r: beta[i + 1] for i, r in enumerate(regs)}))
    fm = pd.DataFrame(rows).set_index('month')
    return {r: dict(zip(('mean', 't'), newey_west(fm[r])[:2])) for r in regs}, int(len(fm))


def fmt_sort(label, out):
    return f"{label:44s} q1..q5 = " + ' '.join(f"{out[f'q{k}']['mean'] * 100:6.3f}" for k in range(1, 6)) + \
        f" | 5-1 = {out['spread']['mean'] * 100:6.3f}% (t {out['spread']['t']:5.2f}) | alpha {out['alpha']['alpha'] * 100:6.3f}% (t {out['alpha']['t']:5.2f}) | months {out['alpha']['months']}"


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['anchor', 'compare']); a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    (OUT / f'RUN_MANIFEST_{a.stage}.json').write_text(json.dumps(dict(
        utc=datetime.now(timezone.utc).isoformat(), analysis_sha256=sha(__file__),
        runner_sha256=sha(HERE / 'run_monthend.py'), stage=a.stage,
        statement='Source hash and timestamp for this analysis run.'), indent=1))
    cache = Cache()
    anchor_path = OUT / 'anchor_panel.parquet'
    if anchor_path.exists():
        panel = pd.read_parquet(anchor_path); factors = pd.read_csv(OUT / 'monthly_factors.csv', index_col=0)
    else:
        panel, factors = build_anchor(cache); panel.to_parquet(anchor_path, index=False); factors.to_csv(OUT / 'monthly_factors.csv')
    results = {}
    full = panel[panel.complete_next]
    results['anchor'] = dict(months=int(full.month.nunique()), stock_months=int(len(full)), stock_months_incomplete_next=int((~panel.complete_next).sum()),
                             mean_stocks_per_month=float(full.groupby('month').size().mean()))
    for w in ('vw', 'ew'):
        tab, out = sorts(full, 'ang_ivol', w, factors); tab.to_csv(OUT / f'anchor_sort_ang_{w}.csv'); results['anchor'][f'ang_{w}'] = out
        print(fmt_sort(f'Ang IVOL, full universe, {w}', out))
    cut = full.groupby('month').mktcap.transform(lambda v: v.quantile(.2)); exm = full[full.mktcap >= cut]
    for w in ('vw', 'ew'):
        tab, out = sorts(exm, 'ang_ivol', w, factors); results['anchor'][f'ang_{w}_exmicro'] = out
        print(fmt_sort(f'Ang IVOL, full universe ex-microcap, {w}', out))
    # subperiods
    for lo, hi in (('1988-01', '2000-12'), ('2001-01', '2025-11')):
        sub = full[(full.month >= lo) & (full.month <= hi)]
        tab, out = sorts(sub, 'ang_ivol', 'vw', factors); results['anchor'][f'ang_vw_{lo[:4]}_{hi[:4]}'] = out
        print(fmt_sort(f'Ang IVOL, full universe, vw, {lo[:4]}-{hi[:4]}', out))
    if a.stage == 'compare':
        frames = []
        for js in sorted(RECORDS.glob('*.json')):
            if js.name.endswith('.FAILED.json') or js.name.startswith('RUN'):
                continue
            s = json.loads(js.read_text()); z = np.load(js.with_suffix('.npz'))
            ym = s['date'][:7]
            frames.append(pd.DataFrame(dict(month=ym, permno=z['permno'], civol_k10=np.sqrt(z['civolk10_q'] / 21), civol_bn=np.sqrt(z['civol_q'] / 21),
                                            civol_k10_ipv=z['civolk10_ipv'], ff3_1m_record=np.sqrt(np.maximum(z['ff3_1m_simple'], 0) / 21))))
        civ = pd.concat(frames, ignore_index=True)
        merged = full.merge(civ, on=['month', 'permno'], how='inner')
        results['compare'] = dict(months=int(merged.month.nunique()), stock_months=int(len(merged)), mean_stocks_per_month=float(merged.groupby('month').size().mean()),
                                  spearman_ang_civol_median=float(np.median([g.ang_ivol.corr(g.civol_k10, method='spearman') for _, g in merged.groupby('month')])),
                                  record_ff3_vs_ang_ivol_median_ratio=float((merged.ff3_1m_record / merged.ang_ivol).median()))
        for meas, label in (('ang_ivol', 'Ang IVOL'), ('civol_k10', 'cIVOL K=10'), ('civol_bn', 'cIVOL Bai-Ng')):
            for w in ('vw', 'ew'):
                tab, out = sorts(merged, meas, w, factors); tab.to_csv(OUT / f'compare_sort_{meas}_{w}.csv'); results['compare'][f'{meas}_{w}'] = out
                print(fmt_sort(f'{label}, measurement universe, {w}', out))
        cutm = merged.groupby('month').mktcap.transform(lambda v: v.quantile(.2)); mex = merged[merged.mktcap >= cutm]
        for meas, label in (('ang_ivol', 'Ang IVOL'), ('civol_k10', 'cIVOL K=10')):
            for w in ('vw', 'ew'):
                tab, out = sorts(mex, meas, w, factors); results['compare'][f'{meas}_{w}_exmicro'] = out
                print(fmt_sort(f'{label}, measurement universe ex-microcap, {w}', out))
        # decomposition of top and bottom quintile membership
        merged = merged.copy()
        merged['q_ang'] = merged.groupby('month').ang_ivol.transform(lambda v: pd.qcut(v.rank(method='first'), 5, labels=False))
        merged['q_civ'] = merged.groupby('month').civol_k10.transform(lambda v: pd.qcut(v.rank(method='first'), 5, labels=False))
        merged['ratio'] = merged.ang_ivol / merged.civol_k10
        decomp = {}
        for side, qv in (('top', 4), ('bottom', 0)):
            groups = {'both': (merged.q_ang == qv) & (merged.q_civ == qv), 'ang_only': (merged.q_ang == qv) & (merged.q_civ != qv),
                      'civol_only': (merged.q_ang != qv) & (merged.q_civ == qv)}
            decomp[side] = {}
            for name, mask in groups.items():
                g = merged[mask]
                rows = []
                for m, gg in g.groupby('month', sort=True):
                    rows.append(dict(month=m, vw=float(np.average(gg.exret_next, weights=gg.mktcap)), ew=float(gg.exret_next.mean()), n=len(gg)))
                tab = pd.DataFrame(rows).set_index('month'); tab.to_csv(OUT / f'decomp_{side}_{name}.csv')
                vw_m, vw_t, _ = newey_west(tab.vw); ew_m, ew_t, _ = newey_west(tab.ew)
                al, alt, _ = ts_alpha(tab.vw, factors.loc[tab.index, ['mkt', 'smb', 'hml']])
                decomp[side][name] = dict(share_of_universe=float(mask.mean()), mean_n=float(tab.n.mean()), vw_exret=vw_m, vw_t=vw_t, ew_exret=ew_m, ew_t=ew_t,
                                          vw_ff3_alpha=al, vw_alpha_t=alt,
                                          median_mktcap_musd=float(g.mktcap.median() / 1000), median_past_ret=float(g.past_ret.median()), median_max_ret=float(g.max_ret.median()),
                                          mean_extreme_days=float(g.extreme_days.mean()), median_ang_over_civol=float(g.ratio.median()),
                                          median_ang_ivol=float(g.ang_ivol.median()), median_civol=float(g.civol_k10.median()))
                print(f"{side} {name:10s}: share {mask.mean():.3f}, vw exret {vw_m * 100:6.3f}% (t {vw_t:5.2f}), alpha {al * 100:6.3f}% (t {alt:5.2f}), "
                      f"ew {ew_m * 100:6.3f}% (t {ew_t:5.2f}); cap ${g.mktcap.median() / 1000:.0f}m, past {g.past_ret.median() * 100:.1f}%, MAX {g.max_ret.median() * 100:.1f}%, "
                      f"extreme days {g.extreme_days.mean():.2f}, Ang/cIVOL {g.ratio.median():.2f}")
            # Ang-only minus civol-only spread
            ta = pd.read_csv(OUT / f'decomp_{side}_ang_only.csv', index_col=0); tc = pd.read_csv(OUT / f'decomp_{side}_civol_only.csv', index_col=0)
            common = ta.index.intersection(tc.index); d = ta.loc[common, 'vw'] - tc.loc[common, 'vw']
            m_, t_, _ = newey_west(d); al, alt, _ = ts_alpha(d, factors.loc[common, ['mkt', 'smb', 'hml']])
            decomp[side]['ang_only_minus_civol_only'] = dict(vw=m_, t=t_, alpha=al, alpha_t=alt)
            print(f"{side}: Ang-only minus cIVOL-only, vw {m_ * 100:6.3f}% (t {t_:5.2f}), alpha {al * 100:6.3f}% (t {alt:5.2f})")
        results['decomposition'] = decomp
        merged['log_ang'] = np.log(merged.ang_ivol); merged['log_civ'] = np.log(merged.civol_k10); merged['surprise'] = merged.log_ang - merged.log_civ
        merged['log_cap'] = np.log(merged.mktcap)
        fm = {}
        for name, regs in {'ang': ['log_ang', 'log_cap', 'past_ret'], 'civol': ['log_civ', 'log_cap', 'past_ret'], 'joint': ['log_ang', 'log_civ', 'log_cap', 'past_ret'],
                           'expected_surprise': ['log_civ', 'surprise', 'log_cap', 'past_ret'], 'ang_no_controls': ['log_ang'], 'civol_no_controls': ['log_civ'],
                           'expected_surprise_no_controls': ['log_civ', 'surprise']}.items():
            fm[name], months = fama_macbeth(merged, regs); fm[name]['months'] = months
            print('FM', name, {k: (round(v['mean'], 4), round(v['t'], 2)) for k, v in fm[name].items() if k != 'months'})
        results['fama_macbeth'] = fm
        merged.to_parquet(OUT / 'compare_panel.parquet', index=False)
    results['seconds'] = round(time.perf_counter() - started, 1)
    (OUT / f'SUMMARY_{a.stage}.json').write_text(json.dumps(results, indent=1, default=float))
    print('seconds', results['seconds'])


if __name__ == '__main__':
    main()
