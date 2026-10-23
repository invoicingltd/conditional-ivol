"""Analyze the historical application records.

Reads only completed DATE.json/DATE.npz records. Produces loss tables,
Diebold-Mariano tests over dates, calibration regressions, diversifiability
summaries, descriptives and figures under work/application/results/.
"""
from __future__ import annotations
import json
import numpy as np
from civol.core.paths import output_path
import pandas as pd

from civol.core.paths import RECORDS
from civol.core.universe import Cache

SUBSAMPLES = {'nonstale': 5, 'liquid': 10}   # Data-quality subsamples: >= k non-zero daily returns in the last 21 formation sessions and in the 21 outcome sessions

OUT = output_path('application', 'results')
METHODS = ['civol_ipv', 'civolk10_ipv', 'civol1c_ipv', 'civol1cT_ipv', 'ff3paired_ipv', 'ff3_1m', 'ff3_12m', 'ff3_ewma', 'ff3_garch', 'pca_1m', 'pca_ewma', 'pca_garch',
           'pcak10_1m', 'pcak10_ewma', 'total_1m', 'civol_surrogate_ipv']
LABELS = dict(civol_ipv='cIVOL', civolk10_ipv='cIVOL-K10', civol1c_ipv='cIVOL one-component', civol1cT_ipv='cIVOL accepted law (total driver)', ff3paired_ipv='FF3-paired', ff3_1m='FF3-1m (standard)', ff3_12m='FF3-12m', ff3_ewma='FF3-EWMA',
              ff3_garch='FF3-GARCH', pca_1m='PCA-1m', pca_ewma='PCA-EWMA', pca_garch='PCA-GARCH', pcak10_1m='PCA(K10)-1m', pcak10_ewma='PCA(K10)-EWMA',
              total_1m='Total variance 1m', civol_surrogate_ipv='cIVOL surrogate')
OUTCOMES = ['rv_ff3', 'rv_broad', 'rv_ff3_fixed', 'rv_broad_k10']
PRIMARY_OUTCOME = 'rv_ff3'


def qlike(rv, f):
    r = rv / f
    return r - np.log(r) - 1


def newey_west_tstat(x, lag=3):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 5:
        return np.nan, np.nan, n
    m = x.mean(); u = x - m
    s = np.sum(u * u) / n
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        s += 2 * w * np.sum(u[L:] * u[:-L]) / n
    se = np.sqrt(s / n)
    return float(m), float(m / se) if se > 0 else np.nan, n


def nonzero_counts(cache, t_idx, permnos, H=21):
    """Number of non-zero valid daily returns in the last 21 formation sessions and
    in the 21 outcome sessions, per issuer (stale-price diagnostic)."""
    cols = np.searchsorted(cache.permnos, permnos)
    form = np.asarray(cache.ret[t_idx - 20:t_idx + 1][:, cols], dtype=np.float64)
    out = np.asarray(cache.ret[t_idx + 1:t_idx + 1 + H][:, cols], dtype=np.float64)
    nz_form = np.sum(np.isfinite(form) & (np.abs(form) > 0), axis=0)
    nz_out = np.sum(np.isfinite(out) & (np.abs(out) > 0), axis=0)
    return nz_form, nz_out


def load_records():
    cache = Cache()
    dates = []; per_date = []; issuer_frames = []
    for js in sorted(RECORDS.glob('*.json')):
        if js.name.endswith('.FAILED.json') or js.name.startswith('RUN'):
            continue
        s = json.loads(js.read_text())
        if not s.get('outcome_available'):
            continue
        z = np.load(js.with_suffix('.npz'))
        scored = z['scored']
        n = int(scored.sum())
        nz_form, nz_out = nonzero_counts(cache, int(s['t_idx']), z['permno'])
        rec = dict(date=pd.Timestamp(s['date']), N=s['N'], scored=n, rank=s['rank'], sign_rank=s['rank_diagnostics']['sign_rank'],
                   icp2=s['rank_diagnostics']['icp2_rank'], er=s['rank_diagnostics']['eigenvalue_ratio_rank'],
                   explained_share=s['rank_diagnostics']['explained_share'],
                   alpha=s['pooled']['coefficients'][0], beta=s['pooled']['coefficients'][1], gamma=s['pooled']['coefficients'][2],
                   rho_m=s['pooled']['coefficients'][3], phi=s['pooled']['coefficients'][4],
                   objective_gain_component=s['pooled_one_component']['objective_gain_component'],
                   mp_edge=(1 + np.sqrt(s['N'] / s['T'])) ** 2,
                   alpha_ff3=s['pooled_ff3']['coefficients'][0], beta_ff3=s['pooled_ff3']['coefficients'][1], gamma_ff3=s['pooled_ff3']['coefficients'][2],
                   pooled_converged=s['pooled']['converged'], static_selected=s['pooled']['static_selected'],
                   median_rel_q_mcse=s['integration']['median_relative_q_mcse'], max_rel_q_mcse=s['integration']['max_relative_q_mcse'],
                   median_rel_ipv_mcse=s['integration']['median_relative_ipv_mcse'], compounding=s['integration']['median_compounding_factor'],
                   surrogate_ratio=s['integration']['median_surrogate_ratio'], backend=s['integration'].get('backend'),
                   div_ff3=s['outcomes']['diversifiability_ff3_fixed'], div_broad=s['outcomes']['diversifiability_broad'], div_total=s['outcomes']['diversifiability_total'],
                   insample_ew_pca=s['residual_content']['pca']['equal_weight_residual_variance_ratio'], insample_ew_ff3=s['residual_content']['ff3']['equal_weight_residual_variance_ratio'],
                   insample_top_pca=s['residual_content']['pca']['top_residual_correlation_eigenvalue'], insample_top_ff3=s['residual_content']['ff3']['top_residual_correlation_eigenvalue'],
                   eligible_cap_share=s['universe']['selected_cap_total'] / s['universe']['eligible_cap_total'], flag_eligible=s['universe']['flag_eligible'],
                   seconds=s['seconds'], pilot=s.get('pilot', False))
        for k, v in s['outcomes'].get('alternative_rank_diversifiability', {}).items():
            rec[f'div_broad_k{k}'] = v
        for k, v in s['rank_diagnostics'].get('alternative_rank_content', {}).items():
            rec[f'insample_ew_k{k}'] = v['equal_weight_residual_variance_ratio']
            rec[f'insample_top_k{k}'] = v['top_residual_correlation_eigenvalue']
        rec['insample_top_ksign'] = rec.get(f"insample_top_k{s['rank_diagnostics']['sign_rank']}", np.nan)
        rec['insample_ew_ksign'] = rec.get(f"insample_ew_k{s['rank_diagnostics']['sign_rank']}", np.nan)
        rec['div_broad_ksign'] = s['outcomes'].get('alternative_rank_diversifiability', {}).get(str(s['rank_diagnostics']['sign_rank']), np.nan)
        per_date.append(rec)
        cols = dict(permno=z['permno'], mktcap=z['mktcap'], scored=scored, civol_q=z['civol_q'], civol_q_mcse=z['civol_q_mcse'],
                    civol_total_variance=z['civol_total_variance'], ff3_1m_simple=z['ff3_1m_simple'] if 'ff3_1m_simple' in z.files else np.nan,
                    nonzero_formation=nz_form, nonzero_outcome=nz_out)
        for m in METHODS + OUTCOMES:
            cols[m] = z[m] if m in z.files else np.full(len(scored), np.nan)
        f = pd.DataFrame(cols); f['date'] = pd.Timestamp(s['date'])
        issuer_frames.append(f)
    return pd.DataFrame(per_date).sort_values('date').reset_index(drop=True), pd.concat(issuer_frames, ignore_index=True)


def date_losses(iss, outcome, loss='qlike', statistic='mean'):
    """Cross-sectional mean (or median) loss per date and method; invalid forecasts
    get the date-method maximum loss and are counted."""
    rows = []; invalid = {m: 0 for m in METHODS}
    for date, g in iss[iss.scored].groupby('date', sort=True):
        rv = g[outcome].to_numpy(float); ok_rv = np.isfinite(rv) & (rv > 0)
        rec = dict(date=date, n=int(ok_rv.sum()))
        for m in METHODS:
            f = g[m].to_numpy(float); ok = ok_rv & np.isfinite(f) & (f > 0)
            if loss == 'qlike':
                l = qlike(rv[ok], f[ok])
            else:
                l = (np.log(rv[ok]) - np.log(f[ok])) ** 2
            bad = int(ok_rv.sum() - ok.sum()); invalid[m] += bad
            if len(l):
                full = np.r_[l, np.full(bad, l.max())]
                if statistic == 'mean':
                    rec[m] = float(full.mean())
                elif statistic == 'median':
                    rec[m] = float(np.median(full))
                elif statistic == 'trimmed':          # 1% trimmed in each tail, descriptive robustness summary
                    lo, hi = np.percentile(full, [1, 99])
                    rec[m] = float(full[(full >= lo) & (full <= hi)].mean())
            else:
                rec[m] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows), invalid


def dm_table(dl, base='civol_ipv', lag=3):
    rows = []
    for m in METHODS:
        if m == base:
            continue
        d = (dl[base] - dl[m]).to_numpy()
        mean, t, n = newey_west_tstat(d, lag)
        rows.append(dict(comparator=m, label=LABELS[m], base_mean=float(dl[base].mean()), comparator_mean=float(dl[m].mean()), mean_difference=mean,
                         nw_tstat=t, dates=n, base_wins=int(np.sum(d < 0)), win_share=float(np.mean(d < 0))))
    return pd.DataFrame(rows)


def mincer_zarnowitz(iss, outcome):
    rows = []
    g = iss[iss.scored & np.isfinite(iss[outcome]) & (iss[outcome] > 0)].copy()
    y = np.log(g[outcome].to_numpy(float))
    for m in METHODS:
        f = g[m].to_numpy(float); ok = np.isfinite(f) & (f > 0)
        x = np.log(f[ok]); yy = y[ok]; d = g.date.to_numpy()[ok]
        # within-date demeaning (date fixed effects)
        frame = pd.DataFrame(dict(x=x, y=yy, d=d))
        xm = frame.groupby('d').x.transform('mean'); ym = frame.groupby('d').y.transform('mean')
        xd = (frame.x - xm).to_numpy(); yd = (frame.y - ym).to_numpy()
        slope = float(np.sum(xd * yd) / np.sum(xd * xd)) if np.sum(xd * xd) > 0 else np.nan
        r2 = float(1 - np.sum((yd - slope * xd) ** 2) / np.sum(yd * yd)) if np.sum(yd * yd) > 0 else np.nan
        pooled_slope, pooled_int = np.polyfit(x, yy, 1)
        rows.append(dict(method=m, label=LABELS[m], fe_slope=slope, fe_r2=r2, pooled_slope=float(pooled_slope), pooled_intercept=float(pooled_int),
                         mean_log_bias=float(np.mean(yy - x)), observations=int(ok.sum())))
    return pd.DataFrame(rows)


def size_terciles(iss, outcome):
    g = iss[iss.scored].copy()
    g['tercile'] = g.groupby('date').mktcap.transform(lambda v: pd.qcut(v.rank(method='first'), 3, labels=['small', 'mid', 'large']))
    rows = []
    for terc, gg in g.groupby('tercile', observed=True):
        dl, _ = date_losses(gg, outcome)
        for m in METHODS:
            rows.append(dict(tercile=terc, method=m, label=LABELS[m], mean_qlike=float(dl[m].mean())))
        d = (dl['civol_ipv'] - dl['ff3_1m']).to_numpy(); mean, t, n = newey_west_tstat(d)
        rows.append(dict(tercile=terc, method='civol_minus_ff3_1m', label='cIVOL minus FF3-1m', mean_qlike=mean, nw_tstat=t, dates=n))
    return pd.DataFrame(rows)


def descriptives(iss, per_date):
    g = iss[iss.scored].copy()
    g['civol'] = np.sqrt(g.civol_q); g['ff3ivol'] = np.sqrt(g.ff3_1m)
    rows = []
    for date, gg in g.groupby('date'):
        rows.append(dict(date=date, n=len(gg), civol_median=gg.civol.median(), civol_p10=gg.civol.quantile(.1), civol_p90=gg.civol.quantile(.9),
                         ff3_median=gg.ff3ivol.median(), ff3_p10=gg.ff3ivol.quantile(.1), ff3_p90=gg.ff3ivol.quantile(.9),
                         spearman_civol_ff3=gg.civol.corr(gg.ff3ivol, method='spearman'), spearman_civol_k10=np.sqrt(gg.civol_q).corr(np.sqrt(gg.civolk10_ipv), method='spearman'),
                         private_share_median=(gg.civol_q / gg.civol_total_variance).median(),
                         rel_q_mcse_median=(gg.civol_q_mcse / gg.civol_q).median()))
    d = pd.DataFrame(rows).sort_values('date')
    # one-period rank persistence (issuers present at consecutive dates)
    pers = []
    dates = sorted(g.date.unique())
    for a, b in zip(dates[:-1], dates[1:]):
        x = g[g.date == a].set_index('permno'); y = g[g.date == b].set_index('permno')
        common = x.index.intersection(y.index)
        if len(common) > 50:
            pers.append(dict(date=b, n=len(common), civol_rank_autocorr=x.loc[common, 'civol'].corr(y.loc[common, 'civol'], method='spearman'),
                             ff3_rank_autocorr=x.loc[common, 'ff3ivol'].corr(y.loc[common, 'ff3ivol'], method='spearman')))
    return d, pd.DataFrame(pers)


def figures(per_date, dl_primary, desc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(per_date.date, per_date['rank'], label='Bai-Ng rank (primary)'); ax[0].plot(per_date.date, per_date.sign_rank, label='sign-rank', alpha=.7)
    ax[0].set_ylabel('selected K'); ax[0].legend(); ax[0].set_title('Common-space rank over time')
    ax[1].plot(per_date.date, per_date.alpha, label='alpha'); ax[1].plot(per_date.date, per_date.beta, label='beta'); ax[1].plot(per_date.date, per_date.gamma, label='gamma')
    ax[1].set_ylabel('pooled coefficient'); ax[1].legend(); ax[1].set_title('Pooled private feedback coefficients')
    ax[2].semilogy(per_date.date, per_date.div_ff3, label='FF3 residuals'); ax[2].semilogy(per_date.date, per_date.div_broad, label='broad residuals (primary K)')
    if 'div_broad_k10' in per_date:
        ax[2].semilogy(per_date.date, per_date.div_broad_k10, label='broad residuals (K=10)', alpha=.7)
    ax[2].axhline(1, color='k', lw=.8); ax[2].set_ylabel('EW residual variance ratio'); ax[2].legend(); ax[2].set_title('Out-of-sample diversifiability (1 = uncorrelated residuals)')
    fig.tight_layout(); fig.savefig(OUT / 'time_series.png', dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    means = pd.Series(dl_primary).drop('civol_surrogate_ipv', errors='ignore').sort_values(ascending=False)
    colors = ['tab:red' if m == 'ff3_1m' else ('tab:blue' if m.startswith('civol') else 'tab:gray') for m in means.index]
    ax.barh([LABELS[m] for m in means.index], means.values, color=colors)
    ax.set_xlabel('mean QLIKE against RV-FF3, non-stale issuer-dates, 455 formation dates (lower is better)')
    ax.set_title('Forecast loss by method, 1988-2025')
    fig.tight_layout(); fig.savefig(OUT / 'loss_by_method.png', dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(desc.date, desc.civol_median, label='cIVOL median'); ax.plot(desc.date, desc.ff3_median, label='FF3-1m IVOL median')
    ax.fill_between(desc.date, desc.civol_p10, desc.civol_p90, alpha=.2); ax.set_ylabel('21-day IVOL'); ax.legend(); ax.set_title('Cross-sectional level and spread of the two measures')
    fig.tight_layout(); fig.savefig(OUT / 'levels.png', dpi=130); plt.close(fig)


def main():
    # Use NumPy-backed pandas arithmetic to avoid intermittent NumExpr errors
    # in fixed-effect calibration on the full panel.
    pd.set_option('compute.use_numexpr', False)
    OUT.mkdir(parents=True, exist_ok=True)
    per_date, iss = load_records()
    per_date.to_csv(OUT / 'per_date.csv', index=False)
    results = dict(dates=int(len(per_date)), first=str(per_date.date.min().date()), last=str(per_date.date.max().date()),
                   issuer_dates=int(len(iss)), scored_issuer_dates=int(iss.scored.sum()), mean_N=float(per_date.N.mean()),
                   mean_eligible_cap_share=float(per_date.eligible_cap_share.mean()), backend_counts=per_date.backend.value_counts().to_dict(),
                   pooled_converged_share=float(per_date.pooled_converged.mean()), static_selected_dates=int(per_date.static_selected.sum()))
    tables = {}
    for outcome in OUTCOMES:
        dl, invalid = date_losses(iss, outcome, 'qlike'); dl.to_csv(OUT / f'date_losses_qlike_{outcome}.csv', index=False)
        dm = dm_table(dl); dm.to_csv(OUT / f'dm_qlike_{outcome}.csv', index=False)
        tables[outcome] = dict(mean_qlike={m: float(dl[m].mean()) for m in METHODS}, invalid_forecasts=invalid, dm=dm.to_dict('records'))
        dl2, _ = date_losses(iss, outcome, 'mselog'); dm2 = dm_table(dl2); dm2.to_csv(OUT / f'dm_mselog_{outcome}.csv', index=False)
        tables[outcome]['mean_mselog'] = {m: float(dl2[m].mean()) for m in METHODS}
        tables[outcome]['dm_mselog'] = dm2.to_dict('records')
        dl3, _ = date_losses(iss, outcome, 'qlike', statistic='median'); dm3 = dm_table(dl3); dm3.to_csv(OUT / f'dm_median_qlike_{outcome}.csv', index=False)
        tables[outcome]['median_qlike'] = {m: float(dl3[m].mean()) for m in METHODS}
        tables[outcome]['dm_median_qlike'] = dm3.to_dict('records')
        dl4, _ = date_losses(iss, outcome, 'qlike', statistic='trimmed'); dm4 = dm_table(dl4); dm4.to_csv(OUT / f'dm_trimmed_qlike_{outcome}.csv', index=False)
        tables[outcome]['trimmed_qlike'] = {m: float(dl4[m].mean()) for m in METHODS}
        tables[outcome]['dm_trimmed_qlike'] = dm4.to_dict('records')
        if outcome == PRIMARY_OUTCOME:
            dl_primary = dl
            # by decade
            dl['decade'] = (dl.date.dt.year // 10) * 10
            dec = []
            for decade, g in dl.groupby('decade'):
                d = (g['civol_ipv'] - g['ff3_1m']).to_numpy(); mean, t, n = newey_west_tstat(d)
                dec.append(dict(decade=int(decade), dates=n, civol=float(g['civol_ipv'].mean()), ff3_1m=float(g['ff3_1m'].mean()), ff3_garch=float(g['ff3_garch'].mean()),
                                pca_garch=float(g['pca_garch'].mean()), civolk10=float(g['civolk10_ipv'].mean()), diff_civol_ff3_1m=mean, nw_tstat=t))
            pd.DataFrame(dec).to_csv(OUT / 'by_decade.csv', index=False); tables[outcome]['by_decade'] = dec
            st = size_terciles(iss, outcome); st.to_csv(OUT / 'by_size_tercile.csv', index=False); tables[outcome]['by_size'] = st.to_dict('records')
    # Stale prices can make realized-variance proxies and one-month forecasts
    # degenerate. Report data-quality subsamples alongside the full sample.
    results['subsamples'] = {}; results['losses_subsamples'] = {}
    nz = {}
    for name, k in SUBSAMPLES.items():
        mask = iss.scored & (iss.nonzero_formation >= k) & (iss.nonzero_outcome >= k)
        results['subsamples'][name] = dict(rule=f'>= {k} non-zero daily returns in the last 21 formation sessions and in the 21 outcome sessions',
                                           share_of_scored=float(mask.sum() / iss.scored.sum()), excluded_issuer_dates=int(iss.scored.sum() - mask.sum()))
        sub = iss[mask]
        tables_sub = {}
        for outcome in (PRIMARY_OUTCOME, 'rv_broad'):
            block = {}
            for stat in ('mean', 'median', 'trimmed'):
                dl, inv = date_losses(sub, outcome, 'qlike', statistic=stat); dm = dm_table(dl)
                dl.to_csv(OUT / f'{name}_date_losses_{stat}_qlike_{outcome}.csv', index=False); dm.to_csv(OUT / f'{name}_dm_{stat}_qlike_{outcome}.csv', index=False)
                block[f'{stat}_qlike'] = {m: float(dl[m].mean()) for m in METHODS}; block[f'dm_{stat}_qlike'] = dm.to_dict('records')
                if stat == 'mean':
                    block['invalid_forecasts'] = inv
                    if outcome == PRIMARY_OUTCOME:
                        dl['decade'] = (dl.date.dt.year // 10) * 10
                        dec = []
                        for decade, g in dl.groupby('decade'):
                            rec = dict(decade=int(decade), dates=int(len(g)))
                            rec.update({m: float(g[m].mean()) for m in METHODS})
                            for other in ('ff3_1m', 'pca_garch', 'ff3_garch', 'pca_ewma', 'civolk10_ipv'):
                                d = (g['civol_ipv'] - g[other]).to_numpy(); mean, t, n = newey_west_tstat(d)
                                rec[f'diff_vs_{other}'] = mean; rec[f't_vs_{other}'] = t
                            dec.append(rec)
                        pd.DataFrame(dec).to_csv(OUT / f'{name}_by_decade.csv', index=False); block['by_decade'] = dec
            dl, _ = date_losses(sub, outcome, 'mselog'); dm = dm_table(dl); dm.to_csv(OUT / f'{name}_dm_mselog_{outcome}.csv', index=False)
            block['mean_mselog'] = {m: float(dl[m].mean()) for m in METHODS}; block['dm_mselog'] = dm.to_dict('records')
            tables_sub[outcome] = block
        results['losses_subsamples'][name] = tables_sub
        mzs = mincer_zarnowitz(sub, PRIMARY_OUTCOME); mzs.to_csv(OUT / f'{name}_mincer_zarnowitz_rv_ff3.csv', index=False)
        results[f'mincer_zarnowitz_rv_ff3_{name}'] = mzs.to_dict('records')
        nz[name] = sub
    tables_liquid = results['losses_subsamples']['nonstale']
    mz = mincer_zarnowitz(iss, PRIMARY_OUTCOME); mz.to_csv(OUT / 'mincer_zarnowitz_rv_ff3.csv', index=False)
    results['nonzero_return_distribution'] = dict(outcome_quantiles=np.percentile(iss.loc[iss.scored, 'nonzero_outcome'], [1, 5, 10, 25, 50, 75]).tolist(),
                                                  formation_quantiles=np.percentile(iss.loc[iss.scored, 'nonzero_formation'], [1, 5, 10, 25, 50, 75]).tolist(),
                                                  share_zero_window=float(((iss.nonzero_formation == 0) | (iss.nonzero_outcome == 0))[iss.scored].mean()),
                                                  share_tiny_rv_ff3=float((iss.loc[iss.scored, PRIMARY_OUTCOME] < 1e-8).mean()),
                                                  share_tiny_ff3_1m_forecast=float((iss.loc[iss.scored, 'ff3_1m'] < 1e-8).mean()))
    mzb = mincer_zarnowitz(iss, 'rv_broad'); mzb.to_csv(OUT / 'mincer_zarnowitz_rv_broad.csv', index=False)
    desc, pers = descriptives(iss, per_date); desc.to_csv(OUT / 'descriptives_by_date.csv', index=False); pers.to_csv(OUT / 'rank_persistence.csv', index=False)
    div_cols = [c for c in per_date.columns if c.startswith('div_') or c.startswith('insample_ew')]
    div = per_date[div_cols].describe(percentiles=[.1, .5, .9]).T
    div.to_csv(OUT / 'diversifiability_summary.csv')
    per_date['decade'] = (per_date.date.dt.year // 10) * 10
    decade_cols = ['rank', 'sign_rank', 'icp2', 'N', 'alpha', 'beta', 'gamma', 'rho_m', 'phi', 'objective_gain_component', 'compounding', 'mp_edge'] + \
        [c for c in ('div_ff3', 'div_broad', 'div_broad_ksign', 'div_broad_k10', 'div_broad_k20', 'div_total',
                     'insample_ew_ff3', 'insample_ew_pca', 'insample_ew_ksign', 'insample_ew_k10', 'insample_ew_k20',
                     'insample_top_ff3', 'insample_top_pca', 'insample_top_ksign', 'insample_top_k10', 'insample_top_k20') if c in per_date]
    by_decade_diag = per_date.groupby('decade')[decade_cols].median()
    by_decade_diag['dates'] = per_date.groupby('decade').size()
    by_decade_diag['share_rank_1'] = per_date.groupby('decade')['rank'].apply(lambda v: float((v == 1).mean()))
    by_decade_diag.to_csv(OUT / 'diagnostics_by_decade.csv')
    results['diagnostics_by_decade'] = by_decade_diag.reset_index().to_dict('records')
    results['rank_distribution'] = per_date['rank'].value_counts().sort_index().to_dict()
    results.update(losses=tables, mincer_zarnowitz_rv_ff3=mz.to_dict('records'), mincer_zarnowitz_rv_broad=mzb.to_dict('records'),
                   diversifiability=div[['mean', '10%', '50%', '90%']].to_dict('index'),
                   descriptives=dict(civol_median_of_medians=float(desc.civol_median.median()), ff3_median_of_medians=float(desc.ff3_median.median()),
                                     spearman_civol_ff3_median=float(desc.spearman_civol_ff3.median()), private_share_median=float(desc.private_share_median.median()),
                                     rel_q_mcse_median=float(desc.rel_q_mcse_median.median()), civol_rank_autocorr_median=float(pers.civol_rank_autocorr.median()),
                                     ff3_rank_autocorr_median=float(pers.ff3_rank_autocorr.median())),
                   rank=dict(mean=float(per_date['rank'].mean()), min=int(per_date['rank'].min()), max=int(per_date['rank'].max()),
                             sign_mean=float(per_date.sign_rank.mean()), share_rank_equals_sign=float((per_date['rank'] == per_date.sign_rank).mean())),
                   pooled=dict(alpha=float(per_date.alpha.median()), beta=float(per_date.beta.median()), gamma=float(per_date.gamma.median()),
                               alpha_ff3=float(per_date.alpha_ff3.median()), beta_ff3=float(per_date.beta_ff3.median()), gamma_ff3=float(per_date.gamma_ff3.median())),
                   compounding_factor_median=float(per_date.compounding.median()), surrogate_ratio_median=float(per_date.surrogate_ratio.median()),
                   seconds_per_date_mean=float(per_date.seconds.mean()))
    (OUT / 'SUMMARY.json').write_text(json.dumps(results, indent=1, default=float))
    figures(per_date, results['losses_subsamples']['nonstale'][PRIMARY_OUTCOME]['mean_qlike'], desc)
    with pd.option_context('display.width', 220, 'display.max_columns', 30):
        print('dates', len(per_date), 'issuer-dates', len(iss), 'scored', int(iss.scored.sum()))
        for outcome in OUTCOMES:
            print(f'\n== losses against {outcome} (mean QLIKE | median QLIKE | trimmed QLIKE | MSE log) ==')
            frame = pd.DataFrame(dict(mean_qlike=tables[outcome]['mean_qlike'], median_qlike=tables[outcome]['median_qlike'],
                                      trimmed_qlike=tables[outcome]['trimmed_qlike'], mse_log=tables[outcome]['mean_mselog'])).rename(index=LABELS)
            print(frame.sort_values('trimmed_qlike').round(4).to_string())
        print('\n== DM: cIVOL minus comparator, primary outcome, mean QLIKE ==')
        print(pd.DataFrame(tables[PRIMARY_OUTCOME]['dm'])[['label', 'mean_difference', 'nw_tstat', 'win_share']].round(4).to_string())
        print('\n== DM: cIVOL minus comparator, primary outcome, trimmed QLIKE ==')
        print(pd.DataFrame(tables[PRIMARY_OUTCOME]['dm_trimmed_qlike'])[['label', 'mean_difference', 'nw_tstat', 'win_share']].round(4).to_string())
        print('\n== DM: cIVOL minus comparator, primary outcome, MSE log ==')
        print(pd.DataFrame(tables[PRIMARY_OUTCOME]['dm_mselog'])[['label', 'mean_difference', 'nw_tstat', 'win_share']].round(4).to_string())
        print('\n== nonzero-return distribution ==', json.dumps(results['nonzero_return_distribution']))
        for name in SUBSAMPLES:
            b = results['losses_subsamples'][name][PRIMARY_OUTCOME]
            print(f"\n== SUBSAMPLE {name}: {results['subsamples'][name]} ==")
            frame = pd.DataFrame(dict(mean_qlike=b['mean_qlike'], median_qlike=b['median_qlike'], trimmed_qlike=b['trimmed_qlike'], mse_log=b['mean_mselog'])).rename(index=LABELS)
            print(frame.sort_values('mean_qlike').round(4).to_string())
            print(f'-- {name} DM: cIVOL minus comparator, primary outcome, mean QLIKE --')
            print(pd.DataFrame(b['dm_mean_qlike'])[['label', 'mean_difference', 'nw_tstat', 'win_share']].round(4).to_string())
            print(f'-- {name} by decade (mean QLIKE, primary outcome) --')
            print(pd.DataFrame(b['by_decade'])[['decade', 'dates', 'civol_ipv', 'ff3_1m', 'pca_garch', 'pca_ewma', 'civolk10_ipv', 'diff_vs_ff3_1m', 't_vs_ff3_1m', 'diff_vs_pca_garch', 't_vs_pca_garch']].round(4).to_string())
        print('\n== diversifiability (median over dates) ==')
        print(div['50%'].round(3))
        print('\n== diagnostics by decade (medians) ==')
        print(by_decade_diag.round(3).to_string())
        print('rank distribution', results['rank_distribution'])
        print('\n== descriptives ==', json.dumps(results['descriptives'], indent=1))
        print('rank', results['rank'], 'pooled', results['pooled'])


if __name__ == '__main__':
    main()
