"""Analyze rank-rule records against the primary application records.

Reads only completed records and joins each date with the production record of the same date (same issuers, asserted).

    python -m civol.rank_rule.analyze_rank_rule [--records DIR] [--out DIR] [--quiet]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from civol.core.paths import records_path, output_path
import pandas as pd

from civol.core.paths import RECORDS as PRIMARY_RECORDS
from civol.core.universe import Cache
from civol.application.analyze import qlike, newey_west_tstat, nonzero_counts

RECORDS = records_path('rank_rule')
OUT = output_path('rank_rule', 'results')
NONSTALE = 5          # >= 5 non-zero daily returns in the last 21 formation sessions and in the 21 outcome sessions (headline sample)
BAND = (1 / 1.25, 1.25)
MATERIAL = .010       # materiality threshold for mean-QLIKE differences

METHODS = ['civol_ipv', 'civolk10_ipv', 'civolA_ipv', 'civolB_ipv', 'civolK20_ipv', 'ff3paired_ipv', 'ff3_garch', 'pca_garch', 'ff3_1m']
NEW = ['civolA_ipv', 'civolB_ipv', 'civolK20_ipv']
LABELS = dict(civol_ipv='cIVOL (Bai-Ng, primary)', civolk10_ipv='cIVOL-K10', civolA_ipv='cIVOL rule A', civolB_ipv='cIVOL rule B', civolK20_ipv='cIVOL-K20',
              ff3paired_ipv='FF3-paired', ff3_garch='FF3-GARCH', pca_garch='PCA-GARCH', ff3_1m='FF3-1m (standard)')
OUTCOMES = ['rv_ff3', 'rv_broad_primary', 'rv_broad_k10', 'rv_broad_kA', 'rv_broad_kB', 'rv_broad_k20']
PRIMARY_OUTCOME = 'rv_ff3'
SPECS = ['ff3', 'primary', 'sign', 'k10', 'k20', 'A', 'B']       # diversifiability specifications (residual definitions)
SPEC_LABELS = dict(ff3='FF3 residuals', primary='Broad, Bai-Ng K (primary)', sign='Broad, sign-rank K', k10='Broad, K = 10', k20='Broad, K = 20',
                   A='Broad, rule A', B='Broad, rule B')
DECISION_SPECS = ['primary', 'k10', 'k20', 'A', 'B']
DECISION_METHOD = dict(primary='civol_ipv', k10='civolk10_ipv', k20='civolK20_ipv', A='civolA_ipv', B='civolB_ipv')


def load_records(records=RECORDS):
    cache = Cache()
    per_date = []; frames = []; failed = sorted(p.name for p in records.glob('*.FAILED.json'))
    curves = {}
    for js in sorted(records.glob('*.json')):
        if js.name.endswith('.FAILED.json') or js.name.startswith('RUN') or js.name.startswith('PILOT'):
            continue
        s = json.loads(js.read_text())
        if not s.get('outcome_available'):
            continue
        date = s['date']
        z = np.load(js.with_suffix('.npz'))
        pj = json.loads((PRIMARY_RECORDS / f'{date}.json').read_text()); pz = np.load(PRIMARY_RECORDS / f'{date}.npz')
        assert np.array_equal(z['permno'], pz['permno']) and np.array_equal(z['scored'], pz['scored']), date
        r = s['rules']; ka, kb = r['rank_a'], r['rank_b']; kp = r['icp2_rank']; ks = r['sign_rank']
        assert kp == pj['rank'], (date, kp, pj['rank'])
        oc = s['outcomes']; ew = oc['equal_weight']; nd = oc['next_direction']; rs = oc['random_signed']
        key = dict(ff3='ff3', primary=str(kp), sign=str(ks), k10='10', k20='20', A=str(ka), B=str(kb))
        rec = dict(date=pd.Timestamp(date), N=s['N'], scored=oc['scored_count'], rank_a=ka, rank_b=kb, rank_primary=kp, rank_sign=ks,
                   a_attainable=r['rank_a_bound_attainable'], b_attained=r['rank_b_attained'], ew_floor=r['equal_weight_floor'], heldout_minimum=r['heldout_minimum'],
                   insample_ew_A=s['curves']['insample_equal_weight'][ka], insample_ew_B=s['curves']['insample_equal_weight'][kb],
                   insample_ew_primary=s['curves']['insample_equal_weight'][kp], insample_ew_k10=s['curves']['insample_equal_weight'][10],
                   insample_ew_k20=s['curves']['insample_equal_weight'][20], heldout_A=s['curves']['heldout_next_direction'][ka],
                   heldout_B=s['curves']['heldout_next_direction'][kb], heldout_k10=s['curves']['heldout_next_direction'][10],
                   heldout_k20=s['curves']['heldout_next_direction'][20], heldout_primary=s['curves']['heldout_next_direction'][kp],
                   div_total=ew['total'], seconds=s['seconds'], pilot=bool(pj.get('pilot', False)))
        for b, v in r['bound_sensitivity']['rule_a'].items():
            rec[f'rank_a_bound_{b}'] = v
        for b, v in r['bound_sensitivity']['rule_b'].items():
            rec[f'rank_b_bound_{b}'] = v
        for spec, k in key.items():
            rec[f'div_{spec}'] = ew[k]; rec[f'next_{spec}'] = nd[k]
            rec[f'rs_mean_{spec}'] = rs[k]['mean']; rec[f'rs_sd_{spec}'] = rs[k]['sd']; rec[f'rs_p95_{spec}'] = rs[k]['p95']; rec[f'rs_identity_{spec}'] = rs[k]['identity_prediction']
        for role in ('A', 'B', 'K20'):
            m = s['models'][role]; c = m['pooled']['coefficients']
            rec.update({f'{role}_alpha': c[0], f'{role}_beta': c[1], f'{role}_gamma': c[2], f'{role}_rho_m': c[3], f'{role}_phi': c[4], f'{role}_converged': m['pooled']['converged'],
                        f'{role}_static': m['pooled']['static_selected'], f'{role}_backend': m['integration']['backend'], f'{role}_rel_q_mcse': m['integration']['median_relative_q_mcse'],
                        f'{role}_compounding': m['integration']['median_compounding_factor'], f'{role}_private_share': m['integration']['median_private_share'],
                        f'{role}_top_eig': m['insample']['top_residual_correlation_eigenvalue'], f'{role}_fg_converged': m['factor_garch']['converged'],
                        f'{role}_fg_failures': m['factor_garch']['optimizer_failures']})
        chk = s.get('primary_record_check', {})
        rec['check_rv_ff3_max_rel_diff'] = chk.get('rv_ff3_max_abs_rel_diff'); rec['check_insample_ew_max_diff'] = chk.get('insample_equal_weight_max_abs_diff')
        per_date.append(rec)
        curves[date] = dict(ew=s['curves']['insample_equal_weight'], heldout=s['curves']['heldout_next_direction'], rs_mean=s['curves']['insample_random_signed_mean'],
                            rs_sd=s['curves']['insample_random_signed_sd'])
        nz_form, nz_out = nonzero_counts(cache, int(s['t_idx']), z['permno'])
        cols = dict(permno=z['permno'], mktcap=z['mktcap'], scored=z['scored'], nonzero_formation=nz_form, nonzero_outcome=nz_out,
                    civol_ipv=pz['civol_ipv'], civolk10_ipv=pz['civolk10_ipv'], ff3paired_ipv=pz['ff3paired_ipv'], ff3_garch=pz['ff3_garch'], pca_garch=pz['pca_garch'],
                    ff3_1m=pz['ff3_1m'], civolA_ipv=z['civolA_ipv'], civolB_ipv=z['civolB_ipv'], civolK20_ipv=z['civolK20_ipv'],
                    civol_q=pz['civol_q'], civolA_q=z['civolA_q'], civolB_q=z['civolB_q'], civolK20_q=z['civolK20_q'], civolk10_q=pz['civolk10_q'],
                    rv_ff3=z['rv_ff3'], rv_broad_primary=pz['rv_broad'], rv_broad_k10=z['rv_broad_k10'], rv_broad_kA=z[f'rv_broad_k{ka}'], rv_broad_kB=z[f'rv_broad_k{kb}'],
                    rv_broad_k20=z['rv_broad_k20'])
        f = pd.DataFrame(cols); f['date'] = pd.Timestamp(date)
        frames.append(f)
    return pd.DataFrame(per_date).sort_values('date').reset_index(drop=True), pd.concat(frames, ignore_index=True), curves, failed


def date_losses(iss, outcome, loss='qlike', statistic='mean'):
    """Cross-sectional mean / median / 1%-trimmed mean loss per date and method;
    invalid forecasts receive the date-method maximum loss and are counted."""
    rows = []; invalid = {m: 0 for m in METHODS}
    for date, g in iss[iss.scored].groupby('date', sort=True):
        rv = g[outcome].to_numpy(float); ok_rv = np.isfinite(rv) & (rv > 0)
        rec = dict(date=date, n=int(ok_rv.sum()))
        for m in METHODS:
            f = g[m].to_numpy(float); ok = ok_rv & np.isfinite(f) & (f > 0)
            l = qlike(rv[ok], f[ok]) if loss == 'qlike' else (np.log(rv[ok]) - np.log(f[ok])) ** 2
            bad = int(ok_rv.sum() - ok.sum()); invalid[m] += bad
            if len(l):
                full = np.r_[l, np.full(bad, l.max())]
                if statistic == 'mean':
                    rec[m] = float(full.mean())
                elif statistic == 'median':
                    rec[m] = float(np.median(full))
                else:
                    lo, hi = np.percentile(full, [1, 99]); rec[m] = float(full[(full >= lo) & (full <= hi)].mean())
            else:
                rec[m] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows), invalid


def dm_table(dl, base, lag=3):
    rows = []
    for m in METHODS:
        if m == base:
            continue
        d = (dl[base] - dl[m]).to_numpy()
        mean, t, n = newey_west_tstat(d, lag)
        rows.append(dict(base=base, comparator=m, label=LABELS[m], base_mean=float(dl[base].mean()), comparator_mean=float(dl[m].mean()), mean_difference=mean,
                         nw_tstat=t, dates=n, base_wins=int(np.sum(d < 0)), win_share=float(np.mean(d < 0))))
    return pd.DataFrame(rows)


def diversifiability_table(per_date, prefix='div'):
    rows = []
    for spec in SPECS:
        v = per_date[f'{prefix}_{spec}'].to_numpy(float); v = v[np.isfinite(v) & (v > 0)]
        if not len(v):
            continue
        lg = np.abs(np.log(v))
        rows.append(dict(spec=spec, label=SPEC_LABELS[spec], dates=len(v), median=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                         median_abs_log=float(np.median(lg)), mean_abs_log=float(lg.mean()), band_share=float(np.mean((v >= BAND[0]) & (v <= BAND[1]))),
                         share_above=float(np.mean(v > BAND[1])), share_below=float(np.mean(v < BAND[0]))))
    return pd.DataFrame(rows)


def by_decade_medians(per_date, cols):
    g = per_date.assign(decade=(per_date.date.dt.year // 10) * 10).groupby('decade')
    out = g[cols].median(); out['dates'] = g.size()
    return out


def paired_abs_log_tests(per_date, base, others, prefix='div'):
    rows = []
    b = np.abs(np.log(per_date[f'{prefix}_{base}'].to_numpy(float)))
    for o in others:
        d = b - np.abs(np.log(per_date[f'{prefix}_{o}'].to_numpy(float)))
        mean, t, n = newey_west_tstat(d)
        rows.append(dict(base=base, comparator=o, mean_difference_abs_log=mean, nw_tstat=t, dates=n, base_closer_share=float(np.mean(d < 0))))
    return pd.DataFrame(rows)


def decision(per_date, mean_qlike_nonstale):
    """Select the lowest median |log OOS equal-weight ratio|
    among the five specifications whose non-stale mean QLIKE (RV-FF3) is within
    MATERIAL of the lowest of the five."""
    rows = []
    best_q = min(mean_qlike_nonstale[DECISION_METHOD[s]] for s in DECISION_SPECS)
    for s in DECISION_SPECS:
        v = per_date[f'div_{s}'].to_numpy(float); v = v[np.isfinite(v) & (v > 0)]
        q = mean_qlike_nonstale[DECISION_METHOD[s]]
        rows.append(dict(spec=s, method=DECISION_METHOD[s], median_abs_log_ew=float(np.median(np.abs(np.log(v)))), mean_qlike_nonstale=q, qlike_excess_over_best=q - best_q,
                         eligible=bool(q - best_q <= MATERIAL)))
    tab = pd.DataFrame(rows)
    eligible = tab[tab.eligible].sort_values('median_abs_log_ew')
    return tab, (eligible.spec.iloc[0] if len(eligible) else None)


def figures(per_date, curves, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    ax[0].plot(per_date.date, per_date.rank_primary, label='Bai-Ng (primary)', lw=.9); ax[0].plot(per_date.date, per_date.rank_a, label='rule A (equal-weight bound)', lw=1.2)
    ax[0].plot(per_date.date, per_date.rank_b, label='rule B (held-out next direction)', lw=.9, alpha=.8); ax[0].axhline(10, color='k', lw=.6, ls='--', label='K = 10')
    ax[0].set_ylabel('selected K'); ax[0].legend(ncol=2); ax[0].set_title('Common-space rank by rule')
    for spec, lab in (('ff3', 'FF3'), ('primary', 'Bai-Ng'), ('k10', 'K = 10'), ('A', 'rule A'), ('B', 'rule B')):
        ax[1].semilogy(per_date.date, per_date[f'div_{spec}'], label=lab, lw=.9, alpha=.85)
    ax[1].axhline(1, color='k', lw=.8); ax[1].axhspan(BAND[0], BAND[1], color='k', alpha=.08); ax[1].set_ylabel('OOS equal-weight residual ratio'); ax[1].legend(ncol=3)
    ax[1].set_title('Out-of-sample equal-weight diversifiability (1 = uncorrelated residuals; band = declared bound)')
    for spec, lab in (('ff3', 'FF3'), ('primary', 'Bai-Ng'), ('k10', 'K = 10'), ('A', 'rule A'), ('B', 'rule B')):
        ax[2].semilogy(per_date.date, per_date[f'next_{spec}'], label=lab, lw=.9, alpha=.85)
    ax[2].axhline(1, color='k', lw=.8); ax[2].set_ylabel('OOS next-direction ratio'); ax[2].legend(ncol=3)
    ax[2].set_title('Realized common variation along the strongest remaining formation direction')
    fig.tight_layout(); fig.savefig(out / 'rank_rule_time_series.png', dpi=130); plt.close(fig)
    # median curves by decade
    decades = sorted(set((d.year // 10) * 10 for d in per_date.date))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for dec in decades:
        dates = [str(d.date()) for d in per_date.date if (d.year // 10) * 10 == dec]
        ew = np.median([curves[d]['ew'] for d in dates if d in curves], axis=0); ho = np.median([curves[d]['heldout'] for d in dates if d in curves], axis=0)
        ax[0].semilogy(range(len(ew)), ew, label=f'{dec}s'); ax[1].semilogy(range(len(ho)), ho, label=f'{dec}s')
    for a, title in zip(ax, ('In-sample equal-weight ratio (rule A statistic)', 'Held-out next-direction ratio (rule B statistic)')):
        a.axhline(BAND[1], color='k', lw=.8, ls='--'); a.axhline(1, color='k', lw=.6); a.set_xlabel('k'); a.set_title(title + ', median over dates'); a.legend()
    fig.tight_layout(); fig.savefig(out / 'rule_curves_by_decade.png', dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--records', default=str(RECORDS)); p.add_argument('--out', default=str(OUT)); p.add_argument('--quiet', action='store_true')
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_date, iss, curves, failed = load_records(Path(a.records))
    per_date.to_csv(out / 'per_date.csv', index=False)
    results = dict(dates=int(len(per_date)), failed_records=failed, first=str(per_date.date.min().date()), last=str(per_date.date.max().date()),
                   issuer_dates=int(len(iss)), scored_issuer_dates=int(iss.scored.sum()),
                   backends={role: per_date[f'{role}_backend'].value_counts().to_dict() for role in ('A', 'B', 'K20')},
                   pooled_converged_share={role: float(per_date[f'{role}_converged'].mean()) for role in ('A', 'B', 'K20')},
                   static_selected_dates={role: int(per_date[f'{role}_static'].sum()) for role in ('A', 'B', 'K20')},
                   consistency=dict(max_rv_ff3_rel_diff=float(np.nanmax(per_date.check_rv_ff3_max_rel_diff.astype(float))),
                                    max_insample_ew_diff=float(np.nanmax(per_date.check_insample_ew_max_diff.astype(float)))),
                   seconds_per_date_mean=float(per_date.seconds.mean()))
    # --- ranks
    results['ranks'] = dict(rank_a_distribution=per_date.rank_a.value_counts().sort_index().to_dict(), rank_b_distribution=per_date.rank_b.value_counts().sort_index().to_dict(),
                            rank_a=dict(median=float(per_date.rank_a.median()), min=int(per_date.rank_a.min()), max=int(per_date.rank_a.max()),
                                        bound_attainable_share=float(per_date.a_attainable.mean()), share_above_primary=float((per_date.rank_a > per_date.rank_primary).mean()),
                                        share_above_10=float((per_date.rank_a > 10).mean()), share_equal_10=float((per_date.rank_a == 10).mean())),
                            rank_b=dict(median=float(per_date.rank_b.median()), min=int(per_date.rank_b.min()), max=int(per_date.rank_b.max()),
                                        attained_share=float(per_date.b_attained.mean()), share_at_kmax=float((per_date.rank_b == per_date.rank_b.max()).mean())),
                            by_decade=by_decade_medians(per_date, ['rank_primary', 'rank_sign', 'rank_a', 'rank_b', 'ew_floor', 'heldout_minimum']).reset_index().to_dict('records'),
                            bound_sensitivity_median={c: float(per_date[c].median()) for c in per_date.columns if c.startswith('rank_a_bound') or c.startswith('rank_b_bound')})
    dec = per_date.assign(decade=(per_date.date.dt.year // 10) * 10).groupby('decade')
    results['ranks']['attainable_share_by_decade'] = dec.a_attainable.mean().to_dict(); results['ranks']['b_attained_share_by_decade'] = dec.b_attained.mean().to_dict()
    # --- diversifiability
    div = diversifiability_table(per_date, 'div'); div.to_csv(out / 'diversifiability_equal_weight.csv', index=False)
    nxt = diversifiability_table(per_date, 'next'); nxt.to_csv(out / 'diversifiability_next_direction.csv', index=False)
    div_dec = by_decade_medians(per_date, [f'div_{s}' for s in SPECS] + [f'next_{s}' for s in SPECS] + ['div_total']); div_dec.to_csv(out / 'diversifiability_by_decade.csv')
    rs_rows = []
    for spec in SPECS:
        rs_rows.append(dict(spec=spec, label=SPEC_LABELS[spec], median_mean=float(per_date[f'rs_mean_{spec}'].median()), p10_mean=float(per_date[f'rs_mean_{spec}'].quantile(.1)),
                            p90_mean=float(per_date[f'rs_mean_{spec}'].quantile(.9)), median_sd=float(per_date[f'rs_sd_{spec}'].median()), median_p95=float(per_date[f'rs_p95_{spec}'].median()),
                            max_abs_dev_from_identity=float((per_date[f'rs_mean_{spec}'] - per_date[f'rs_identity_{spec}']).abs().max())))
    rs = pd.DataFrame(rs_rows); rs.to_csv(out / 'diversifiability_random_signed.csv', index=False)
    calib = dict(insample_vs_oos_median={s: dict(insample=float(per_date[f'insample_ew_{s}'].median()), oos=float(per_date[f'div_{s}'].median()),
                                                   median_log_ratio_oos_over_insample=float(np.median(np.log(per_date[f'div_{s}'] / per_date[f'insample_ew_{s}']))))
                                          for s in ('A', 'B', 'primary', 'k10', 'k20')},
                 heldout_vs_oos_next_median={s: dict(heldout=float(per_date[f'heldout_{s}'].median()), oos=float(per_date[f'next_{s}'].median())) for s in ('A', 'B', 'primary', 'k10', 'k20')})
    tests_div = pd.concat([paired_abs_log_tests(per_date, 'A', ['primary', 'k10', 'k20', 'B', 'ff3']), paired_abs_log_tests(per_date, 'B', ['primary', 'k10', 'k20'])])
    tests_div.to_csv(out / 'diversifiability_paired_tests.csv', index=False)
    results.update(diversifiability_equal_weight=div.to_dict('records'), diversifiability_next_direction=nxt.to_dict('records'), random_signed=rs.to_dict('records'),
                   calibration=calib, diversifiability_paired_tests=tests_div.to_dict('records'), diversifiability_by_decade=div_dec.reset_index().to_dict('records'))
    # --- forecast losses
    iss['nonstale'] = iss.scored & (iss.nonzero_formation >= NONSTALE) & (iss.nonzero_outcome >= NONSTALE)
    results['samples'] = dict(nonstale_share_of_scored=float(iss.nonstale.sum() / iss.scored.sum()))
    losses = {}
    for sample in ('nonstale', 'full'):
        sub = iss[iss.nonstale] if sample == 'nonstale' else iss
        losses[sample] = {}
        for outcome in OUTCOMES:
            block = {}
            for loss, stat in (('qlike', 'mean'), ('qlike', 'median'), ('qlike', 'trimmed'), ('mselog', 'mean')):
                name = f'{stat}_{loss}' if loss == 'qlike' else 'mselog'
                dl, inv = date_losses(sub, outcome, loss, stat)
                dl.to_csv(out / f'{sample}_date_losses_{name}_{outcome}.csv', index=False)
                block[name] = {m: float(dl[m].mean()) for m in METHODS}
                if name == 'mean_qlike':
                    block['invalid_forecasts'] = inv
                dms = pd.concat([dm_table(dl, base) for base in NEW + ['civolk10_ipv']])
                dms.to_csv(out / f'{sample}_dm_{name}_{outcome}.csv', index=False)
                block[f'dm_{name}'] = dms.to_dict('records')
                if outcome == PRIMARY_OUTCOME and name == 'mean_qlike':
                    dl['decade'] = (dl.date.dt.year // 10) * 10
                    rows = []
                    for decade, g in dl.groupby('decade'):
                        rec = dict(decade=int(decade), dates=int(len(g))); rec.update({m: float(g[m].mean()) for m in METHODS})
                        for base in NEW:
                            for other in ('civolk10_ipv', 'civol_ipv'):
                                d = (g[base] - g[other]).to_numpy(); mean, t, n = newey_west_tstat(d)
                                rec[f'{base}_minus_{other}'] = mean; rec[f't_{base}_minus_{other}'] = t
                        rows.append(rec)
                    pd.DataFrame(rows).to_csv(out / f'{sample}_by_decade_{outcome}.csv', index=False); block['by_decade'] = rows
            losses[sample][outcome] = block
    results['losses'] = losses
    # --- decision rule
    tab, recommended = decision(per_date, losses['nonstale'][PRIMARY_OUTCOME]['mean_qlike'])
    tab.to_csv(out / 'decision_rule.csv', index=False)
    results['decision'] = dict(table=tab.to_dict('records'), recommended=recommended, materiality=MATERIAL, band=BAND)
    # --- descriptives of the new measures
    g = iss[iss.scored]
    desc = {}
    for role, qcol in (('A', 'civolA_q'), ('B', 'civolB_q'), ('K20', 'civolK20_q')):
        rows = []
        for date, gg in g.groupby('date'):
            rows.append(dict(spearman_vs_primary=np.sqrt(gg[qcol]).corr(np.sqrt(gg.civol_q), method='spearman'),
                             spearman_vs_k10=np.sqrt(gg[qcol]).corr(np.sqrt(gg.civolk10_q), method='spearman'),
                             median_ratio_to_primary=float(np.median(gg[qcol] / gg.civol_q)), median_ratio_to_k10=float(np.median(gg[qcol] / gg.civolk10_q)),
                             median_ivol=float(np.median(np.sqrt(gg[qcol])))))
        d = pd.DataFrame(rows)
        desc[role] = {c: float(d[c].median()) for c in d.columns}
        desc[role].update(private_share_median=float(per_date[f'{role}_private_share'].median()), rel_q_mcse_median=float(per_date[f'{role}_rel_q_mcse'].median()),
                          compounding_median=float(per_date[f'{role}_compounding'].median()),
                          pooled_median={c: float(per_date[f'{role}_{c}'].median()) for c in ('alpha', 'beta', 'gamma', 'rho_m', 'phi')},
                          top_residual_eigenvalue_median=float(per_date[f'{role}_top_eig'].median()))
    results['descriptives'] = desc
    (out / 'SUMMARY.json').write_text(json.dumps(results, indent=1, default=float))
    figures(per_date, curves, out)
    if a.quiet:
        print('analysis complete:', len(per_date), 'dates', len(iss), 'issuer-dates; tables written to', out)
        return
    with pd.option_context('display.width', 230, 'display.max_columns', 40):
        print('dates', len(per_date), 'issuer-dates', len(iss), 'scored', int(iss.scored.sum()), 'failed records', failed)
        print('consistency', results['consistency'], 'backends', results['backends'], 'converged', results['pooled_converged_share'])
        print('\n== ranks ==\n', json.dumps({k: v for k, v in results['ranks'].items() if k != 'by_decade'}, indent=1, default=float))
        print(pd.DataFrame(results['ranks']['by_decade']).round(2).to_string())
        print('\n== OOS equal-weight diversifiability ==\n', div.round(3).to_string())
        print('\n== OOS next-direction ratio ==\n', nxt.round(3).to_string())
        print('\n== random-signed check ==\n', rs.round(4).to_string())
        print('\n== by decade (medians) ==\n', div_dec.round(2).to_string())
        print('\n== calibration ==', json.dumps(calib, indent=1))
        print('\n== paired |log| tests ==\n', tests_div.round(3).to_string())
        for sample in ('nonstale', 'full'):
            for outcome in OUTCOMES:
                b = losses[sample][outcome]
                frame = pd.DataFrame(dict(mean_qlike=b['mean_qlike'], median_qlike=b['median_qlike'], trimmed_qlike=b['trimmed_qlike'], mselog=b['mselog'])).rename(index=LABELS)
                print(f'\n== {sample} losses against {outcome} ==\n', frame.sort_values('mean_qlike').round(4).to_string())
            b = losses[sample][PRIMARY_OUTCOME]
            print(f'\n== {sample} DM mean QLIKE, RV-FF3 (base minus comparator; negative favours base) ==')
            print(pd.DataFrame(b['dm_mean_qlike'])[['base', 'label', 'mean_difference', 'nw_tstat', 'win_share']].round(4).to_string())
            print(f'\n== {sample} DM median QLIKE, RV-FF3 ==')
            print(pd.DataFrame(b['dm_median_qlike'])[['base', 'label', 'mean_difference', 'nw_tstat', 'win_share']].round(4).to_string())
            print(f'\n== {sample} DM MSE-log, RV-FF3 ==')
            print(pd.DataFrame(b['dm_mselog'])[['base', 'label', 'mean_difference', 'nw_tstat', 'win_share']].round(4).to_string())
            print(f'\n== {sample} by decade, mean QLIKE, RV-FF3 ==')
            print(pd.DataFrame(b['by_decade']).round(4).to_string())
        print('\n== decision rule ==\n', tab.round(4).to_string(), '\nrecommended:', recommended)
        print('\n== descriptives ==', json.dumps(desc, indent=1))


if __name__ == '__main__':
    main()
