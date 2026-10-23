"""Analyze simulation records against the known population values."""
from __future__ import annotations
import json
import numpy as np
from civol.core.paths import output_path
import pandas as pd

from civol.core.paths import SIM_RECORDS
from civol.simulation.population import CELLS, K0

OUT = output_path('simulation', 'results')
IPV_METHODS = ['civol_ipv', 'civolk10_ipv', 'civol1c_ipv', 'civol1cT_ipv', 'ff3paired_ipv', 'ff3_1m', 'ff3_12m', 'ff3_ewma', 'ff3_garch', 'pca_1m', 'pca_ewma', 'pca_garch',
               'pcak10_1m', 'pcak10_ewma', 'total_1m', 'civol_surrogate_ipv']
Q_METHODS = ['civol_q', 'civolk10_q', 'civol1c_q', 'civol1cT_q', 'ff3paired_q', 'civol_surrogate_q']
LABELS = dict(civol_ipv='cIVOL', civolk10_ipv='cIVOL-K10', civol1c_ipv='cIVOL one-component', civol1c_q='cIVOL one-component q', civol1cT_ipv='cIVOL accepted law (total driver)', civol1cT_q='cIVOL accepted law q', ff3paired_ipv='FF3-paired', ff3_1m='FF3-1m (standard)', ff3_12m='FF3-12m', ff3_ewma='FF3-EWMA',
              ff3_garch='FF3-GARCH', pca_1m='PCA-1m', pca_ewma='PCA-EWMA', pca_garch='PCA-GARCH', pcak10_1m='PCA(K10)-1m', pcak10_ewma='PCA(K10)-EWMA',
              total_1m='Total variance 1m', civol_surrogate_ipv='cIVOL surrogate (deterministic)', civol_q='cIVOL q', civolk10_q='cIVOL-K10 q',
              ff3paired_q='FF3-paired q', civol_surrogate_q='cIVOL surrogate q')


def qlike(truth, forecast):
    r = truth / forecast
    return r - np.log(r) - 1


def metrics(truth, forecast):
    ok = np.isfinite(forecast) & (forecast > 0)
    f = np.where(ok, forecast, np.nan)
    rel = f / truth
    out = dict(qlike=float(np.nanmean(qlike(truth, f))), median_abs_rel_error=float(np.nanmedian(np.abs(rel - 1))),
               rel_rmse=float(np.sqrt(np.nanmean((rel - 1) ** 2))), spearman=float(pd.Series(f).corr(pd.Series(truth), method='spearman')),
               outside_factor2=float(np.nanmean((rel < .5) | (rel > 2))), median_ratio=float(np.nanmedian(rel)), invalid=int((~ok).sum()))
    return out


def figure(agg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a = agg[(agg.truth == 'ipv') & (agg.method != 'civol_surrogate_ipv')]
    piv = a.pivot_table(index='label', columns='cell', values='qlike')
    piv = piv.loc[piv['baseline'].sort_values(ascending=False).index, list(CELLS)]
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(piv)); width = .2
    for j, cell in enumerate(CELLS):
        ax.barh(y + (j - 1.5) * width, piv[cell].values, height=width, label=cell.replace('_', ' '))
    ax.set_yticks(y); ax.set_yticklabels(piv.index); ax.set_xscale('log')
    ax.set_xlabel('mean QLIKE against true integrated private variance (log scale, lower is better)')
    ax.set_title('Truth-known simulation, N = 2000, T = 750, 8 economies per cell'); ax.legend(loc='upper right')
    fig.tight_layout(); fig.savefig(OUT / 'simulation_qlike.png', dpi=130); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []; ranks = []
    for js in sorted(SIM_RECORDS.glob('*.json')):
        if js.name.endswith('.FAILED.json') or js.name.startswith('RUN'):
            continue
        s = json.loads(js.read_text()); z = np.load(js.with_suffix('.npz'))
        ranks.append(dict(cell=s['cell'], rep=s['rep'], rank=s['rank'], sign_rank=s['rank_diagnostics']['sign_rank'], icp2=s['rank_diagnostics']['icp2_rank'],
                          pooled=s['pooled']['coefficients'], truth_q_mcse=s['truth']['median_relative_q_mcse'], seconds=s['seconds']))
        for m in IPV_METHODS:
            if m in z.files:
                rows.append(dict(cell=s['cell'], rep=s['rep'], method=m, truth='ipv', **metrics(z['ipv_true'], z[m])))
        for m in Q_METHODS:
            if m in z.files:
                rows.append(dict(cell=s['cell'], rep=s['rep'], method=m, truth='q', **metrics(z['q_true'], z[m])))
    df = pd.DataFrame(rows); rk = pd.DataFrame(ranks)
    df.to_csv(OUT / 'economy_metrics.csv', index=False); rk.to_csv(OUT / 'economy_ranks.csv', index=False)
    agg = df.groupby(['cell', 'truth', 'method']).agg(qlike=('qlike', 'mean'), qlike_se=('qlike', lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                                                     median_abs_rel_error=('median_abs_rel_error', 'mean'), rel_rmse=('rel_rmse', 'mean'),
                                                     spearman=('spearman', 'mean'), outside_factor2=('outside_factor2', 'mean'),
                                                     median_ratio=('median_ratio', 'mean'), economies=('rep', 'count')).reset_index()
    agg['label'] = agg.method.map(LABELS)
    agg.to_csv(OUT / 'cell_metrics.csv', index=False)
    # paired primary: cIVOL minus FF3-1m, per cell
    piv = df[df.truth == 'ipv'].pivot_table(index=['cell', 'rep'], columns='method', values='qlike')
    paired = []
    for cell in CELLS:
        if cell not in piv.index.get_level_values(0):
            continue
        p = piv.loc[cell]
        for other in IPV_METHODS:
            if other == 'civol_ipv' or other not in p:
                continue
            diff = p['civol_ipv'] - p[other]
            paired.append(dict(cell=cell, comparator=other, label=LABELS.get(other, other), mean_difference=float(diff.mean()),
                               se=float(diff.std(ddof=1) / np.sqrt(len(diff))), wins=int((diff < 0).sum()), economies=int(len(diff))))
    paired = pd.DataFrame(paired); paired.to_csv(OUT / 'paired_differences.csv', index=False)
    summary = dict(economies=int(len(rk)), cells={c: int((rk.cell == c).sum()) for c in CELLS},
                   rank_selected={c: rk.loc[rk.cell == c, 'rank'].tolist() for c in CELLS}, true_rank=K0,
                   sign_rank={c: rk.loc[rk.cell == c, 'sign_rank'].tolist() for c in CELLS},
                   primary={c: paired.loc[(paired.cell == c) & (paired.comparator == 'ff3_1m')].drop(columns=['cell']).to_dict('records') for c in CELLS},
                   cell_metrics=agg.to_dict('records'))
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=1, default=float))
    figure(agg)
    with pd.option_context('display.width', 200, 'display.max_rows', 500):
        print(agg[agg.truth == 'ipv'].pivot_table(index='label', columns='cell', values='qlike').round(5))
        print(paired[paired.comparator == 'ff3_1m'])
        print(rk.groupby('cell')['rank'].agg(['mean', 'min', 'max']))


if __name__ == '__main__':
    main()
