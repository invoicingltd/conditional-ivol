"""Simulation with known truth: 4 designs x 8 economies at N=2000, T=750.

Every economy writes CELL_REP.npz (truth and every forecast) and CELL_REP.json.
Completed economies are skipped on rerun; nothing is overwritten.
"""
from __future__ import annotations
import argparse, json, time, traceback
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from civol.records import completed_record_stems

from civol.core.paths import SIM_RECORDS
from civol.simulation.population import CELLS

REPS = 8
_config = None


def _init(threads, config):
    global _config
    from threadpoolctl import threadpool_limits
    import numba
    globals()['_limits'] = threadpool_limits(limits=threads)
    numba.set_num_threads(threads)
    _config = config


def _work(job):
    cell, rep = job
    from civol.simulation.population import simulate, truth
    from civol.core.estimator import estimate_panel, source_hashes
    started = time.perf_counter(); stamp = datetime.now(timezone.utc).isoformat()
    try:
        sim = simulate(cell, rep, N=_config['N'], T=_config['T'], H=_config['H'], transitory=_config['transitory_driver'], level=_config['level_driver'])
        tr = truth(sim, pairs=_config['truth_pairs'], batches=16, seed=1000 + rep, backend=_config['backend'])
        est = estimate_panel(sim['x'], sim['ff'], sim['rf'], seed=20000 + 100 * CELLS.index(cell) + rep, config=_config)
        rec = dict(est['forecasts'])
        rec.update(q_true=tr['q'], ipv_true=tr['ipv'], q_true_mcse=tr['q_mcse'], ipv_true_mcse=tr['ipv_mcse'],
                   total_variance_true=tr['total_variance'], d_true=sim['parameters']['d'], s_true=sim['parameters']['s'],
                   coefficients_true=sim['parameters']['coefficients'], h_next_true=sim['truth_model']['h_next'],
                   loadings_true=sim['parameters']['loadings'])
        rec.update({f'issuer_{k}': v for k, v in est['issuer'].items()})
        summary = est['summary']
        summary.update(cell=cell, rep=rep, started_utc=stamp, config=dict(_config), source_hashes=source_hashes(),
                       truth=dict(pairs=tr['pairs'], backend=tr.get('backend'), median_relative_q_mcse=float(np.median(tr['q_mcse'] / tr['q'])),
                                  max_relative_q_mcse=float(np.max(tr['q_mcse'] / tr['q'])), lower_contacts=tr['lower_contacts'], upper_contacts=tr['upper_contacts']),
                       seconds=round(time.perf_counter() - started, 1))
        SIM_RECORDS.mkdir(parents=True, exist_ok=True)
        stem = SIM_RECORDS / f'{cell}_{rep:02d}'
        np.savez_compressed(str(stem) + '.npz', **{k: np.asarray(v) for k, v in rec.items()})
        tmp = Path(str(stem) + '.json.tmp'); tmp.write_text(json.dumps(summary, indent=1, default=_plain)); tmp.replace(str(stem) + '.json')
        return dict(cell=cell, rep=rep, status='complete', rank=summary['rank'], seconds=summary['seconds'])
    except Exception as exc:
        SIM_RECORDS.mkdir(parents=True, exist_ok=True)
        (SIM_RECORDS / f'{cell}_{rep:02d}.FAILED.json').write_text(json.dumps(dict(cell=cell, rep=rep, error_type=type(exc).__name__, error=str(exc),
                                                                                    traceback=traceback.format_exc(), utc=stamp), indent=1))
        return dict(cell=cell, rep=rep, status='failed', error=str(exc)[:200], seconds=round(time.perf_counter() - started, 1))


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (set, tuple)):
        return list(v)
    raise TypeError(str(type(v)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--backend', default='gpu')
    p.add_argument('--N', type=int, default=2000)
    p.add_argument('--reps', type=int, default=REPS)
    p.add_argument('--cells', default=','.join(CELLS))
    a = p.parse_args()
    from civol.core.estimator import CONFIG
    config = dict(CONFIG, backend=a.backend, N=a.N, truth_pairs=65536)
    cells = a.cells.split(',')
    SIM_RECORDS.mkdir(parents=True, exist_ok=True)
    done = completed_record_stems(SIM_RECORDS)
    jobs = [(c, r) for c in cells for r in range(a.reps) if f'{c}_{r:02d}' not in done]
    log = SIM_RECORDS / 'RUN_LOG.txt'
    with log.open('a', encoding='utf8') as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} start: {len(jobs)} economies, workers={a.workers}, backend={a.backend}, N={a.N}\n")
    started = time.perf_counter()
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.threads, config)) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, jobs), 1):
            line = f"{datetime.now(timezone.utc).isoformat()} [{i}/{len(jobs)}] {r['cell']} rep {r['rep']} {r['status']} K={r.get('rank')} {r['seconds']}s {r.get('error', '')}"
            print(line, flush=True)
            with log.open('a', encoding='utf8') as f:
                f.write(line + '\n')
    print('done in', round(time.perf_counter() - started, 1), 's')


if __name__ == '__main__':
    main()
