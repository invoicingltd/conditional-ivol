"""Run the estimator at the last session of every calendar month.

Same code path and configuration as the production run; records go to
the runtime ang_comparison/records directory. Completed months are skipped.
"""
from __future__ import annotations
import argparse, json, time
import multiprocessing as mp
from datetime import datetime, timezone
import numpy as np
from civol.records import completed_record_stems
from civol.core.paths import records_path, CACHE
import pandas as pd

RECORDS = records_path('ang_comparison')
_cache = None; _config = None; _limits = None


def month_end_schedule(dates, T, H, start, end):
    s = pd.Series(np.arange(len(dates)), index=pd.DatetimeIndex(dates))
    last = s.groupby([s.index.year, s.index.month]).max()
    idx = [int(t) for t in last.values if t >= T - 1 and t + H < len(dates)]
    return [t for t in idx if np.datetime64(start) <= dates[t] <= np.datetime64(end)]


def _init(threads, config):
    global _cache, _config, _limits
    from threadpoolctl import threadpool_limits
    import numba
    _limits = threadpool_limits(limits=threads); numba.set_num_threads(threads)
    from civol.core.universe import Cache
    _cache = Cache(); _config = config


def _work(t_idx):
    from civol.core.estimator import safe_run_date
    started = time.perf_counter()
    s = safe_run_date(_cache, t_idx, _config, pilot=False, out=RECORDS)
    return dict(t_idx=int(t_idx), date=s.get('date'), status=s.get('status', 'complete'), N=s.get('N'), rank=s.get('rank'), seconds=round(time.perf_counter() - started, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=6); p.add_argument('--threads', type=int, default=2); p.add_argument('--backend', default='gpu')
    p.add_argument('--start', default='1988-01-01'); p.add_argument('--end', default='2025-11-30')
    a = p.parse_args()
    from civol.core.estimator import CONFIG
    config = dict(CONFIG, backend=a.backend)
    dates = np.load(CACHE / 'dates.npy')
    sched = month_end_schedule(dates, config['T'], config['H'], a.start, a.end)
    RECORDS.mkdir(parents=True, exist_ok=True)
    done = completed_record_stems(RECORDS)
    todo = [t for t in sched if str(dates[t]) not in done]
    log = RECORDS / 'RUN_LOG.txt'
    with log.open('a', encoding='utf8') as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} start: {len(sched)} month-ends, {len(todo)} to run\n")
    started = time.perf_counter(); results = []
    with mp.get_context('spawn').Pool(a.workers, initializer=_init, initargs=(a.threads, config)) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, todo), 1):
            results.append(r); line = f"{datetime.now(timezone.utc).isoformat()} [{i}/{len(todo)}] {r['date']} {r['status']} N={r['N']} K={r['rank']} {r['seconds']}s"
            print(line, flush=True)
            with log.open('a', encoding='utf8') as f:
                f.write(line + '\n')
    (RECORDS / f"RUN_SUMMARY_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json").write_text(json.dumps(dict(
        finished_utc=datetime.now(timezone.utc).isoformat(), scheduled=len(sched), ran=len(todo), seconds=round(time.perf_counter() - started, 1),
        failed=[r for r in results if r['status'] != 'complete']), indent=1))
    print('done', len(todo), round(time.perf_counter() - started, 1), 's')


if __name__ == '__main__':
    main()
