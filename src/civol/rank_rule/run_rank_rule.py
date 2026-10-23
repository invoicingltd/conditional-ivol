"""Run rank-rule estimation on the application schedule in parallel.
Formation dates are every 21st session from 1988 through 2025.

Each date writes records/DATE.npz + DATE.json or DATE.FAILED.json. Re-running
skips completed dates; nothing is overwritten.
    python -m civol.rank_rule.run_rank_rule --workers 6 --threads 2 --backend gpu
"""
from __future__ import annotations
import argparse, json, time
import multiprocessing as mp
from datetime import datetime, timezone
import numpy as np
from civol.records import completed_record_stems
from civol.core.paths import CACHE


_cache = None
_limits = None
_config = None


def schedule(dates, T, H, start, end):
    """Identical to application/run_application.py: first session of `start`
    with T - 1 prior sessions, then every 21st session with H sessions after it."""
    first = np.flatnonzero((dates >= np.datetime64(start)) & (np.arange(len(dates)) >= T - 1))[0]
    return [t for t in range(first, len(dates) - H, 21) if dates[t] <= np.datetime64(end)]


def _init(threads, config):
    global _cache, _limits, _config
    from threadpoolctl import threadpool_limits
    import numba
    _limits = threadpool_limits(limits=threads)
    numba.set_num_threads(threads)
    from civol.core.universe import Cache
    _cache = Cache()
    _config = config


def _work(t_idx):
    from civol.rank_rule.estimate_rank_rule import safe_run_date, RECORDS
    started = time.perf_counter()
    summary = safe_run_date(_cache, t_idx, _config, out=RECORDS)
    rules = summary.get('rules') or {}
    return dict(t_idx=int(t_idx), date=summary.get('date'), status=summary.get('status', 'complete'), N=summary.get('N'),
                rank_a=rules.get('rank_a'), rank_b=rules.get('rank_b'), seconds=round(time.perf_counter() - started, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--backend', default='gpu')
    p.add_argument('--start', default='1988-01-01')
    p.add_argument('--end', default='2025-12-31')
    p.add_argument('--limit', type=int, default=0, help='run at most this many dates (0 = all)')
    a = p.parse_args()
    from civol.rank_rule.estimate_rank_rule import CONFIG, RECORDS
    config = dict(CONFIG, backend=a.backend)
    dates = np.load(CACHE / 'dates.npy')
    sched = schedule(dates, config['T'], config['H'], a.start, a.end)
    RECORDS.mkdir(parents=True, exist_ok=True)
    done = completed_record_stems(RECORDS)
    todo = [t for t in sched if str(dates[t]) not in done]
    if a.limit:
        todo = todo[:a.limit]
    log = RECORDS / 'RUN_LOG.txt'
    with log.open('a', encoding='utf8') as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} start: {len(sched)} scheduled, {len(done)} complete, {len(todo)} to run, workers={a.workers}, threads={a.threads}, backend={a.backend}\n")
    started = time.perf_counter(); results = []
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.threads, config)) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, todo), 1):
            results.append(r)
            line = f"{datetime.now(timezone.utc).isoformat()} [{i}/{len(todo)}] {r['date']} {r['status']} N={r['N']} KA={r['rank_a']} KB={r['rank_b']} {r['seconds']}s"
            print(line, flush=True)
            with log.open('a', encoding='utf8') as f:
                f.write(line + '\n')
    summary = dict(finished_utc=datetime.now(timezone.utc).isoformat(), scheduled=len(sched), ran=len(todo), seconds=round(time.perf_counter() - started, 1),
                   failed=[r for r in results if r['status'] != 'complete'], workers=a.workers, threads=a.threads, backend=a.backend,
                   first=str(dates[sched[0]]), last=str(dates[sched[-1]]))
    (RECORDS / f"RUN_SUMMARY_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ('scheduled', 'ran', 'seconds')}), 'failed:', len(summary['failed']))


if __name__ == '__main__':
    main()
