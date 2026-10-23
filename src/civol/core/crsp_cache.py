"""Build the wide daily CRSP cache for estimation.

Two passes over the local annual CRSP (CIZ) parquet files:
  1. collect every PERMNO that is flag-eligible (ordinary US common equity on a
     major exchange, active) on at least one session in the cache window;
  2. fill wide (sessions x permnos) arrays: log return (NaN where the daily
     observation fails the quality gate), validity, eligibility flags, market
     capitalization and PERMCO.

The quality gate requires a finite simple return > -1, dlyretmissflg == 'NA',
a one-session duration code (D1-D4, DU),
dlydelflg == 'N', and dlyprevdt equal to the preceding market session.
Only formation-dated information is stored; nothing here forecasts.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from civol.core.paths import CRSP_DAILY, FACTORS_DAILY, MARKET_CALENDAR, CACHE

FLAG_COLUMNS = ['sharetype', 'securitytype', 'securitysubtype', 'usincflg', 'issuertype', 'primaryexch', 'tradingstatusflg']
ONE_SESSION = {'D1', 'D2', 'D3', 'D4', 'DU'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def eligible_flags(frame):
    return ((frame.sharetype == 'NS') & (frame.securitytype == 'EQTY') & (frame.securitysubtype == 'COM')
            & (frame.usincflg == 'Y') & frame.issuertype.isin(['CORP', 'ACOR'])
            & frame.primaryexch.isin(['N', 'A', 'Q']) & (frame.tradingstatusflg == 'A')).to_numpy()


def build(first_year, last_year, out=CACHE):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    years = list(range(first_year, last_year + 1))
    files = [CRSP_DAILY / f'crsp_daily_{y}.parquet' for y in years]
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f)
    calendar = pd.read_parquet(MARKET_CALENDAR)
    calendar['date'] = pd.to_datetime(calendar.date)
    calendar['previous_market_date'] = pd.to_datetime(calendar.previous_market_date)
    calendar = calendar.sort_values('date')
    # Pass 1: eligible permnos and the session calendar actually present in the files.
    permnos = set()
    sessions = set()
    for f in files:
        t = pq.read_table(f, columns=['permno', 'date'] + FLAG_COLUMNS).to_pandas()
        permnos.update(t.loc[eligible_flags(t), 'permno'].unique().tolist())
        sessions.update(pd.to_datetime(t.date.unique()).tolist())
        print(f'pass1 {f.name}: {len(permnos)} eligible permnos so far', flush=True)
    dates = pd.DatetimeIndex(sorted(sessions))
    cal = calendar.set_index('date').reindex(dates)
    if cal.previous_market_date.isna().any():
        raise ValueError('Session missing from market calendar')
    previous = cal.previous_market_date
    permnos = np.array(sorted(permnos), dtype=np.int64)
    col = pd.Index(permnos)
    T, N = len(dates), len(permnos)
    ret = np.full((T, N), np.nan, dtype=np.float32)
    valid = np.zeros((T, N), dtype=bool)
    elig = np.zeros((T, N), dtype=bool)
    mktcap = np.full((T, N), np.nan, dtype=np.float32)
    permco = np.zeros((T, N), dtype=np.int32)
    duplicates = 0
    sources = []
    # Pass 2: fill arrays.
    columns = ['permno', 'permco', 'date', 'ret', 'mktcap', 'dlydelflg', 'dlyretmissflg', 'dlyretdurflg', 'dlyprevdt'] + FLAG_COLUMNS
    for f in files:
        t = pq.read_table(f, columns=columns, filters=[('permno', 'in', permnos.tolist())]).to_pandas()
        t['date'] = pd.to_datetime(t.date)
        dup = t.duplicated(['permno', 'date'], keep=False)
        duplicates += int(dup.sum())
        t = t.loc[~dup]
        r = dates.get_indexer(t.date)
        c = col.get_indexer(t.permno)
        keep = (r >= 0) & (c >= 0)
        t = t.loc[keep]; r = r[keep]; c = c[keep]
        simple = pd.to_numeric(t.ret, errors='coerce').to_numpy(np.float64)
        prev_ok = pd.to_datetime(t.dlyprevdt).to_numpy() == previous.to_numpy()[r]
        good = (np.isfinite(simple) & (simple > -1) & (t.dlyretmissflg == 'NA').to_numpy()
                & t.dlyretdurflg.isin(ONE_SESSION).to_numpy() & (t.dlydelflg == 'N').to_numpy() & prev_ok)
        lr = np.full(len(simple), np.nan)
        lr[good] = np.log1p(simple[good])
        ret[r, c] = lr.astype(np.float32)
        valid[r, c] = good
        elig[r, c] = eligible_flags(t)
        mktcap[r, c] = pd.to_numeric(t.mktcap, errors='coerce').to_numpy(np.float32)
        permco[r, c] = t.permco.fillna(0).to_numpy(np.int32)
        sources.append(dict(path=str(f), sha256=sha256(f), rows=int(len(t)), year=int(f.stem[-4:])))
        print(f'pass2 {f.name}: {len(t):,} rows, valid {int(good.sum()):,}', flush=True)
    # Daily Fama-French factors aligned to the cache sessions.
    ff = pd.read_csv(FACTORS_DAILY, parse_dates=['date']).set_index('date')
    ff = ff[['Mkt-RF', 'SMB', 'HML', 'RF']].reindex(dates)
    if ff.isna().any().any():
        missing = ff.index[ff.isna().any(axis=1)]
        raise ValueError(f'Missing factor observations on {len(missing)} sessions, first {missing[0].date()}')
    np.save(out / 'dates.npy', dates.values.astype('datetime64[D]'))
    np.save(out / 'permnos.npy', permnos)
    np.save(out / 'ret.npy', ret)
    np.save(out / 'valid.npy', valid)
    np.save(out / 'elig.npy', elig)
    np.save(out / 'mktcap.npy', mktcap)
    np.save(out / 'permco.npy', permco)
    np.save(out / 'ff.npy', ff.to_numpy(np.float32))
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), first_year=first_year, last_year=last_year,
                    sessions=int(T), permnos=int(N), first_session=str(dates[0].date()), last_session=str(dates[-1].date()),
                    valid_cells=int(valid.sum()), eligible_cells=int(elig.sum()), duplicate_rows_dropped=duplicates,
                    factors=dict(path=str(FACTORS_DAILY), sha256=sha256(FACTORS_DAILY), columns=['Mkt-RF', 'SMB', 'HML', 'RF']),
                    calendar=dict(path=str(MARKET_CALENDAR), sha256=sha256(MARKET_CALENDAR)),
                    quality_gate='finite simple return > -1; dlyretmissflg NA; duration D1-D4/DU; dlydelflg N; dlyprevdt == previous market session',
                    eligibility_flags='NS/EQTY/COM, usincflg Y, issuertype CORP|ACOR, primaryexch N|A|Q, tradingstatusflg A',
                    sources=sources, seconds=time.perf_counter() - started)
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ('sessions', 'permnos', 'valid_cells', 'eligible_cells', 'duplicate_rows_dropped', 'seconds')}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--first-year', type=int, default=1985)
    p.add_argument('--last-year', type=int, default=2025)
    a = p.parse_args()
    build(a.first_year, a.last_year)


if __name__ == '__main__':
    main()
