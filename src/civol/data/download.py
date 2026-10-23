"""Download annual CRSP CIZ extracts and daily Fama-French inputs.

The factor CSV supplies decimal returns; the daily TXT series supplies market
calendar dates. Return-quality and duplicate filters are applied by the cache
builder, and formation eligibility is checked by the universe selector.

WRDS credentials use wrds.Connection. Completed downloads are reused, and
acquisition manifests record query parameters, retrieval times, and hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
from urllib.request import urlopen
import zipfile

import numpy as np
import pandas as pd

from civol.core.paths import CRSP_DAILY, FACTORS_DAILY, MARKET_CALENDAR


CRSP_COLUMNS = (
    "permno", "permco", "dlycaldt", "dlyret", "dlyprc", "shrout", "dlycap",
    "dlyvol", "ticker", "sharetype", "securitytype", "securitysubtype",
    "usincflg", "issuertype", "primaryexch", "conditionaltype",
    "tradingstatusflg", "dlydelflg", "dlyprcflg", "dlyretmissflg",
    "dlyretdurflg", "dlyprevdt",
)
CRSP_DAILY_SQL = """
    SELECT
        permno, permco, dlycaldt, dlyret, dlyprc, shrout, dlycap,
        dlyvol, ticker, sharetype, securitytype, securitysubtype,
        usincflg, issuertype, primaryexch, conditionaltype,
        tradingstatusflg, dlydelflg, dlyprcflg, dlyretmissflg,
        dlyretdurflg, dlyprevdt
    FROM crsp.dsf_v2
    WHERE dlycaldt BETWEEN %(start_date)s AND %(end_date)s
    ORDER BY dlycaldt, permno
"""
CRSP_RENAMES = {
    "dlycaldt": "date", "dlyret": "ret", "dlyprc": "prc", "dlycap": "mktcap",
}
FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FACTOR_CSV_URL = FRENCH_BASE + "F-F_Research_Data_Factors_daily_CSV.zip"
CALENDAR_TXT_URL = FRENCH_BASE + "F-F_Research_Data_Factors_daily_TXT.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".manifest.json")


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _record(path: Path, **metadata) -> dict:
    record = dict(
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        file=path.name, bytes=path.stat().st_size, sha256=sha256(path), **metadata,
    )
    _write_json(_sidecar(path), record)
    return record


def _existing(path: Path, request: dict | None = None) -> bool:
    if not path.exists():
        if _sidecar(path).exists():
            raise FileNotFoundError(f"Manifest exists without its data file: {path}")
        return False
    manifest_path = _sidecar(path)
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["sha256"] != sha256(path):
            raise ValueError(f"Completed input has changed: {path}")
        if request is not None and saved.get("request") != request:
            raise ValueError(f"Existing input has a different request: {path}")
        print(f"Verified existing input: {path}", flush=True)
    else:
        # Files without manifests can be reused, but their acquisition date is unknown.
        print(f"Using existing input without acquisition manifest: {path}", flush=True)
    return True


def _save_new(path: Path, writer) -> None:
    """Publish a finished file only; leave failed partials visibly incomplete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"Refusing to replace an existing file or partial: {path}")
    writer(partial)
    # rename, rather than replace, also protects existing destinations on Windows.
    if path.exists():
        raise FileExistsError(path)
    partial.rename(path)


def transform_crsp(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply column aliases and integer PERMNOs; retain row order, missingness and units."""
    missing = set(CRSP_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"CRSP response is missing requested fields: {sorted(missing)}")
    result = frame.rename(columns=CRSP_RENAMES).copy()
    result["permno"] = result["permno"].astype("int64")
    return result


def download_crsp_year(year: int, db, output_dir: Path = CRSP_DAILY) -> Path:
    output_path = Path(output_dir) / f"crsp_daily_{year}.parquet"
    params = {"start_date": f"{year}-01-01", "end_date": f"{year}-12-31"}
    request = dict(sql=CRSP_DAILY_SQL, params=params, date_cols=["dlycaldt", "dlyprevdt"])
    if _existing(output_path, request):
        return output_path
    frame = transform_crsp(db.raw_sql(CRSP_DAILY_SQL, params=params, date_cols=request["date_cols"]))
    if frame.empty:
        raise ValueError(f"CRSP returned no rows for {year}; no completed file was written")
    _save_new(output_path, lambda p: frame.to_parquet(p, index=False, compression="zstd"))
    _record(output_path, request=request, rows=len(frame),
            first_date=str(frame["date"].min().date()), last_date=str(frame["date"].max().date()),
            columns=list(frame.columns), dtypes={k: str(v) for k, v in frame.dtypes.items()})
    print(f"{year}: saved {len(frame):,} rows", flush=True)
    return output_path


def _zip_text(payload: bytes, suffix: str) -> str:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(suffix)]
        if len(names) != 1:
            raise ValueError(f"Expected one {suffix} file in the French archive")
        return archive.read(names[0]).decode("utf-8-sig")


def parse_factor_csv(payload: bytes) -> pd.DataFrame:
    """Parse daily Fama-French CSV returns and convert percentages to decimals."""
    lines = _zip_text(payload, ".csv").splitlines()
    header = next(line for line in lines if "Mkt-RF" in line)
    rows = [line for line in lines if re.match(r"^\s*\d{8},", line)]
    frame = pd.read_csv(io.StringIO(header + "\n" + "\n".join(rows)))
    frame.columns = ["date"] + [c.strip() for c in frame.columns[1:]]
    frame["date"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d")
    frame.iloc[:, 1:] = frame.iloc[:, 1:] / 100
    if frame.empty or frame.date.duplicated().any() or frame.iloc[:, 1:].min().min() <= -0.99:
        raise ValueError("Invalid French daily CSV dates or return scale")
    return frame


def parse_calendar_factors(payload: bytes) -> pd.DataFrame:
    """Parse daily Fama-French TXT returns and dates for the market calendar."""
    rows = []
    for line in _zip_text(payload, ".txt").splitlines():
        fields = line.split()
        if len(fields) == 5 and re.fullmatch(r"\d{8}", fields[0]):
            rows.append([fields[0]] + [float(value) for value in fields[1:]])
    names = ["mkt_rf", "smb", "hml", "rf"]
    frame = pd.DataFrame(rows, columns=["date"] + names)
    frame["date"] = pd.to_datetime(frame.date, format="%Y%m%d")
    frame[names] = frame[names].mask(frame[names].isin([-99.99, -999])) / 100.
    if (frame.empty or frame.date.duplicated().any() or not frame.date.is_monotonic_increasing
            or not np.isfinite(frame[names].to_numpy()).all() or frame[names].abs().max().max() >= 2):
        raise ValueError("Invalid French daily TXT dates or return scale")
    return frame


def build_calendar(daily_factors_full: pd.DataFrame, first_year: int, last_year: int) -> pd.DataFrame:
    calendar = daily_factors_full[["date"]].copy()
    # Shift before restricting the requested window: January's first session
    # must retain the previous December session in the cache's quality gate.
    calendar["previous_market_date"] = calendar.date.shift()
    calendar = calendar[calendar.date.between(f"{first_year}-01-01", f"{last_year}-12-31")].copy()
    calendar["month"] = calendar.date.dt.to_period("M").dt.to_timestamp("M")
    if calendar.empty or calendar.previous_market_date.isna().any():
        raise ValueError("Calendar must include dates before the requested first session")
    return calendar


def validate_calendar_coverage(calendar: pd.DataFrame, factor_dates: pd.Series,
                               first_year: int, last_year: int) -> None:
    """Reject an old narrow calendar instead of silently reusing it for new years."""
    all_dates = pd.DatetimeIndex(pd.to_datetime(factor_dates))
    expected = all_dates[(all_dates >= f"{first_year}-01-01") & (all_dates <= f"{last_year}-12-31")]
    missing_years = sorted(set(range(first_year, last_year + 1)).difference(expected.year))
    if missing_years:
        raise ValueError(f"Factor source does not cover requested years {missing_years}; use a new data directory for a new vintage")
    actual = pd.DatetimeIndex(pd.to_datetime(calendar["date"]))
    absent = expected.difference(actual)
    if len(absent):
        raise ValueError(
            f"Market calendar does not cover {len(absent)} requested sessions "
            f"({absent[0].date()} through {absent[-1].date()}); "
            "use a new CIVOL_CALENDAR_FILE or data directory to build the expanded calendar"
        )
    if actual.has_duplicates:
        raise ValueError("Market calendar contains duplicate sessions")
    previous = pd.Series(all_dates, index=all_dates).shift().reindex(expected)
    saved = calendar.assign(date=actual).set_index("date")["previous_market_date"].reindex(expected)
    if not np.array_equal(pd.to_datetime(saved).to_numpy(), previous.to_numpy()):
        raise ValueError("Market calendar previous-session dates disagree with the factor calendar")


def _download_archive(url: str, directory: Path) -> tuple[Path, bytes]:
    path = directory / url.rsplit("/", 1)[-1]
    if _existing(path, {"url": url}):
        return path, path.read_bytes()
    with urlopen(url, timeout=90) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Corrupt French factor archive: {url}")
    _save_new(path, lambda p: p.write_bytes(payload))
    _record(path, request={"url": url})
    return path, payload


def download_factors(first_year: int, last_year: int) -> None:
    sources = FACTORS_DAILY.parent / "sources"
    if not _existing(FACTORS_DAILY):
        archive, payload = _download_archive(FACTOR_CSV_URL, sources)
        frame = parse_factor_csv(payload)
        _save_new(FACTORS_DAILY, lambda p: frame.to_csv(p, index=False))
        _record(FACTORS_DAILY, url=FACTOR_CSV_URL, source_sha256=sha256(archive),
                rows=len(frame), units="decimal returns")
    factor_dates = pd.read_csv(FACTORS_DAILY, usecols=["date"], parse_dates=["date"])["date"]
    if _existing(MARKET_CALENDAR):
        calendar = pd.read_parquet(MARKET_CALENDAR, columns=["date", "previous_market_date"])
        validate_calendar_coverage(calendar, factor_dates, first_year, last_year)
    else:
        archive, payload = _download_archive(CALENDAR_TXT_URL, sources)
        calendar = build_calendar(parse_calendar_factors(payload), first_year, last_year)
        validate_calendar_coverage(calendar, factor_dates, first_year, last_year)
        _save_new(MARKET_CALENDAR, lambda p: calendar.to_parquet(p, index=False))
        _record(MARKET_CALENDAR, url=CALENDAR_TXT_URL, source_sha256=sha256(archive),
                first_year=first_year, last_year=last_year, rows=len(calendar))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-year", type=int, default=1985)
    parser.add_argument("--last-year", type=int, default=2025)
    parser.add_argument("--username", default=os.environ.get("WRDS_USERNAME"),
                        help="WRDS username; defaults to WRDS_USERNAME or WRDS's normal prompt")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--factors-only", action="store_true")
    selection.add_argument("--crsp-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print SQL, dates and destinations without connecting")
    args = parser.parse_args(argv)
    if not 1926 <= args.first_year <= args.last_year <= 9999:
        parser.error("Expected 1926 <= first-year <= last-year <= 9999")
    if args.dry_run:
        print(json.dumps(dict(
            sql=CRSP_DAILY_SQL, first_year=args.first_year, last_year=args.last_year,
            crsp_directory=str(CRSP_DAILY), factors_file=str(FACTORS_DAILY),
            calendar_file=str(MARKET_CALENDAR), factor_url=FACTOR_CSV_URL, calendar_url=CALENDAR_TXT_URL,
            raw_transform=CRSP_RENAMES, filters="Date range only; universe/quality gates occur in the cache builder",
        ), indent=2))
        return
    if not args.factors_only:
        years = list(range(args.first_year, args.last_year + 1))
        missing = [year for year in years if not (CRSP_DAILY / f"crsp_daily_{year}.parquet").exists()]
        if missing:
            import wrds
            options = {"wrds_username": args.username} if args.username else {}
            with wrds.Connection(**options) as db:
                for year in years:
                    download_crsp_year(year, db)
        else:
            for year in years:
                download_crsp_year(year, None)
    if not args.crsp_only:
        download_factors(args.first_year, args.last_year)


if __name__ == "__main__":
    main()
