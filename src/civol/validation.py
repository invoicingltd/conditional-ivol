"""Replay archived estimation records and compare the resulting analysis tables.

Requires an archive containing thesis_method/ with its cache, records, and
saved results. Replayed outputs are written to a separate directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


STAGES = {
    'application': [('analyze-application',)],
    'simulation': [('analyze-simulation',)],
    'rank_rule': [('analyze-rank-rule', '--quiet')],
    'puzzle': [('puzzle',)],
    'encompassing': [('encompassing',)],
    'ang_comparison': [('analyze-ang', 'anchor'), ('analyze-ang', 'compare')],
}


def _equal_json(left, right, rtol=0, atol=0):
    if isinstance(left, dict) and isinstance(right, dict):
        # Wall time for the current analysis is not a research result. Retain
        # stored estimation times such as seconds_per_date_mean in comparison.
        keys_left = set(left) - {'seconds'}
        keys_right = set(right) - {'seconds'}
        return keys_left == keys_right and all(_equal_json(left[k], right[k], rtol, atol) for k in keys_left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equal_json(a, b, rtol, atol) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    if type(left) in (int, float) and type(right) in (int, float):
        if type(left) is int and type(right) is int:
            return left == right
        return math.isclose(left, right, rel_tol=rtol, abs_tol=atol)
    return left == right


def compare_results(expected, actual, rtol=0, atol=0):
    """Compare complete table inventories, including row/column order and NaNs."""
    import pandas as pd
    suffixes = {'.csv', '.json', '.parquet'}
    def inventory(folder):
        return {p.name: p for p in folder.iterdir()
                if p.suffix in suffixes and not (p.suffix == '.json' and p.name.startswith('RUN_MANIFEST'))}
    baseline = inventory(expected)
    replay = inventory(actual)
    errors = []
    if not baseline:
        errors.append('No baseline result tables found')
    for name in sorted(baseline.keys() - replay.keys()):
        errors.append(f'Missing replay: {name}')
    for name in sorted(replay.keys() - baseline.keys()):
        errors.append(f'Unexpected replay: {name}')
    checked = 0
    for name in sorted(baseline.keys() & replay.keys()):
        a, b = baseline[name], replay[name]
        try:
            if a.suffix == '.json':
                if not _equal_json(json.loads(a.read_text()), json.loads(b.read_text()), rtol, atol):
                    raise AssertionError('JSON values differ (excluding current-analysis seconds)')
            elif a.suffix == '.csv':
                if rtol == 0 and atol == 0:
                    pd.testing.assert_frame_equal(pd.read_csv(a, dtype=str, keep_default_na=False),
                                                  pd.read_csv(b, dtype=str, keep_default_na=False), check_exact=True)
                else:
                    # Round-trip conversion avoids parser rounding obscuring a difference.
                    left = pd.read_csv(a, float_precision='round_trip')
                    right = pd.read_csv(b, float_precision='round_trip')
                    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=rtol, atol=atol)
            else:
                pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b),
                                              check_exact=(rtol == 0 and atol == 0), rtol=rtol, atol=atol)
            checked += 1
        except (AssertionError, ValueError) as exc:
            errors.append(f'{name}: {str(exc)[:500]}')
    return {'matched': checked, 'baseline_tables': len(baseline), 'replay_tables': len(replay),
            'rtol': rtol, 'atol': atol, 'errors': errors}


def main():
    from civol.core.paths import WORK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', type=Path, required=True, help='Research archive directory containing thesis_method/')
    parser.add_argument('--out', type=Path, default=WORK / 'verification', help='New, separate replay directory')
    parser.add_argument('--stages', nargs='+', choices=list(STAGES), default=list(STAGES))
    parser.add_argument('--rtol', type=float, default=0, help='Explicit relative tolerance; default is exact')
    parser.add_argument('--atol', type=float, default=0, help='Explicit absolute tolerance; default is exact')
    args = parser.parse_args()
    if not (math.isfinite(args.rtol) and math.isfinite(args.atol)) or args.rtol < 0 or args.atol < 0:
        parser.error('Tolerances must be finite and nonnegative')
    legacy = args.legacy_root.resolve() / 'thesis_method'
    out = args.out.resolve()
    if out == legacy or legacy in out.parents:
        parser.error('--out must be outside the original thesis_method tree')
    if out.exists() and any(out.iterdir()):
        parser.error('--out must be empty so stale replay files cannot pass verification')
    if not (legacy / 'application' / 'cache' / 'MANIFEST.json').is_file():
        parser.error('Original application/cache/MANIFEST.json is required')
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CIVOL_WORK_DIR=str(out), CIVOL_CACHE_DIR=str(legacy / 'application' / 'cache'))
    for stage in ('application', 'simulation', 'rank_rule', 'ang_comparison'):
        env[f'CIVOL_{stage.upper()}_RECORDS'] = str(legacy / stage / 'records')
    report = {}
    for stage in args.stages:
        print(f'Replaying {stage}...', flush=True)
        for index, command in enumerate(STAGES[stage]):
            log_path = out / f'{stage}_{index}.log'
            with log_path.open('w', encoding='utf-8') as log:
                process = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'civol', *command],
                                         env=env, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f'{stage} failed; inspect {log_path}')
        result = compare_results(legacy / stage / 'results', out / stage / 'results', args.rtol, args.atol)
        report[stage] = result
        (out / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f"{stage}: {result['matched']}/{result['baseline_tables']} tables match; {len(result['errors'])} differences", flush=True)
    if any(result['errors'] for result in report.values()):
        raise SystemExit(1)
    print('All replayed result tables match the saved research results.', flush=True)
