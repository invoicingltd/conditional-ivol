"""Single command interface; configure paths before importing the workflows."""
from __future__ import annotations

import argparse
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import platform
import sys


COMMANDS = {
    'download': ('civol.data.download', 'Download CRSP CIZ and public daily factors'),
    'cache': ('civol.core.crsp_cache', 'Build the wide daily input cache'),
    'application': ('civol.application.run_application', 'Estimate the historical 21-session application'),
    'analyze-application': ('civol.application.analyze', 'Rebuild application tables and figures'),
    'simulation': ('civol.simulation.run_simulation', 'Run the four truth-known simulation cells'),
    'analyze-simulation': ('civol.simulation.analyze_simulation', 'Rebuild simulation tables and figures'),
    'rank-rule': ('civol.rank_rule.run_rank_rule', 'Estimate rank rules A/B and K=20'),
    'analyze-rank-rule': ('civol.rank_rule.analyze_rank_rule', 'Compare rank-rule results'),
    'puzzle': ('civol.puzzle.run_puzzle', 'Analyze the IVOL-return relationship'),
    'monthend': ('civol.ang_comparison.run_monthend', 'Estimate calendar-month-end records'),
    'analyze-ang': ('civol.ang_comparison.analyze_ang', 'Run the Ang comparison (anchor, then compare)'),
    'encompassing': ('civol.encompassing.run_encompassing', 'Analyze forecast encompassing and combinations'),
    'verify-results': ('civol.validation', 'Replay saved records and compare all result tables'),
}
NO_ARGUMENTS = {'analyze-application', 'analyze-simulation', 'puzzle', 'encompassing'}


def doctor():
    """Report paths and versions without reading observations or connecting to WRDS."""
    from civol.core import paths
    versions = {}
    for name in ('civol-research', 'numpy', 'scipy', 'pandas', 'numba', 'pyarrow', 'matplotlib', 'threadpoolctl', 'wrds'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = 'not installed'
    print(json.dumps({
        'python': sys.version,
        'platform': platform.platform(),
        'versions': versions,
        'paths': {name: str(getattr(paths, name)) for name in
                  ('DATA', 'WORK', 'CRSP_DAILY', 'FACTORS_DAILY', 'MARKET_CALENDAR', 'CACHE', 'GPU_PYTHON', 'RECORDS', 'SIM_RECORDS')},
        'cache_manifest_exists': (paths.CACHE / 'MANIFEST.json').is_file(),
    }, indent=2))


def main(argv=None):
    descriptions = '\n'.join(f'  {key:22} {description}' for key, (_, description) in COMMANDS.items())
    parser = argparse.ArgumentParser(
        prog='civol', description='Conditional IVOL replication workflows.',
        epilog=descriptions + '\n  doctor                 Show configured paths and installed versions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for option in ('data-dir', 'work-dir', 'cache-dir', 'gpu-python'):
        parser.add_argument('--' + option, help='Override CIVOL_' + option.upper().replace('-', '_'))
    parser.add_argument('command', nargs='?', choices=[*COMMANDS, 'doctor'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER, help='Arguments passed to the selected workflow')
    args = parser.parse_args(argv)
    for option in ('data_dir', 'work_dir', 'cache_dir', 'gpu_python'):
        value = getattr(args, option)
        if value is not None:
            os.environ['CIVOL_' + option.upper()] = str(Path(value).expanduser().resolve())
    if args.command is None:
        parser.print_help()
        return
    if args.command == 'doctor':
        if args.arguments:
            parser.error('doctor takes no arguments')
        doctor()
        return
    if args.command in NO_ARGUMENTS and args.arguments:
        if args.arguments in (['--help'], ['-h']):
            print(f'usage: civol {args.command}\n\n{COMMANDS[args.command][1]}.\nPaths are configured with CIVOL_* environment variables or global options.')
            return
        parser.error(f'{args.command} takes no arguments')
    module = import_module(COMMANDS[args.command][0])
    old_argv = sys.argv
    try:
        sys.argv = [f'civol {args.command}', *args.arguments]
        module.main()
    finally:
        sys.argv = old_argv
