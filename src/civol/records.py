"""Validate resumable records without altering completed or partial artifacts."""
from pathlib import Path


class IncompleteRecordError(RuntimeError):
    """A record has only one of its required JSON summary and NPZ array files."""


def completed_record_stems(directory):
    """Return complete record names, raising if a partial record needs attention.

    A successful record consists of matching ``NAME.json`` and ``NAME.npz``.
    Run summaries and retained failure reports do not mark work complete.
    Refuse to overwrite an orphaned summary or array file: the caller must
    inspect and move the partial artifacts before retrying that record.
    This function is read-only and permits a directory that does not exist yet.
    """
    directory = Path(directory)
    summaries = {
        path.stem for path in directory.glob('*.json')
        if path.is_file()
        and not path.name.endswith('.FAILED.json')
        and not path.name.startswith('RUN')
    }
    arrays = {path.stem for path in directory.glob('*.npz') if path.is_file()}
    missing_arrays = sorted(summaries - arrays)
    missing_summaries = sorted(arrays - summaries)
    if missing_arrays or missing_summaries:
        details = []
        if missing_arrays:
            details.append('missing .npz: ' + ', '.join(missing_arrays))
        if missing_summaries:
            details.append('missing .json: ' + ', '.join(missing_summaries))
        raise IncompleteRecordError(
            f'Incomplete records in {directory}: {"; ".join(details)}. '
            'No records were overwritten. Inspect and move the partial files '
            'before rerunning.'
        )
    return summaries
