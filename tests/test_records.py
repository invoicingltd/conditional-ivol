"""Resume checks protect completed and interrupted research records."""
import pytest

from civol.records import IncompleteRecordError, completed_record_stems


def test_missing_directory_is_empty_and_stays_missing(tmp_path):
    directory = tmp_path / 'records'
    assert completed_record_stems(directory) == set()
    assert not directory.exists()


def test_only_pairs_count_as_complete(tmp_path):
    (tmp_path / '1988-01-04.json').write_bytes(b'{"status": "complete"}')
    (tmp_path / '1988-01-04.npz').write_bytes(b'preserved scientific arrays')
    (tmp_path / '1988-02-02.FAILED.json').write_bytes(b'{"error": "retained"}')
    (tmp_path / 'RUN_SUMMARY_20261022.json').write_bytes(b'{"ran": 1}')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert completed_record_stems(tmp_path) == {'1988-01-04'}
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize('suffix, missing', [('.json', '.npz'), ('.npz', '.json')])
def test_partial_record_raises_without_changing_any_files(tmp_path, suffix, missing):
    (tmp_path / ('baseline_00' + suffix)).write_bytes(b'valuable partial output')
    (tmp_path / 'baseline_01.json').write_bytes(b'{}')
    (tmp_path / 'baseline_01.npz').write_bytes(b'complete arrays')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(IncompleteRecordError, match='baseline_00') as error:
        completed_record_stems(tmp_path)
    assert f'missing {missing}' in str(error.value)
    assert 'Inspect and move the partial files' in str(error.value)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_failed_report_does_not_hide_partial_arrays(tmp_path):
    (tmp_path / 'baseline_00.FAILED.json').write_bytes(b'{"error": "interrupted write"}')
    (tmp_path / 'baseline_00.npz').write_bytes(b'partial arrays')
    with pytest.raises(IncompleteRecordError, match='missing .json: baseline_00'):
        completed_record_stems(tmp_path)
