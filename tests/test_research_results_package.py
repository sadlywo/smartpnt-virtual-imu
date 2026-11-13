"""Publication packages contain reports, aggregates and figures, not measurements."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/package_research_results.py'


def module():
    spec = importlib.util.spec_from_file_location('research_package', SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_measurement_csv_is_rejected_even_under_an_aggregate_filename(tmp_path):
    path = tmp_path/'summary.csv'
    path.write_text('timestamp_s,gx_low,gx_high\n0,1,2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='measurement'):
        module().validate_public_csv(path)


def test_sealed_report_hash_mismatch_aborts_publication(tmp_path):
    root = tmp_path/'runs'/'study'
    root.mkdir(parents=True)
    (root/'report.md').write_text('changed', encoding='utf-8')
    (root/'complete.json').write_text(json.dumps({'status': 'PASS'}), encoding='utf-8')
    (root/'artifact_manifest.json').write_text(json.dumps({'files': {'report.md': '0'*64}}), encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        module().build_package(tmp_path, tmp_path/'release', {'archive_name': 'results.zip',
                               'studies': [{'run': 'study', 'figures': True, 'frozen_source': True}],
                               'preview_assets': []})


def test_archive_excludes_models_predictions_and_nested_archives(tmp_path):
    root = tmp_path/'runs'/'study'
    (root/'figures').mkdir(parents=True)
    (root/'predictions').mkdir()
    (root/'complete.json').write_text('{"status":"PASS"}', encoding='utf-8')
    (root/'report.md').write_text('results', encoding='utf-8')
    (root/'summary.csv').write_text('family,rmse\nFIR,0.1\n', encoding='utf-8')
    (root/'figures'/'plot.png').write_bytes(b'figure')
    for name in ['best.ckpt', 'archive.zip', 'predictions/raw.parquet']:
        (root/name).write_bytes(b'private measurements/model')
    result = module().build_package(tmp_path, tmp_path/'release', {
        'archive_name': 'results.zip', 'studies': [{'run': 'study', 'figures': True,
                                                 'frozen_source': False}], 'preview_assets': []})
    with zipfile.ZipFile(result['archive']) as archive:
        names = archive.namelist()
    assert 'studies/study/report.md' in names
    assert 'studies/study/figures/plot.png' in names
    assert not any(name.endswith(('.ckpt', '.parquet', '.zip')) for name in names)
    assert not any('/predictions/' in name for name in names)
