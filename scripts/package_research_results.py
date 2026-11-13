"""Package sealed research reports for GitHub Releases without measurements/models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import zipfile

import yaml

LOG = logging.getLogger(__name__)
METADATA = {
    'config.yaml', 'config_snapshot.yaml', 'data_manifest.json', 'complete.json',
    'verification.json', 'independent_audit.json', 'numerical_audit.json',
    'metric_definitions.json', 'numeric_precision_audit.json', 'provenance.json',
    'visual_qa.json', 'independent_review.json', 'artifact_manifest.json',
    'deployment_latency_audit.json', 'numeric_precision_recovery.lock.json',
}
FIGURE_SUFFIXES = {'.svg', '.pdf', '.png', '.tiff', '.md', '.json'}
SECRET = re.compile(rb'gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_public_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    forbidden = {'timestamp_s', 'sample_index'}
    forbidden.update(axis+'_'+role for axis in ['gx', 'gy', 'gz', 'ax', 'ay', 'az']
                     for role in ['low', 'high', 'pred', 'virtual'])
    if forbidden.intersection(header):
        raise ValueError('Raw/per-sample measurement CSV is not public-package content: '+str(path))


def expected_hashes(run):
    result = {}
    for filename in ['artifact_manifest.json', 'complete.json', 'verification.json']:
        if not (run/filename).exists():
            continue
        data = read_json(run/filename)
        for key in ['files', 'files_sha256', 'artifacts_sha256', 'outputs_sha256']:
            values = data.get(key, {})
            if isinstance(values, dict):
                result.update({name.replace('\\', '/'): value for name, value in values.items()
                               if isinstance(value, str)})
    return result


def collect_files(run, study):
    paths = [p for p in run.iterdir() if p.is_file() and
             (p.suffix.lower() in {'.md', '.csv'} or p.name in METADATA)]
    for name in ['tables', 'source_data', 'support']:
        if (run/name).is_dir():
            paths += [p for p in (run/name).rglob('*') if p.is_file() and p.suffix in {'.csv', '.md'}]
    if study.get('figures') and (run/'figures').is_dir():
        paths += [p for p in (run/'figures').rglob('*') if p.is_file() and p.suffix.lower() in FIGURE_SUFFIXES]
    if study.get('frozen_source'):
        for name in ['source_snapshot', 'analysis_source']:
            if (run/name).is_dir():
                paths += [p for p in (run/name).rglob('*') if p.is_file() and p.suffix in {'.py', '.yaml', '.yml'}]
    return sorted(set(paths))


def build_package(source_root, output, config):
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = output/config['archive_name']
    entries, studies = [], []
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=7) as bundle:
        bundle.writestr('README.md',
            '# ALiMU research results\n\n'
            'Reports, aggregate metrics, publication figures and frozen source identities.\n'
            'Study directories preserve original filenames so report-to-figure links work locally.\n'
            'The context study numerical source is alimu_context_mechanisms_20261108; use '
            'alimu_context_mechanisms_20261109_figures_final for final context figures.\n'
            'The matched classical supplement is alimu_context_classical_20261112.\n'
            'MANIFEST.json identifies every member and its original SHA256.\n'
            'Dataset measurements, prepared tensors, predictions, fitted model states and checkpoints '
            'remain in the original local source runs.\n'
            'Frozen experiment sources in this package identify the original training implementations; '
            'the publication Git branch also incorporates the existing upstream research framework.\n')
        for study in config['studies']:
            run = (source_root/'runs'/study['run']).resolve()
            if source_root/'runs' not in run.parents:
                raise ValueError('Source run path escapes runs/')
            receipt = run/'complete.json' if (run/'complete.json').exists() else run/'verification.json'
            if not receipt.exists() or str(read_json(receipt).get('status', '')).lower() not in {'pass', 'complete', 'completed'}:
                raise ValueError('Source study lacks a completion/PASS receipt: '+study['run'])
            expected = expected_hashes(run)
            count = 0
            for path in collect_files(run, study):
                if run not in path.resolve().parents:
                    raise ValueError('Publication member escapes source run')
                relative = path.relative_to(run).as_posix()
                if path.suffix == '.csv':
                    validate_public_csv(path)
                payload = path.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                if relative in expected and digest != expected[relative]:
                    raise ValueError('Sealed artifact hash mismatch: '+study['run']+'/'+relative)
                if path.suffix in {'.py', '.yaml', '.yml', '.json', '.md', '.csv'} and SECRET.search(payload):
                    raise ValueError('Credential-like literal in publication member: '+relative)
                member = 'studies/'+study['run']+'/'+relative
                bundle.writestr(member, payload)
                entries.append(dict(member=member, source_run=study['run'], source_relative_path=relative,
                                    sha256=digest, bytes=len(payload), checked_against_seal=relative in expected))
                count += 1
            studies.append(dict(run=study['run'], status=read_json(receipt)['status'],
                                receipt=receipt.name, receipt_sha256=sha(receipt), published_files=count))
            LOG.info('READ sealed study %s; WRITE %d report/aggregate/figure/source members; SKIP measurements/models', study['run'], count)
        manifest = dict(schema_version=1, release_tag=config.get('release_tag'), studies=studies, files=entries)
        bundle.writestr('MANIFEST.json', json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    with zipfile.ZipFile(archive) as bundle:
        for item in entries:
            if hashlib.sha256(bundle.read(item['member'])).hexdigest() != item['sha256']:
                raise ValueError('Archive verification failed '+item['member'])
        if bundle.testzip() is not None:
            raise ValueError('Archive CRC verification failed')
    assets = [archive]
    for preview in config.get('preview_assets', []):
        source = source_root/'runs'/preview['source']
        relative = 'studies/'+preview['source']
        if relative not in {item['member'] for item in entries}:
            raise ValueError('Preview asset is not an already verified archive member')
        destination = output/preview['name']
        if destination.parent != output or destination.suffix.lower() != '.png':
            raise ValueError('Preview must be a simple PNG filename')
        shutil.copy2(source, destination)
        assets.append(destination)
    checksums = {asset.name: sha(asset) for asset in assets}
    (output/'SHA256SUMS.txt').write_text(''.join(value+'  '+name+'\n' for name, value in checksums.items()), encoding='utf-8')
    evidence = dict(schema_version=1, release_tag=config.get('release_tag'),
                    studies=studies, archive=dict(name=archive.name, sha256=checksums[archive.name],
                                                 bytes=archive.stat().st_size, members=len(entries)+2),
                    preview_sha256={p.name: checksums[p.name] for p in assets[1:]},
                    excluded=['dataset measurements', 'prepared tensors', 'prediction arrays',
                              'fitted states', 'checkpoints', 'training run directories in Git'])
    (output/'publication_evidence.json').write_text(json.dumps(evidence, indent=2)+'\n', encoding='utf-8')
    LOG.info('WRITE verified archive %s (%d bytes), preview assets and SHA256SUMS; SKIP uploads', archive, archive.stat().st_size)
    return dict(archive=str(archive), evidence=evidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/analysis/github_results_20261113.yaml')
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    LOG.info('READ config %s and sealed source runs; WRITE only %s; SKIP training and source mutations', args.config, args.out)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    result = build_package(args.source_root, args.out, cfg)
    print(json.dumps(result['evidence'], indent=2))
