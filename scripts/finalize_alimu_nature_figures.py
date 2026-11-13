"""Validate editable PDF exports and seal a visually reviewed figure package.

Use any Python environment with PyMuPDF (fitz). Rendering remains Python only.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import zipfile
import fitz

LOG=logging.getLogger('alimu_nature_finalize')

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def write(p,obj):
 LOG.info('WRITE %s',p)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
 parser.add_argument('--seal',action='store_true');args=parser.parse_args()
 logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
 run=args.run_dir.resolve();root=Path(__file__).resolve().parents[1]
 if (run/'complete.json').exists():
  LOG.info('SKIP sealed presentation package %s',run);return
 LOG.info('READ %s',run/'verification.json')
 verification=json.loads((run/'verification.json').read_text(encoding='utf-8'))
 if verification['status']!='PASS':raise ValueError('Numerical/export audit must pass first')
 qa=run/'qa';qa.mkdir(exist_ok=True);checks=[]
 for path in sorted((run/'figures').glob('*.pdf')):
  LOG.info('READ PDF %s',path)
  with fitz.open(path) as doc:
   if len(doc)!=1:raise ValueError('Expected one figure page')
   page=doc[0];text=page.get_text();fonts=page.get_fonts(full=True)
   if len(text)<300 or len(fonts)<1:raise ValueError('PDF text selection failed')
   if abs(page.rect.width-7.2*72)>.02:raise ValueError('Wrong final physical width')
   if any(font[1]!='ttf' for font in fonts):raise ValueError('PDF fonts must embed TrueType')
   for font in fonts:
    if len(doc.extract_font(font[0])[3])<100:raise ValueError('PDF font not embedded')
   # A PDF renderer within the selected Python backend provides final-file QA.
   preview=qa/(path.stem+'_pdf_preview.png')
   page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(preview)
   checks.append(dict(figure=path.stem,pdf_sha256=sha(path),page_width_pt=page.rect.width,
    page_height_pt=page.rect.height,selectable_text_characters=len(text),
    embedded_true_type_fonts=len(fonts),preview=preview.name,status='PASS'))
 if len(checks)!=5:raise ValueError('Expected five PDF bundles')
 write(qa/'pdf_export_review.json',dict(status='PASS',backend='Python/PyMuPDF',figures=checks))
 for entry in verification['exports']:
  if entry['clipped_text'] or not (abs(entry['width_inches']-7.2)<1e-10):raise ValueError('Layout/export check failed')
  if entry['png_pixels'][0]!=2160 or entry['tiff_pixels'][0]!=4320:raise ValueError('Raster width failed')
  if not all(abs(v-600)<.1 for v in entry['tiff_dpi']):raise ValueError('TIFF resolution failed')
 LOG.info('SKIP training/source-run mutation; export QA passed')
 if not args.seal:
  LOG.info('SKIP seal until all PNG/PDF previews have been visually reviewed');return
 visual=run/'qa/visual_review.json'
 review=json.loads(visual.read_text(encoding='utf-8'))
 if review['status']!='PASS' or len(review['figures'])!=5:raise ValueError('Visual review not complete')
 for r in review['figures']:
  if sha(run/'figures'/(r['figure']+'.png'))!=r['png_sha256']:raise ValueError('PNG changed after visual review')
 manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'))
 for r in manifest['inputs']:
  cfg={'shared':'cnn_lstm_module_replacement_20261107','windows':'cnn_bilstm_shared_ablation_20261106','matched':'alimu_matched_ddpm_20261106_final'}
  source=root/'runs'/cfg[r['scope']]/r['file']
  if sha(source)!=r['sha256']:raise ValueError('Read-only source input changed')
 verification['visual_review']='PASS: all five final PNGs and PDF previews inspected'
 verification['pdf_export_review_sha256']=sha(qa/'pdf_export_review.json')
 verification['visual_review_sha256']=sha(visual)
 write(run/'verification.json',verification)
 snapshot=run/'plotting_source/scripts'/Path(__file__).name;snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__),snapshot)
 prov=json.loads((run/'provenance.json').read_text(encoding='utf-8'))
 prov['finalizer_source_sha256']=sha(snapshot);prov['verification_sha256']=sha(run/'verification.json')
 write(run/'provenance.json',prov)
 files={str(p.relative_to(run)).replace('\\','/'):sha(p) for p in sorted(run.rglob('*'))
        if p.is_file() and p.suffix not in ('.log','.zip') and p.name not in ('complete.json','artifact_manifest.json')}
 write(run/'artifact_manifest.json',dict(files_sha256=files,excluded='Mutable logs, ZIP, manifest and completion seal'))
 write(run/'complete.json',dict(status='complete',run_id=run.name,created_utc=datetime.now(timezone.utc).isoformat(),
  figures=5,figure_files=20,numerical_verification='PASS',visual_review='PASS',editable_pdf_svg='PASS',
  verification_sha256=sha(run/'verification.json'),artifact_manifest_sha256=sha(run/'artifact_manifest.json'),
  git_commit=prov['git_commit'],source_runs=prov['source_runs'],new_training=False))
 archive=run/'ALiMU_Nature_figures_20261108.zip'
 LOG.info('WRITE %s',archive)
 # Only this explicit generated directory is packaged; no raw data or checkpoints.
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(run.rglob('*')):
   if p.is_file() and p!=archive and p.suffix!='.log':z.write(p,p.relative_to(run))
 with zipfile.ZipFile(archive) as z:
  if z.testzip() is not None:raise ValueError('Archive integrity failed')
 LOG.info('PASS: five figures, all numerical/PDF/visual checks and ZIP integrity; bytes=%d',archive.stat().st_size)

if __name__=='__main__':main()
