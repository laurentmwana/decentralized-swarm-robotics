"""Contrôles documentaires : citations, renvois, figures, sources et chiffres clés."""
from pathlib import Path
import argparse
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[2]
TEX=ROOT/'latex'
files=[TEX/'main.tex',*sorted((TEX/'chapters').glob('*.tex')),
       *sorted((TEX/'frontmatter').glob('*.tex'))]
text='\n'.join(p.read_text(encoding='utf-8') for p in files)
bib=(TEX/'references.bib').read_text(encoding='utf-8')
keys=set(re.findall(r'@\w+\s*\{\s*([^,]+)',bib))
citations={key.strip() for group in re.findall(r'\\cite\{([^}]+)\}',text) for key in group.split(',')}
labels=re.findall(r'\\label\{([^}]+)\}',text)
refs=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',text))
assert len(labels)==len(set(labels)),'Labels en double'
assert citations<=keys,f'Citations absentes : {citations-keys}'
assert refs<=set(labels),f'Renvois absents : {refs-set(labels)}'
figures=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text)
for figure in figures:
    p=TEX/figure
    assert p.exists() or any(p.with_suffix(s).exists() for s in ('.png','.pdf','.jpg')),figure
for name,expected in [('training-multiform-expanded-v1','a1e3b88b39386a3d490798b5c7717e3939416e52572a058045d7402ec6cf6d84'),
                      ('training-vector-gat-v2','411d902b0fbf52d60f79ca1c115fb07c4563602d3dc2ce03f6885c4824bc952a')]:
    p=ROOT/'outputs/imitation'/name/'best.pt'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
manifest=json.loads((TEX/'generated/sources.json').read_text(encoding='utf-8'))
for relative,expected in manifest['sha256'].items():
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
report=json.loads((ROOT/'tasks/evidence/014-vector-fresh-test.json').read_text())
assert len(report['reports'])==18
assert all(r['metrics']['success'] and r['metrics']['collisions']==0 and
           r['metrics']['motion_safety_intervention_robot_steps']==0 for r in report['reports'])
reference=json.loads((ROOT/'tasks/evidence/008d-local-target-expanded-test.json').read_text())
assert reference['summary']['raw']['collisions_total']==54
assert reference['summary']['guarded']['success_rate_percent']==100
nav=json.loads((ROOT/'tasks/evidence/019-near-targets-evaluation.json').read_text())
assert all(s['episodes']==8 and s['successes']==8 and s['collisions']==0 for s in nav['summary'].values())
for forbidden in ('9\\,600','20, 40 ou 100','catalogue de 64 figures','entraîné à travers le simulateur'):
    assert forbidden not in text,forbidden
result={'status':'passed','files_checked':len(files),'citations':len(citations),
        'labels':len(labels),'figure_references':len(figures),'source_hashes_checked':len(manifest['sha256']),
        'checkpoint_hashes_checked':2,'reserved_formation_cases':18,'navigation_cases_per_controller':8,
        'note':'Contrôle documentaire ; ne remplace ni compilation ni relecture du PDF.'}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report',type=Path,default=ROOT/'latex/generated/manuscript-audit.json')
args=parser.parse_args()
out=args.report if args.report.is_absolute() else ROOT/args.report
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
