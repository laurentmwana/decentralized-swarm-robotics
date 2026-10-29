"""Assembler les campagnes conservées sans relancer ni sélectionner d’essais."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]


def build_report(evidence=None):
    evidence=ROOT/'tasks/evidence' if evidence is None else Path(evidence)
    sources={};rows=[]
    def read(name):
        path=evidence/name;content=path.read_bytes();value=json.loads(content)
        if value.get('status')!='complete':raise ValueError('Campagne incomplète : '+name)
        sources[name]=hashlib.sha256(content).hexdigest();return value
    formations=read('008d-local-target-expanded-test.json')
    for mode,summary in formations['summary'].items():
        rows.append(dict(campaign='Formes réservées · test',scope='épisodes',mode=mode,
            trials=summary['episodes'],success_rate_percent=summary['success_rate_percent'],
            formation_error_mean_m=summary['formation_error_mean_m'],
            successful_time_mean_s=summary['stabilization_time_mean_s'],time_definition='stabilisation relative',
            collisions=summary['collisions_total']))
    transitions=read('008e-transitions-validation.json')
    for mode,summary in transitions['summary'].items():
        stage=summary['stages']
        rows.append(dict(campaign='Transitions · validation',scope=f"stages de {summary['sequences']} séquences",mode=mode,
            trials=stage['episodes'],success_rate_percent=stage['success_rate_percent'],
            formation_error_mean_m=stage['formation_error_mean_m'],
            successful_time_mean_s=stage['stabilization_time_mean_s'],time_definition='stabilisation relative par stage',
            collisions=summary['collisions_total']))
    navigation=read('009-navigation-validation.json')
    for mode in navigation['summary']:
        for group,label in [('all','ensemble'),('nominal','parcours prévus'),('stress','cas difficiles')]:
            reports=[r['modes'][mode] for r in navigation['reports'] if group=='all' or r['group']==group]
            successful=[r for r in reports if r['success']]
            rows.append(dict(campaign='Navigation · '+label,scope='épisodes',mode=mode,trials=len(reports),
                success_rate_percent=100*len(successful)/len(reports),
                formation_error_mean_m=mean(r['formation_error_m'] for r in reports),
                successful_time_mean_s=mean(r['arrival_time_s'] for r in successful) if successful else None,
                time_definition='arrivée absolue, préparation comprise',collisions=sum(r['collisions'] for r in reports)))
    return dict(schema_version=1,kind='final_reference_comparison',sources_sha256=sources,rows=rows,
        notes=['Les lignes ensemble/nominal/difficile de navigation se recouvrent ; ne pas les additionner.',
               'Les temps moyens portent uniquement sur les essais réussis ; null signifie aucune réussite.',
               'Le GAT protégé utilise le filtre classique ; le GAT brut échoue aux tests réservés.',
               'Les stages partagent des séquences et les campagnes réutilisent des graines : les essais ne sont pas tous indépendants.',
               'Les essais interactifs de démonstration sont distincts des campagnes conservées.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    report=build_report()
    (output/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    with (output/'comparison.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(report['rows'][0]));writer.writeheader();writer.writerows(report['rows'])
    print(f'{len(report["rows"])} comparaisons enregistrées dans {output}')


if __name__=='__main__':main()
