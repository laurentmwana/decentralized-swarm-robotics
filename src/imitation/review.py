"""Créer une version revue du jeu, sans modifier la collecte brute."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from src.reference.metrics import summarize
from .data import load_episode, load_manifest


def review_dataset(source, output):
    source, output = Path(source), Path(output)
    manifest = load_manifest(source, allow_quality_failures=True)
    # Un échec hors entraînement ne doit pas servir à rendre le test plus facile.
    failures = [e for e in manifest['episodes'] if not e['quality']['success']]
    if any(e['split'] != 'train' for e in failures):
        raise ValueError('La revue ne peut exclure aucun épisode de validation ou de test')
    kept = [e for e in manifest['episodes'] if e['quality']['success']]
    if not any(e['split'] == 'train' for e in kept):
        raise ValueError('Aucun épisode train utilisable')
    # Vérifier aussi les fichiers exclus, pour rendre la provenance vérifiable.
    for entry in manifest['episodes']:
        load_episode(source, entry, entry['split'])
    source_hash = hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    manifest['status'] = 'incomplete'
    manifest['review'] = dict(policy='exclude_failed_train_only', source_manifest_sha256=source_hash,
                             source=str(source.resolve()), excluded=failures,
                             warning='Sélection des démonstrations train ; ne prouve pas la sûreté du professeur')
    manifest['episodes'] = kept
    manifest['collection']['splits'] = {split: [e['seed'] for e in kept if e['split']==split]
                                         for split in ('train','validation','test')}
    path=output/'manifest.json'
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for split in ('train','validation','test'):
        (output/split).mkdir()
    for entry in kept:
        shutil.copyfile(source/entry['path'],output/entry['path'])
    manifest['summary'] = {split: summarize([e['quality'] for e in kept if e['split']==split])
                           for split in ('train','validation','test')}
    manifest['quality_status']='passed'
    manifest['status']='complete'
    path.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    load_manifest(output)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description='Revue explicite : exclure seulement les démonstrations train en échec')
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    result=review_dataset(args.source,args.output)
    print(json.dumps(dict(kept=len(result['episodes']),excluded=[e['seed'] for e in result['review']['excluded']]),indent=2))
    return result


if __name__=='__main__':
    main()
