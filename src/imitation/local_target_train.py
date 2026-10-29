"""Imitation locale multi-formes ; train/validation uniquement, alias exclus."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from .multiform_data import load_manifest, load_episode, sha256
from .local_target_model import LocalTargetGAT, LocalTargetConfig, save_model
from .objective import acceleration_mse
from .train import TrainingConfig


def prepare_data(dataset):
    manifest = load_manifest(dataset)
    arrays, selection = {}, {}
    for split in ('train', 'validation'):
        rows = [e for e in manifest['episodes'] if e['split'] == split and e['duplicate_within_split_of'] is None]
        if not rows: raise ValueError('Ensemble d’imitation vide après déduplication')
        pieces = []
        for entry in rows:
            episode = load_episode(dataset, entry, split)
            keep = episode['learn_mask']
            pieces.append((episode['node_features'][keep], episode['relative_neighbors'][keep],
                           episode['adjacency'][keep], episode['targets'][keep]))
        combined = [np.concatenate([p[i] for p in pieces]) for i in range(4)]
        if not len(combined[0]): raise ValueError('Aucun exemple dans les phases apprises')
        arrays[split] = tuple(torch.as_tensor(v, dtype=torch.bool if i == 2 else torch.float32) for i, v in enumerate(combined))
        selection[split] = dict(episodes=len(rows), samples=len(combined[0]), paths=[e['path'] for e in rows])
    return manifest, arrays, selection


def batches(arrays, size, *, generator=None, shuffle=False):
    order = torch.randperm(len(arrays[0]), generator=generator) if shuffle else torch.arange(len(arrays[0]))
    for start in range(0, len(order), size):
        indices = order[start:start+size]
        yield tuple(a[indices] for a in arrays)


def validation_mse(model, arrays, batch_size, device):
    total, count = 0., 0
    model.eval()
    with torch.inference_mode():
        for own, neighbors, mask, target in batches(arrays, batch_size):
            prediction = model(own.to(device), neighbors.to(device), mask.to(device))
            loss = acceleration_mse(prediction, target.to(device))
            total += float(loss)*target.numel(); count += target.numel()
    return total/count


def run_training(dataset, output, model_config, training_config, *, device='cpu', dry_run=False):
    manifest, arrays, selection = prepare_data(dataset)
    if device != 'cpu' and not (device == 'cuda' and torch.cuda.is_available()):
        raise ValueError('Appareil cpu ou cuda disponible requis')
    model = LocalTargetGAT(model_config, manifest['simulator']['physics']['max_acceleration']).to(device)
    report = dict(schema_version=1, kind='multiform_local_target_training', status='dry_run' if dry_run else 'training',
        dataset=str(Path(dataset).resolve()), dataset_manifest_sha256=sha256(Path(dataset)/'manifest.json'),
        model=model.metadata(), training=asdict(training_config), device=device, selection=selection,
        torch_version=str(torch.__version__), numpy_version=str(np.__version__), history=[])
    if dry_run: return report
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    def save():
        (output/'run.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    save()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    generator = torch.Generator().manual_seed(training_config.seed)
    best, stale, best_path = float('inf'), 0, None
    for epoch in range(1, training_config.epochs+1):
        model.train(); total, count = 0., 0
        for own, neighbors, mask, targets in batches(arrays['train'], training_config.batch_size, generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(own.to(device), neighbors.to(device), mask.to(device))
            loss = acceleration_mse(prediction, targets.to(device))
            if not torch.isfinite(loss): raise ValueError('Perte d’apprentissage non finie')
            loss.backward(); optimizer.step()
            total += float(loss.detach())*targets.numel(); count += targets.numel()
        score = validation_mse(model, arrays['validation'], training_config.batch_size, device)
        if not math.isfinite(score): raise ValueError('Perte de validation non finie')
        report['history'].append(dict(epoch=epoch, train_mse=total/count, validation_mse=score))
        if score < best:
            best, stale = score, 0
            best_path = output/f'epoch-{epoch:04d}.pt'; save_model(best_path, model)
            report.update(best_epoch=epoch, best_validation_mse=score)
        else: stale += 1
        save()
        print(f'Epoch {epoch}: train={total/count:.6f}, validation={score:.6f}', flush=True)
        if stale >= training_config.patience: break
    shutil.copyfile(best_path, output/'best.pt')
    report.update(status='trained', checkpoint='best.pt', epochs_completed=len(report['history']))
    save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='config/local-target-model.yaml')
    parser.add_argument('--training', default='config/local-target-training.yaml')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    result = run_training(args.dataset, args.output, LocalTargetConfig.from_yaml(args.model),
                          TrainingConfig.from_yaml(args.training), device=args.device, dry_run=args.dry_run)
    if args.dry_run: print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
