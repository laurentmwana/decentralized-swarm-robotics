"""Pilote multi-effectifs : protocole figé, professeur et stockage local CSR."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .multisize import PROFILES,controlled_start,sparse_examples
from .multiform_data import geometry_fingerprint,sha256
from src.reference.catalog import catalog_entries
from src.reference.shape_control import ShapeGains,ShapeMetricConfig,execute_shape,evaluate_shape
from src.reference.shapes import named_shape


PARTITIONS=dict(train=['geometries_square','letters_A','numbers_1'],
                validation=['geometries_triangle','letters_B','numbers_7'],
                test=['geometries_rectangle','letters_C','numbers_3'])
SEEDS=dict(train=[8100],validation=[8200],test=[8300])


def catalog_audit():
    audit=[]
    for count,profile in PROFILES.items():
        for entry in catalog_entries():
            mission=named_shape(entry['identifier'],count,width=profile.width)
            try:mission.validate_environment(profile.config,.4);reason=None
            except ValueError as exc:reason=str(exc)
            audit.append(dict(count=count,shape=mission.identifier,feasible=reason is None,reason=reason))
    return audit


def execute_teacher(profile,name,seed):
    config=profile.config;gains=ShapeGains(detour_only_on_constraint=True)
    mission=named_shape(name,profile.count,width=profile.width)
    result=execute_shape(mission,config,controlled_start(profile,seed),gains=gains,reconnection=profile.reconnection,
                         seed=seed,safety_filter=True)
    return result,evaluate_shape(result,ShapeMetricConfig(.1,.05,1.,config.horizon*config.physics.dt))


def collect(output,*,benchmark=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    audit=catalog_audit();fingerprints={}
    for split,names in PARTITIONS.items():
        for name in names:
            fingerprint=geometry_fingerprint(named_shape(name,12,width=6.))
            if fingerprint in fingerprints and fingerprints[fingerprint]!=split:raise ValueError('Alias entre partitions')
            fingerprints[fingerprint]=split
            for count in PROFILES:
                row=next(a for a in audit if a['count']==count and a['shape']==name)
                if not row['feasible']:raise ValueError(str(row))
    manifest=dict(schema_version=3,feature_schema_version=2,kind='multi_size_local_target_imitation',status='incomplete',
        domain='controlled_near_own_rendezvous_ring_not_general_scattered_starts',profiles={str(n):asdict(p) for n,p in PROFILES.items()},
        shape_splits=PARTITIONS,episode_seeds=SEEDS,catalog_audit=audit,
        allocation='fixed_N_squared_rounds_no_global_early_stop',sampling_stride=2,
        model=dict(hidden_dim=64,spatial_scale=28.,speed_scale=1.,count_scale=64.,init_seed=42),
        training=dict(epochs=40,batch_size=512,learning_rate=.001,patience=8,seed=42,max_samples_per_count=24000),
        metrics=dict(error_threshold=.1,speed_threshold=.05,hold_seconds=1.),
        storage='own_8_labels_2_relative_neighbors_4_CSR_offsets',episodes=[])
    def save():(output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    save()
    work=[('benchmark',64,'geometries_square',8000)] if benchmark else [
        (split,count,name,seed) for split in ('train','validation') for count in PROFILES
        for seed in SEEDS[split] for name in PARTITIONS[split]]
    for split,count,name,seed in work:
        started=time.perf_counter();result,metrics=execute_teacher(PROFILES[count],name,seed)
        arrays=sparse_examples(result);prefix=f'{split}-{count}-{name}-{seed}'
        result.episode.save(output/f'{prefix}-physics.npz')
        with (output/f'{prefix}-samples.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
        duration=time.perf_counter()-started
        manifest['episodes'].append(dict(split=split,count=count,shape=name,seed=seed,
            physics=f'{prefix}-physics.npz',samples=f'{prefix}-samples.npz',sha256=sha256(output/f'{prefix}-samples.npz'),
            sample_count=len(arrays['own']),neighbor_records=len(arrays['edges']),max_neighbors=int(np.diff(arrays['offsets']).max()),
            examples_bytes=sum(a.nbytes for a in arrays.values()),wall_seconds=duration,metrics=metrics))
        save();print(f'{split} N={count} {name}: success={metrics["success"]} contacts={metrics["collisions"]} time={duration:.1f}s',flush=True)
    manifest.update(status='complete',quality_status='passed' if all(e['metrics']['success'] for e in manifest['episodes']) else 'review_required')
    save();return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--benchmark',action='store_true');args=parser.parse_args();collect(args.output,benchmark=args.benchmark)


if __name__=='__main__':main()
