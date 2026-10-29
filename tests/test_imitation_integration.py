"""Intégration, lots et vérification à blanc uniquement : aucun entraînement."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.imitation.controller import LocalGATController
from src.imitation.data import CollectionConfig,collect_dataset,load_manifest
from src.imitation.model import AccelerationGAT,GATConfig
from src.imitation.review import review_dataset
from src.imitation.train import TrainingConfig,batches,prepare_data,run_training,validation_mse
from src.reference.circle import CircleTeacher
from src.reference.metrics import CircleMetricConfig
from src.simulator.acceleration_episode import AccelerationEpisode,AccelerationEpisodeConfig,LocalObservation


def setup():
    cfg=replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'),horizon=3)
    teacher=CircleTeacher.from_yaml('config/circle-teacher.yaml',cfg)
    return cfg,teacher,CircleMetricConfig(.1,.05,.1,.3),CollectionConfig(dict(train=[1,2],validation=[3],test=[4]),.4,10000)


@pytest.fixture
def source(tmp_path):
    folder=tmp_path/'raw'; manifest=collect_dataset(*setup(),folder)
    # Fixture de revue : étiquettes de qualité contrôlées, pas une mesure réelle de succès.
    for entry in manifest['episodes']:
        entry['quality'].update(success=entry['seed']!=2,stabilization_time_s=0. if entry['seed']!=2 else None)
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return folder


@pytest.fixture
def reviewed(source,tmp_path):
    folder=tmp_path/'reviewed'; review_dataset(source,folder)
    return folder


def test_review_excludes_only_failed_train_with_provenance(source,tmp_path):
    original=(source/'manifest.json').read_bytes()
    output=tmp_path/'reviewed'
    manifest=review_dataset(source,output)
    assert [e['seed'] for e in manifest['review']['excluded']]==[2]
    assert manifest['collection']['splits']==dict(train=[1],validation=[3],test=[4])
    assert (source/'manifest.json').read_bytes()==original
    assert load_manifest(output)['quality_status']=='passed'
    for entry in manifest['episodes']:
        assert (source/entry['path']).read_bytes()==(output/entry['path']).read_bytes()


@pytest.mark.parametrize('split',['validation','test'])
def test_review_cannot_remove_validation_or_test(source,tmp_path,split):
    manifest=json.loads((source/'manifest.json').read_text())
    for entry in manifest['episodes']:
        if entry['split']==split: entry['quality']['success']=False
    (source/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='validation ou de test'):
        review_dataset(source,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_review_refuses_all_train_failures(source,tmp_path):
    manifest=json.loads((source/'manifest.json').read_text())
    for entry in manifest['episodes']:
        if entry['split']=='train': entry['quality']['success']=False
    (source/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Aucun épisode'):
        review_dataset(source,tmp_path/'output')


def test_prepare_data_never_loads_test(reviewed,monkeypatch):
    import src.imitation.train as training
    original=training.load_episode
    seen=[]
    def tracked(dataset,entry,expected_split):
        assert expected_split!='test'
        seen.append(expected_split)
        return original(dataset,entry,expected_split)
    monkeypatch.setattr(training,'load_episode',tracked)
    manifest,arrays=prepare_data(reviewed)
    assert seen==['train','validation']
    assert set(arrays)=={'train','validation'}
    assert arrays['train'][0].shape==(3,12,8)


def test_batches_keep_graphs_whole_and_cover_every_step():
    values=torch.arange(5).view(5,1,1).expand(5,3,8)
    graph=torch.zeros(5,3,3,dtype=torch.bool); targets=torch.zeros(5,3,2)
    result=list(batches((values,graph,targets),2))
    assert [len(batch[0]) for batch in result]==[2,2,1]
    assert torch.equal(torch.cat([b[0] for b in result]),values)
    first=list(batches((values,graph,targets),2,generator=torch.Generator().manual_seed(42),shuffle=True))
    second=list(batches((values,graph,targets),2,generator=torch.Generator().manual_seed(42),shuffle=True))
    assert torch.equal(torch.cat([b[0] for b in first]),torch.cat([b[0] for b in second]))


def test_dry_run_creates_no_optimizer_or_weights(reviewed,tmp_path,monkeypatch):
    def forbidden(*args,**kwargs): pytest.fail('Optimiseur créé pendant la vérification à blanc')
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    output=tmp_path/'training'
    result=run_training(reviewed,output,GATConfig.from_yaml('config/imitation-model.yaml'),TrainingConfig(2,2,.001,1,42),dry_run=True)
    assert result['status']=='dry_run' and result['graphs']==dict(train=3,validation=3)
    assert not output.exists()


def test_unreviewed_data_cannot_start_training(source,tmp_path,monkeypatch):
    monkeypatch.setattr(torch.optim,'Adam',lambda *args,**kwargs: pytest.fail('Optimiseur créé avant contrôle qualité'))
    with pytest.raises(ValueError,match='examiner'):
        run_training(source,tmp_path/'training',GATConfig.from_yaml('config/imitation-model.yaml'),TrainingConfig(2,2,.001,1,42),dry_run=True)


@pytest.mark.parametrize('change',[dict(epochs=0),dict(batch_size=True),dict(learning_rate=0),
                                   dict(patience=0),dict(seed=-1)])
def test_invalid_training_configuration(change):
    with pytest.raises(ValueError): replace(TrainingConfig(2,2,.001,1,42),**change)


@pytest.mark.parametrize('relative',[False,True])
def test_local_star_adapter_matches_global_one_hop(relative):
    cfg,teacher,_,_=setup()
    net=AccelerationGAT(replace(GATConfig.from_yaml('config/imitation-model.yaml'),relative_edges=relative),2.)
    initial=AccelerationEpisode(cfg).execute(teacher,12,7)
    p,v=initial.positions[0],initial.velocities[0]; graph=initial.adjacency[0]
    features=np.column_stack((p,v,np.tile(teacher.mission.center,(12,1)),np.full(12,2.),np.full(12,12)))
    with torch.inference_mode(): expected=net(torch.tensor(features,dtype=torch.float32),torch.tensor(graph)).numpy()
    controller=LocalGATController(net,teacher.mission)
    for i in range(12):
        neighbors=graph[i]
        observation=LocalObservation(p[i],v[i],p[neighbors]-p[i],v[neighbors]-v[i],0.)
        np.testing.assert_allclose(controller(observation),expected[i],atol=1e-6,rtol=1e-6)
    result=AccelerationEpisode(cfg).execute(controller,12,7)
    assert np.isfinite(result.positions).all()
    assert np.linalg.norm(result.accelerations,axis=-1).max()<=2+1e-6


def test_validation_only_forward_preserves_weights(reviewed):
    _,arrays=prepare_data(reviewed)
    net=AccelerationGAT(GATConfig.from_yaml('config/imitation-model.yaml'),2.)
    before={k:v.clone() for k,v in net.state_dict().items()}
    score=validation_mse(net,arrays['validation'],2,'cpu')
    assert np.isfinite(score)
    for key,value in net.state_dict().items(): torch.testing.assert_close(before[key],value,atol=0,rtol=0)


def test_comparison_refuses_untrained_run(tmp_path):
    from src.imitation.compare import compare_run
    run=tmp_path/'run';run.mkdir();(run/'run.json').write_text(json.dumps(dict(status='dry_run')))
    with pytest.raises(ValueError,match='entraînement terminé'):
        compare_run(run,tmp_path/'comparison')


def test_training_cli_dry_run(reviewed,tmp_path,monkeypatch):
    from src.imitation.train import main
    monkeypatch.setattr(torch.optim,'Adam',lambda *args,**kwargs: pytest.fail('Entraînement interdit dans ce test'))
    result=main(['--dataset',str(reviewed),'--output',str(tmp_path/'unused'),'--dry-run'])
    assert result['status']=='dry_run'


@pytest.mark.parametrize('relocated',[False,True])
def test_comparison_wiring_uses_only_test_and_identical_initial_states(reviewed,tmp_path,relocated):
    """Poids initiaux et fiche fictive : vérifie le raccordement, aucune qualité apprise."""
    import hashlib
    from src.imitation.checkpoint import save_model
    from src.imitation.compare import compare_run
    net=AccelerationGAT(GATConfig.from_yaml('config/imitation-model.yaml'),2.)
    run=tmp_path/'synthetic-run';run.mkdir();save_model(run/'best.pt',net)
    (run/'run.json').write_text(json.dumps(dict(status='trained',synthetic_fixture=True,
        dataset=str(tmp_path/'old-location') if relocated else str(reviewed.resolve()),dataset_manifest_sha256=hashlib.sha256((reviewed/'manifest.json').read_bytes()).hexdigest(),
        checkpoint='best.pt',model=net.metadata())))
    output=tmp_path/'comparison'
    report=compare_run(run,output,dataset=reviewed if relocated else None)
    assert report['dataset']==str(reviewed.resolve())
    assert [entry['seed'] for entry in report['reports']]==[4]
    with np.load(output/'teacher-4.npz') as teacher, np.load(output/'gat-4.npz') as gat:
        np.testing.assert_array_equal(teacher['positions'][0],gat['positions'][0])
        np.testing.assert_array_equal(teacher['velocities'][0],gat['velocities'][0])
    assert np.isfinite(report['test_teacher_state_mse'])


def test_comparison_refuses_changed_manifest(reviewed,tmp_path):
    from src.imitation.compare import compare_run
    run=tmp_path/'run';run.mkdir()
    (run/'run.json').write_text(json.dumps(dict(status='trained',dataset=str(reviewed),dataset_manifest_sha256='changed')))
    with pytest.raises(ValueError,match='manifeste'):
        compare_run(run,tmp_path/'comparison')


def test_relocated_dataset_must_match_training_fingerprint(reviewed,tmp_path):
    from src.imitation.compare import compare_run
    run=tmp_path/'run'; run.mkdir()
    (run/'run.json').write_text(json.dumps(dict(status='trained',dataset='old-location',dataset_manifest_sha256='changed')))
    output=tmp_path/'comparison'
    with pytest.raises(ValueError,match='manifeste'):
        compare_run(run,output,dataset=reviewed)
    assert not output.exists()
