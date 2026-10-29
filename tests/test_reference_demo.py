"""Intégration de la démonstration actuelle : calculs réels et exports relisibles."""
import csv
import io
import json
from pathlib import Path
from threading import Thread
import time
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import zipfile

import numpy as np
import pytest

from demo.reference_server import ReferenceService,handler_for,validate_request,public_catalog
from demo.comparison_report import build_report


@pytest.mark.parametrize('payload',[
    {'shape':'../../private'}, {'robots':True}, {'robots':10000}, {'seed':-1},
    {'checkpoint':'arbitrary.pt'}, {'controller':'raw'}, {'scenario':'unknown'},
])
def test_invalid_public_requests_are_rejected(payload):
    with pytest.raises(ValueError):validate_request(payload)


def test_all_catalog_entries_and_missing_model_fallback(tmp_path):
    service=ReferenceService(tmp_path,model_run=tmp_path/'absent')
    try:
        entries=public_catalog()
        assert len(entries)==56 and len({e['identifier'] for e in entries})==56
        assert not service.catalog()['model_available']
        with pytest.raises(ValueError,match='absent'):service._controller()
    finally:service.executor.shutdown()


def test_final_report_keeps_failed_raw_and_difficult_cases():
    report=build_report()
    raw=[r for r in report['rows'] if r['mode']=='raw']
    assert len(raw)==1 and raw[0]['trials']==9 and raw[0]['collisions']==54
    assert raw[0]['success_rate_percent']==0 and raw[0]['successful_time_mean_s'] is None
    difficult=[r for r in report['rows'] if r['campaign']=='Navigation · cas difficiles']
    assert len(difficult)==2 and all(r['trials']==4 and r['success_rate_percent']==0 for r in difficult)
    assert len(report['sources_sha256'])==3


def test_multisize_choices_are_explicit_and_restrict_unstudied_navigation():
    assert validate_request(dict(robots=64,controller='teacher'))['robots']==64
    assert validate_request(dict(robots=24,controller='multisize'))['controller']=='multisize'
    for request in (dict(robots=64),dict(robots=64,controller='teacher',scenario='straight'),dict(robots=8,controller='multisize')):
        with pytest.raises(ValueError):validate_request(request)


def test_completed_export_survives_server_restart(tmp_path):
    identifier='1'*32;folder=tmp_path/identifier;folder.mkdir()
    manifest=dict(kind='reference_demo',request=validate_request(dict(controller='teacher')),reports=dict(teacher=dict(success=False)))
    (folder/'manifest.json').write_text(json.dumps(manifest))
    with zipfile.ZipFile(folder/'export.zip','w') as archive:archive.writestr('manifest.json',json.dumps(manifest))
    service=ReferenceService(tmp_path)
    try:
        restored=service.status(identifier)
        assert restored['status']=='complete' and not restored['summary']['teacher']['success']
        assert service.submitted==0
    finally:service.executor.shutdown()


def test_structured_model_is_gated_by_complete_success_and_audit(tmp_path,monkeypatch):
    import demo.reference_server as server
    import hashlib
    run=tmp_path/'model';run.mkdir();(run/'best.pt').write_bytes(b'candidate')
    digest=hashlib.sha256(b'candidate').hexdigest();evaluation=tmp_path/'evaluation.json';audit=tmp_path/'audit.json'
    monkeypatch.setattr(server,'MULTISIZE_RUN',run);monkeypatch.setattr(server,'MULTISIZE_EVALUATION',evaluation)
    monkeypatch.setattr(server,'MULTISIZE_AUDIT',audit)
    rows=[dict(count=n,shape=s,seed=seed,metrics=dict(success=True,collisions=0)) for n in (12,24,64)
          for s in ('geometries_rectangle','letters_C','numbers_3') for seed in (8500,8501)]
    report=dict(status='complete',split='fresh_test',model_sha256=digest,reports=rows)
    evaluation.write_text(json.dumps(report));audit.write_text(json.dumps(dict(success=True,model_sha256=digest,raw_motion_filter_interventions=0)))
    assert server.verified_structured_report() is not None
    rows[0]['metrics']['collisions']=1;evaluation.write_text(json.dumps(report))
    assert server.verified_structured_report() is None


def test_structured_demo_has_unfiltered_motion_and_lossless_graph_bank(tmp_path,monkeypatch):
    from src.imitation.vector_gat import VectorGAT,VectorConfig
    from src.imitation.local_target_model import LocalTargetController
    service=ReferenceService(tmp_path)
    policy=LocalTargetController(VectorGAT(VectorConfig(8)))
    monkeypatch.setattr(service,'_multisize_controller',lambda:policy)
    try:
        manifest=service.simulate(dict(controller='multisize',robots=12,seed=8200),tmp_path/'raw')
        assert set(manifest['reports'])=={'raw'} and manifest['guard']=='shared_braking_allocation_only'
        display=json.loads((tmp_path/'raw/raw.json').read_text())
        with np.load(tmp_path/'raw/raw.npz',allow_pickle=False) as archive:
            graph=np.asarray(display['graph_bank'],dtype=bool)[display['graph_ids']]
            np.testing.assert_array_equal(graph,archive['adjacency'])
            moving=np.isin(archive['phases'],(0,3))
            assert not archive['safety_interventions'][moving].any()
    finally:service.executor.shutdown()


def test_real_simulation_export_has_exact_states_and_reproducible_configuration(tmp_path):
    service=ReferenceService(tmp_path)
    try:
        request=validate_request(dict(controller='teacher',robots=8,seed=7701))
        manifest=service.simulate(request,tmp_path/'run')
        assert manifest['request']==request and manifest['model_sha256'] is None
        assert manifest['communication_radius']==3 and manifest['experimental_robot_count']
        with np.load(tmp_path/'run/teacher.npz',allow_pickle=False) as data:
            display=json.loads((tmp_path/'run/teacher.json').read_text())
            np.testing.assert_array_equal(data['positions'],display['positions'])
            np.testing.assert_array_equal(data['velocities'],display['velocities'])
            np.testing.assert_array_equal(data['positions'][0],manifest['initial_positions'])
            assert len(display['times'])==len(display['positions'])==401
            assert np.linalg.norm(data['velocities'],axis=-1).max()<=1+1e-9
            rows=list(csv.DictReader(io.StringIO((tmp_path/'run/teacher-trajectory.csv').read_text())))
            assert len(rows)==401*8
            assert float(rows[-1]['x_m'])==data['positions'][-1,-1,0]
        with zipfile.ZipFile(tmp_path/'run/export.zip') as archive:
            assert set(archive.namelist())=={'manifest.json','teacher.json','teacher.npz','teacher-trajectory.csv'}
            assert json.loads(archive.read('manifest.json'))==manifest
        with pytest.raises(FileExistsError):service.simulate(request,tmp_path/'run')
    finally:service.executor.shutdown()


def test_http_job_lifecycle_and_origin_guard(tmp_path):
    service=ReferenceService(tmp_path)
    # The job system is tested independently from the real numerical test above.
    def simulate(request,folder):
        folder.mkdir(parents=True);(folder/'teacher.json').write_text('{"verified":true}')
        return dict(reports=dict(teacher=dict(success=False,stabilization_time_s=None)))
    service.simulate=simulate
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(service))
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base+'/api/catalog') as response:assert len(json.load(response)['shapes'])==56
        with urlopen(base+'/guide.md') as response:
            guide=response.read().decode('utf-8')
            assert '# Guide des expériences' in guide and 'Pourquoi un cercle' in guide
        with urlopen(Request(base+'/api/geometry',data=b'{"robots":24,"shape":"geometries_arrow","controller":"teacher"}')) as response:
            preview=json.load(response)['options']['geometries_arrow']
            assert preview['compatible'] and preview['adjusted'] and preview['width_m']>6
        assert service.submitted==0 and not service.jobs
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base+'/api/jobs',data=b'{}',headers={'Origin':'https://foreign.example'}))
        assert error.value.code==403
        with urlopen(Request(base+'/api/jobs',data=b'{"controller":"teacher"}')) as response:
            assert response.status==202;identifier=json.load(response)['id']
        deadline=time.monotonic()+5
        while True:
            with urlopen(base+'/api/jobs/'+identifier) as response:job=json.load(response)
            if job['status'] not in ('running','queued'):break
            assert time.monotonic()<deadline;time.sleep(.02)
        assert job['status']=='complete' and not job['summary']['teacher']['success']
        with urlopen(base+'/api/jobs/'+identifier+'/teacher.json') as response:assert json.load(response)==dict(verified=True)
        with pytest.raises(HTTPError) as error:urlopen(base+'/api/jobs/'+identifier+'/../../manifest.json')
        assert error.value.code==404
    finally:
        server.shutdown();server.server_close();thread.join();service.executor.shutdown()
