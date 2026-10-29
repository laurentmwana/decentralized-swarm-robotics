"""Démonstration locale du plan à accélérations : python -m demo.reference_server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,replace
import csv
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
from threading import RLock
import uuid
import zipfile

import numpy as np

from src.reference.catalog import catalog_entries
from src.reference.navigation import MovingReference,execute_navigation,evaluate_navigation,initial_positions
from src.reference.obstacles import DiskWorld,DiskObstacle
from src.reference.route_planning import plan_formation_route
from src.reference.reconnection import ReconnectionConfig
from src.reference.shape_control import ShapeGains,execute_shape,evaluate_shape,ShapeMetricConfig
from src.reference.shapes import named_shape
from src.reference.transitions import FormationStage,execute_sequence,evaluate_sequence
from src.simulator.acceleration_episode import AccelerationEpisodeConfig

ROOT=Path(__file__).resolve().parents[1]
MODEL_RUN=ROOT/'outputs/imitation/training-multiform-expanded-v1'
MULTISIZE_RUN=ROOT/'outputs/imitation/training-vector-gat-v2'
MULTISIZE_DATA=ROOT/'outputs/imitation/multisize-pilot-task012'
MULTISIZE_EVALUATION=ROOT/'outputs/imitation/vector-fresh-test-v2/evaluation.json'
MULTISIZE_AUDIT=ROOT/'tasks/evidence/014-vector-final-audit.json'
EXPECTED_MODEL_SHA='a1e3b88b39386a3d490798b5c7717e3939416e52572a058045d7402ec6cf6d84'
SHAPES={entry['identifier'] for entry in catalog_entries()}
GEOMETRY_LABELS=dict(zip(
    ('circle','ellipse','square','rectangle','triangle','pentagon','hexagon','heptagon','octagon','decagon','diamond','trapezoid','parallelogram','star','cross','arrow','semicircle','heart','line','v'),
    ('Cercle','Ellipse','Carré','Rectangle','Triangle','Pentagone','Hexagone','Heptagone','Octogone','Décagone','Losange','Trapèze','Parallélogramme','Étoile','Croix','Flèche','Demi-cercle','Cœur','Ligne','V')))


def public_catalog():
    return [dict(**entry,label=(GEOMETRY_LABELS[entry['name']] if entry['category']=='geometries' else entry['name']))
            for entry in catalog_entries()]


def verified_structured_report():
    try:
        report=json.loads(MULTISIZE_EVALUATION.read_text(encoding='utf-8'))
        audit=json.loads(MULTISIZE_AUDIT.read_text(encoding='utf-8'))
        expected={(n,shape,seed) for n in (12,24,64) for shape in ('geometries_rectangle','letters_C','numbers_3') for seed in (8500,8501)}
        rows=report['reports']
        if report.get('status')!='complete' or report.get('split')!='fresh_test' or len(rows)!=18:return None
        if {(r['count'],r['shape'],r['seed']) for r in rows}!=expected:return None
        if not all(r['metrics']['success'] and r['metrics']['collisions']==0 for r in rows):return None
        digest=hashlib.sha256((MULTISIZE_RUN/'best.pt').read_bytes()).hexdigest()
        if not audit.get('success') or audit.get('model_sha256')!=digest or report['model_sha256']!=digest:return None
        if audit.get('raw_motion_filter_interventions')!=0:return None
        return report
    except (OSError,ValueError,KeyError,TypeError):return None


def validate_request(raw):
    if not isinstance(raw,dict) or set(raw)-{'shape','next_shape','robots','seed','controller','scenario'}:
        raise ValueError('Paramètres de démonstration invalides.')
    request=dict(shape='geometries_square',next_shape='letters_A',robots=12,seed=7700,
                 controller='guarded',scenario='formation');request.update(raw)
    if any(not isinstance(request[key],str) or request[key] not in SHAPES for key in ('shape','next_shape')):
        raise ValueError('Choisissez une forme du catalogue.')
    if type(request['robots']) is not int or request['robots'] not in (8,12,16,24,64):
        raise ValueError('Choisissez 8, 12, 16, 24 ou 64 robots.')
    if type(request['seed']) is not int or not 0<=request['seed']<=2**31-1:
        raise ValueError('Le départ doit être un entier entre 0 et 2147483647.')
    if request['controller'] not in ('teacher','guarded','comparison','multisize','comparison_multisize'):
        raise ValueError('Contrôleur inconnu.')
    if request['scenario'] not in ('formation','transition','straight','bypass','direct','blocked'):
        raise ValueError('Parcours inconnu.')
    multi=request['controller'] in ('multisize','comparison_multisize')
    if multi and request['robots'] not in (12,24,64):raise ValueError('Le pilote multi-effectifs couvre 12, 24 et 64 robots.')
    if (multi or request['robots']==64) and request['scenario']!='formation':
        raise ValueError('Le pilote multi-effectifs est limité à la formation ; transitions et navigation restent à évaluer.')
    if request['robots']==64 and request['controller'] not in ('teacher','multisize','comparison_multisize'):
        raise ValueError('À 64 robots, choisissez le pilote multi-effectifs ou le professeur classique.')
    return request


def navigation_scene(scenario):
    if scenario=='bypass':return MovingReference(((-1,0),(-1,1),(1,1),(1,0)),(10,16,10)),DiskWorld((DiskObstacle((-1,-3),.25),)),460
    world=DiskWorld((DiskObstacle((-1,-3),.25),)) if scenario=='direct' else (
        DiskWorld((DiskObstacle((4,-3),.35),)) if scenario=='blocked' else DiskWorld())
    return MovingReference(((-1,0),(1,0)),(16,)),world,260


class ReferenceService:
    def __init__(self,output=None,model_run=MODEL_RUN):
        self.config=replace(AccelerationEpisodeConfig.from_yaml(ROOT/'config/acceleration-simulator.yaml'),communication_radius=3.)
        self.output=Path(output) if output is not None else ROOT/'outputs/demo-reference'
        self.model_run=Path(model_run);self.controller=None;self.model_hash=None
        self.multisize_controller=None;self.multisize_hash=None
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='swarm-demo')
        self.lock=RLock();self.jobs={};self.submitted=0
        # Restaurer les calculs terminés : un redémarrage ne casse pas leurs exports.
        if self.output.is_dir():
            folders=sorted((p for p in self.output.iterdir() if p.is_dir() and re.fullmatch('[0-9a-f]{32}',p.name)),
                           key=lambda p:p.stat().st_mtime)[-20:]
            for folder in folders:
                if not folder.resolve().is_relative_to(self.output.resolve()):continue
                try:
                    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
                    if manifest.get('kind')=='reference_demo' and (folder/'export.zip').is_file():
                        self.jobs[folder.name]=dict(id=folder.name,status='complete',request=manifest['request'],summary=manifest['reports'])
                except (OSError,ValueError,KeyError):continue

    def catalog(self):
        evaluation=verified_structured_report();multi_available=evaluation is not None
        multi_summary=({n:dict(raw=evaluation['summary'][n],teacher=evaluation['teacher_summary'][n]) for n in ('12','24','64')}
                       if multi_available else None)
        return dict(shapes=public_catalog(),robot_counts=[8,12,16,24,64],model_available=(self.model_run/'best.pt').is_file(),
                    multisize_model_available=multi_available,multisize_status='structured_raw_controlled_start_formation_only',
                    multisize_summary=multi_summary,
                    trained_robot_count=12,units=dict(position='m',time='s',speed='m/s',acceleration='m/s²'))

    def _multisize_controller(self):
        if not self.catalog()['multisize_model_available']:
            raise ValueError('Le pilote multi-effectifs n’a pas encore terminé son évaluation.')
        if self.multisize_controller is None:
            import torch
            torch.set_num_threads(1)
            from src.imitation.local_target_model import load_model,LocalTargetController
            from src.imitation.vector_gat import load_vector
            from src.imitation.multiform_data import sha256
            run=json.loads((MULTISIZE_RUN/'run.json').read_text(encoding='utf-8'))
            evaluation=json.loads(MULTISIZE_EVALUATION.read_text(encoding='utf-8'))
            checkpoint=MULTISIZE_RUN/'best.pt';digest=sha256(checkpoint)
            if run.get('kind')!='vector_gat_training' or run.get('status')!='trained' or digest!=evaluation['model_sha256']:
                raise ValueError('Poids multi-effectifs différents du pilote évalué.')
            if sha256(MULTISIZE_DATA/'manifest.json')!=run['dataset_manifest_sha256']:
                raise ValueError('Jeu du pilote modifié depuis l’entraînement.')
            model=load_vector(checkpoint)
            if model.metadata()!=run['model']:raise ValueError('Contrat multi-effectifs incompatible.')
            self.multisize_controller=LocalTargetController(model);self.multisize_hash=digest
        return self.multisize_controller

    def _controller(self):
        if self.controller is None:
            import torch
            torch.set_num_threads(1)
            from src.imitation.local_target_model import load_model,LocalTargetController
            checkpoint=self.model_run/'best.pt'
            if not checkpoint.is_file():raise ValueError('Le modèle entraîné est absent. Choisissez le contrôleur classique.')
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest!=EXPECTED_MODEL_SHA:raise ValueError('Le modèle disponible ne correspond pas aux poids validés de cette démonstration.')
            run=json.loads((self.model_run/'run.json').read_text(encoding='utf-8'))
            model=load_model(checkpoint)
            if run.get('status')!='trained' or model.metadata()!=run['model'] or model.max_acceleration!=self.config.physics.max_acceleration:
                raise ValueError('Le modèle est incompatible avec le simulateur.')
            self.controller=LocalTargetController(model);self.model_hash=digest
        return self.controller

    def submit(self,raw):
        request=validate_request(raw)
        gains=ShapeGains(detour_only_on_constraint=True)
        multi=request['controller'] in ('multisize','comparison_multisize') or request['robots']==64
        if multi:
            from src.imitation.multisize import PROFILES
            profile=PROFILES[request['robots']];config=profile.config;width=profile.width
        else:config=self.config;width=6.
        for key in ('shape','next_shape') if request['scenario']=='transition' else ('shape',):
            named_shape(request[key],request['robots'],width=width).validate_environment(config,gains.safety_distance)
        if request['scenario'] not in ('formation','transition'):
            reference,world,_=navigation_scene(request['scenario'])
            plan_formation_route(named_shape(request['shape'],request['robots'],width=width),reference,world,config)
        with self.lock:
            if self.submitted>=20:raise ValueError('Cette session contient déjà 20 calculs. Relancez le serveur pour une nouvelle session ; les exports sont conservés.')
            if sum(j['status'] in ('queued','running') for j in self.jobs.values())>=2:
                raise ValueError('Deux calculs sont déjà en attente. Attendez leur résultat.')
            identifier=uuid.uuid4().hex
            self.jobs[identifier]=dict(id=identifier,status='queued',request=request)
            self.submitted+=1
            self.executor.submit(self._run,identifier,request)
        return identifier

    def status(self,identifier):
        with self.lock:
            if identifier not in self.jobs:raise KeyError(identifier)
            return dict(self.jobs[identifier])

    def _run(self,identifier,request):
        with self.lock:self.jobs[identifier]['status']='running'
        try:
            manifest=self.simulate(request,self.output/identifier)
            with self.lock:self.jobs[identifier].update(status='complete',summary=manifest['reports'])
        except ValueError as exc:
            with self.lock:self.jobs[identifier].update(status='failed',error=str(exc))
        except Exception:
            logging.exception('Échec du calcul %s',identifier)
            with self.lock:self.jobs[identifier].update(status='failed',error='Le calcul a échoué. Consultez le journal du serveur pour le diagnostic.')

    def simulate(self,request,folder):
        request=validate_request(request);cfg=self.config;n=request['robots'];seed=request['seed']
        # Rendez-vous 9 s, N² tours locaux, puis au moins 20 s de formation.
        preparation_steps=max(400,90+n*n+200)
        cfg=replace(cfg,horizon=preparation_steps)
        multi=request['controller'] in ('multisize','comparison_multisize') or n==64
        reconnection=ReconnectionConfig();width=6.
        if multi:
            from src.imitation.multisize import PROFILES,controlled_start
            profile=PROFILES[n];cfg=profile.config;preparation_steps=cfg.horizon;reconnection=profile.reconnection;width=profile.width
        shape=named_shape(request['shape'],n,width=width)
        policy=(None if request['controller']=='teacher' else self._multisize_controller() if multi else self._controller())
        learned_mode='raw' if multi else 'guarded'
        modes=[('teacher',None)] if policy is None else ([('teacher',None),(learned_mode,policy)] if request['controller'] in ('comparison','comparison_multisize') else [(learned_mode,policy)])
        navigation=request['scenario'] not in ('formation','transition')
        reference,world,nav_steps=navigation_scene(request['scenario']) if navigation else (None,DiskWorld(),None)
        requested_reference=reference
        if navigation:
            reference=plan_formation_route(shape,reference,world,cfg)
            nav_steps=int(np.ceil((reference.duration+12.)/cfg.physics.dt))
        initial=controlled_start(profile,seed) if multi else initial_positions(cfg,n,seed,world)
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
        manifest=dict(schema_version=1,kind='reference_demo',request=request,physics=asdict(cfg.physics),
            communication_radius=cfg.communication_radius,initial_positions=initial.tolist(),
            model_sha256=(self.multisize_hash if multi else self.model_hash) if policy is not None else None,
            experimental_model=multi,domain='controlled_near_rendezvous' if multi else 'dispersed_spawn_disk',width_m=width,
            experimental_robot_count=n!=12,guard='shared_braking_allocation_only' if multi and policy is not None else 'classical_safety_filter',
            playback='full_calculation_then_replay',reports={},files={})
        if navigation:
            manifest['navigation_planner']=dict(kind='classical_static_map_translation',
                requested_reference=asdict(requested_reference),planned_reference=asdict(reference),
                clearance_m=.35,maximum_reference_speed_m_s=.2,
                map_source='declared_demo_scene',learned_policy_receives_map=False)
        for mode,controller in modes:
            gains=ShapeGains(detour_only_on_constraint=True)
            if navigation:
                result=execute_navigation(shape,reference,world,cfg,initial,navigation_steps=nav_steps,
                    preparation_steps=preparation_steps,motion_policy=controller,seed=seed)
                metrics=evaluate_navigation(result);targets=result.local_targets
                route=[list(point) for point in reference.points]
            elif request['scenario']=='transition':
                second=named_shape(request['next_shape'],n,width=6.)
                result=execute_sequence((FormationStage(shape,preparation_steps),FormationStage(second,preparation_steps)),
                    cfg,initial,gains=gains,reconnection=ReconnectionConfig(),safety_filter=True,
                    motion_policy=controller,guard_motion=True,seed=seed)
                metrics=evaluate_sequence(result);targets=result.local_targets;route=[]
            else:
                result=execute_shape(shape,cfg,initial,gains=gains,reconnection=reconnection,
                    safety_filter=True,motion_policy=controller,guard_motion=mode!='raw',seed=seed)
                metrics=evaluate_shape(result,ShapeMetricConfig(.1,.05,1.,preparation_steps*cfg.physics.dt))
                targets=result.local_targets;route=[]
            episode=result.episode
            if navigation or request['scenario']=='transition':
                result.save(folder/f'{mode}.npz')
            else:
                with (folder/f'{mode}.npz').open('xb') as stream:
                    np.savez_compressed(stream,positions=episode.positions,velocities=episode.velocities,
                        accelerations=episode.accelerations,requested_accelerations=episode.requested_accelerations,
                        adjacency=episode.adjacency,times=episode.times,targets=result.targets,
                        assignments=result.assignments,local_targets=result.local_targets,
                        allocation_consensus=result.allocation_consensus,knowledge_sizes=result.knowledge_sizes,
                        messages_per_step=result.messages_per_step,phases=result.phases,
                        safety_infeasible=result.safety_infeasible,safety_interventions=result.safety_interventions,
                        metadata=json.dumps(dict(schema_version=1,kind='shape_demonstration',
                            config=asdict(cfg),seed=seed,mission=asdict(shape),gains=asdict(gains),
                            units=dict(position='m',time='s',velocity='m/s',acceleration='m/s^2'))))
            # Every physical frame is retained; playback does not round coordinates.
            display=dict(schema_version=1,mode=mode,request=request,metrics=metrics,physics=asdict(cfg.physics),
                times=episode.times.tolist(),positions=episode.positions.tolist(),velocities=episode.velocities.tolist(),
                targets=targets.tolist(),route=route,
                obstacles=[asdict(o) for o in world.obstacles])
            if multi:
                bank=[];identifiers=[];lookup={}
                for graph in episode.adjacency:
                    key=np.packbits(graph).tobytes()
                    if key not in lookup:lookup[key]=len(bank);bank.append(graph.astype(int).tolist())
                    identifiers.append(lookup[key])
                display.update(graph_bank=bank,graph_ids=identifiers)
            else:display['adjacency']=episode.adjacency.astype(int).tolist()
            display['phases']=(np.concatenate((result.preparation.phases,np.full(nav_steps,4))).tolist()
                               if navigation else result.phases.tolist())
            display['stage_ids']=result.stage_ids.tolist() if request['scenario']=='transition' else []
            (folder/f'{mode}.json').write_text(json.dumps(display,allow_nan=False),encoding='utf-8')
            with (folder/f'{mode}-trajectory.csv').open('x',encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(('time_s','robot','x_m','y_m','vx_m_s','vy_m_s'))
                for step,time in enumerate(episode.times):
                    for robot in range(n):writer.writerow((time,robot,*episode.positions[step,robot],*episode.velocities[step,robot]))
            manifest['reports'][mode]=metrics
            manifest['files'][mode]=dict(archive=f'{mode}.npz',display=f'{mode}.json',trajectory=f'{mode}-trajectory.csv')
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
        with zipfile.ZipFile(folder/'export.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            for path in folder.iterdir():
                if path.name!='export.zip':archive.write(path,path.name)
        return manifest


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,payload,content_type='application/json; charset=utf-8'):
            body=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode() if isinstance(payload,dict) else payload
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)

        def do_GET(self):
            if self.path=='/':return self.send(200,(ROOT/'demo/reference.html').read_bytes(),'text/html; charset=utf-8')
            if self.path=='/api/catalog':return self.send(200,service.catalog())
            if self.path=='/api/benchmarks':
                from demo.comparison_report import build_report
                try:return self.send(200,build_report())
                except (OSError,ValueError):return self.send(503,dict(error='Campagnes conservées indisponibles.'))
            match=re.fullmatch(r'/api/jobs/([0-9a-f]{32})(?:/(teacher.json|guarded.json|raw.json|export.zip))?',self.path)
            if match:
                identifier,filename=match.groups()
                try:job=service.status(identifier)
                except KeyError:return self.send(404,dict(error='Calcul introuvable.'))
                if filename is None:return self.send(200,job)
                if job['status']!='complete':return self.send(409,dict(error='Le calcul n’est pas terminé.'))
                file=service.output/identifier/filename
                if not file.is_file():return self.send(404,dict(error='Résultat indisponible pour ce contrôleur.'))
                return self.send(200,file.read_bytes(),'application/zip' if filename.endswith('.zip') else 'application/json; charset=utf-8')
            return self.send(404,dict(error='Page introuvable.'))

        def do_POST(self):
            if self.path!='/api/jobs':return self.send(404,dict(error='Page introuvable.'))
            # Local page only: foreign web pages cannot enqueue computations.
            origin=self.headers.get('Origin')
            if origin is not None and origin!=f'http://{self.headers.get("Host")}':
                return self.send(403,dict(error='Origine non autorisée.'))
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('Requête trop grande ou vide.')
                identifier=service.submit(json.loads(self.rfile.read(length)))
                return self.send(202,dict(id=identifier))
            except (ValueError,TypeError) as exc:return self.send(400,dict(error=str(exc)))

        def log_message(self,format,*args):logging.info(format,*args)
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args();logging.basicConfig(level=logging.INFO)
    service=ReferenceService();server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(service))
    print(f'Démonstration : http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();service.executor.shutdown(wait=True)


if __name__=='__main__':main()
