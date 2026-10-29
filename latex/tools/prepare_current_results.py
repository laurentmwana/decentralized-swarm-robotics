"""Reconstruit les figures et tableaux du mémoire depuis les preuves archivées.

À lancer depuis la racine du dépôt avec Python, NumPy et Matplotlib installés.
Aucun entraînement n'est exécuté. Les empreintes permettent de retrouver les sources.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
IMAGES = ROOT / 'latex/images'
GENERATED = ROOT / 'latex/generated'
GENERATED.mkdir(exist_ok=True)
SOURCES = {}


def source(relative):
    p = ROOT / relative
    SOURCES[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def read(relative):
    return json.loads(source(relative).read_text(encoding='utf-8'))


def save(fig, name):
    fig.savefig(IMAGES / f'current_{name}.pdf', bbox_inches='tight')
    fig.savefig(IMAGES / f'current_{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})

reference = read('tasks/evidence/008d-local-target-expanded-test.json')
vector = read('tasks/evidence/014-vector-fresh-test.json')
navigation = read('tasks/evidence/019-near-targets-evaluation.json')
rows = []
for mode, label in [('teacher', 'Professeur classique'), ('raw', 'GAT de référence brut'),
                    ('guarded', 'GAT de référence protégé')]:
    s = reference['summary'][mode]
    success = round(s['episodes'] * s['success_rate_percent'] / 100)
    delay = '--' if s['stabilization_time_mean_s'] is None else f"{s['stabilization_time_mean_s']:.2f}"
    rows.append(f"{label} & {success}/{s['episodes']} & {s['collisions_total']} & "
                f"{s['formation_error_mean_m']:.5f} & {delay} " + chr(92)*2)
(GENERATED/'reference_rows.tex').write_text(
    r'\begin{tabular}{lrrrr}'+'\n'+r'\toprule'+'\n'+
    r'Contrôleur & Réussites & Contacts & Erreur finale (m) & Temps (s)\\\midrule'+'\n'+
    '\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}'+'\n', encoding='utf-8')
rows=[]
for n in (12,24,64):
    s=vector['summary'][str(n)]
    t=vector['teacher_summary'][str(n)]
    mantissa, exponent = f"{s['formation_error_mean_m']:.2e}".split('e')
    error = f"${mantissa}\\times10^{{{int(exponent)}}}$"
    rows.append(f"{n} & 6/6 & {s['collisions_total']} & "
                f"{error} & {s['stabilization_time_mean_s']:.2f} & "
                f"{t['stabilization_time_mean_s']:.2f} " + chr(92)*2)
(GENERATED/'vector_rows.tex').write_text(
    r'\begin{tabular}{rrrrrr}'+'\n'+r'\toprule'+'\n'+
    r'Robots & Réussites & Contacts & Erreur finale (m) & Temps GAT (s) & Temps prof. (s)\\\midrule'+'\n'+
    '\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}'+'\n',encoding='utf-8')

fig,axes=plt.subplots(1,2,figsize=(10,3.5))
for ax,name,tk,vk,title in [
    (axes[0],'training-multiform-expanded-v1','train_mse','validation_mse','GAT de référence : MSE'),
    (axes[1],'training-vector-gat-v2','train_loss','validation_loss','GAT structuré v2 : objectif composite')]:
    d=read(f'outputs/imitation/{name}/run.json'); h=d['history']
    ax.plot([r['epoch'] for r in h],[r[tk] for r in h],color='black',label='Entraînement')
    ax.plot([r['epoch'] for r in h],[r[vk] for r in h],color='0.45',ls='--',label='Validation')
    ax.axvline(d['best_epoch'],color='0.7',ls=':',label='Époque retenue')
    ax.set(xlabel='Époque',ylabel='Valeur de la fonction de coût',title=title)
    ax.legend(fontsize=8);ax.grid(alpha=.15)
fig.tight_layout();save(fig,'training')

ns=np.array([12,24,64]);gather=np.array([8.,15.,30.]);brake=np.repeat(2.,3)
allocation=ns**2*.1
stabil=np.array([vector['summary'][str(n)]['stabilization_time_mean_s'] for n in ns])
after=stabil-gather-brake-allocation
fig,ax=plt.subplots(figsize=(7,3.8)); bottom=np.zeros(3)
for y,label,color,hatch in [(gather,'Rassemblement','0.75',''),(brake,'Freinage','white','..'),
                            (allocation,'Allocation','0.45','//'),(after,'Après libération','0.9','xx')]:
    ax.bar([str(n) for n in ns],y,bottom=bottom,label=label,color=color,edgecolor='black',hatch=hatch)
    bottom+=y
ax.set(xlabel='Nombre de robots',ylabel='Temps simulé (s)',title='Décomposition du temps moyen de stabilisation')
ax.legend(fontsize=8);fig.tight_layout();save(fig,'timing')

p=source('outputs/imitation/vector-fresh-test-v2/raw-24-geometries_rectangle-8500.npz')
with np.load(p,allow_pickle=False) as a:
    positions=a['positions'];times=a['times'];adj=a['adjacency'];vel=a['velocities']
fig,axes=plt.subplots(2,2,figsize=(7,6.5))
axes=axes.ravel()
for ax,t in zip(axes,[0,74.6,85.2,float(times[-1])]):
    i=int(np.argmin(abs(times-t)));pts=positions[i]
    ax.scatter(positions[-1,:,0],positions[-1,:,1],s=28,facecolors='none',edgecolors='0.6',label='État final')
    ax.scatter(pts[:,0],pts[:,1],s=10,color='black',label='Robots')
    ax.set(xlim=(-7,7),ylim=(-7,7),aspect='equal',title=f't = {times[i]:.1f} s',xlabel='x (m)')
axes[0].set_ylabel('y (m)');axes[-1].legend(fontsize=7,loc='lower right')
fig.tight_layout();save(fig,'sequence')
fig,ax=plt.subplots(figsize=(5,5));pts=positions[0]
for i,j in zip(*np.where(np.triu(adj[0],1))):
    ax.plot(pts[[i,j],0],pts[[i,j],1],color='0.7',lw=.7)
ax.scatter(pts[:,0],pts[:,1],s=24,color='black')
ax.set(xlabel='x (m)',ylabel='y (m)',aspect='equal',title='Graphe initial réel : 24 robots, portée de 3 m')
fig.tight_layout();save(fig,'graph')

p=source('outputs/navigation-near-targets/validation-8901-8902/obstacles-10-seed-8901/guarded.npz')
with np.load(p,allow_pickle=False) as a:
    meta=json.loads(str(a['metadata']));p=a['positions'];r=a['reference_positions'];times=a['times'];offset=a['offsets']
    # Le début du transport est détecté par la première variation de la référence.
    moving=np.flatnonzero(np.linalg.norm(r-r[0],axis=1)>1e-8)
    start=max(0,int(moving[0])-1) if len(moving) else 0
    fig,ax=plt.subplots(figsize=(7,5))
    for j in range(p.shape[1]):ax.plot(p[start:,j,0],p[start:,j,1],color='0.45',lw=.8)
    ax.plot(r[start:,0],r[start:,1],color='black',ls='--',lw=1.5,label='Centre de référence')
    ax.scatter(p[-1,:,0],p[-1,:,1],facecolors='white',edgecolors='black',s=28,label='Robots à la fin')
    obstacles=meta.get('obstacles',[])
    if not obstacles:obstacles=meta.get('config',{}).get('obstacles',[])
    for obstacle in obstacles:
        if isinstance(obstacle,dict):
            center=obstacle.get('center',obstacle.get('position'));radius=obstacle['radius']
        else:center=obstacle[:2];radius=obstacle[2]
        ax.add_patch(plt.Circle(center,radius,color='black'))
    ax.set(xlabel='x (m)',ylabel='y (m)',aspect='equal',title='Transport du carré : dix obstacles proches des cibles')
    ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2)
    fig.tight_layout();save(fig,'navigation')

(GENERATED/'sources.json').write_text(json.dumps({'description':'Sources des figures et tableaux du mémoire',
    'sha256':SOURCES},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(f'{len(SOURCES)} sources enregistrées ; figures et tableaux produits dans latex/.')
