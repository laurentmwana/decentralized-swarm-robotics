"""Vérifier le plan imposé, les annonces et la position des légendes.

Complément à check_manuscript.py ; la pagination se vérifie sur le PDF compilé.
"""
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parents[2]
CHAPTERS=ROOT/'latex/chapters'

def read(name):return (CHAPTERS/(name+'.tex')).read_text(encoding='utf-8')

def headings(text,level='section',star=False):
    token=r'\*' if star else ''
    return re.findall(r'\\'+level+token+r'\{([^}]+)\}',text)

intro=['Contexte de l\'étude','Problématique','Hypothèses','Objectifs poursuivis',
       'Méthodes et techniques utilisées','Contribution de l\'étude','Plan sommaire et subdivision du travail']
assert headings(read('introduction'),star=True)==intro
assert headings(read('etat-art'))==['Introduction','Méthodes traditionnelles','Méthodes avancées',
                                    'Quelques travaux connexes','Positionnement','Conclusion partielle']
studies=read('etat-art').split(r'\section{Quelques travaux connexes}')[1].split(r'\section{Positionnement}')[0]
assert len(headings(studies,'subsection'))==3
assert headings(studies,'subsubsection')==[
    "Contexte de l'étude et questions de recherche",'Méthodologie adoptée','Aperçu sur les résultats','Critique']*3
for name in ['concept','etat-art','methode','experiences']:
    sections=headings(read(name))
    assert sections[0]=='Introduction' and sections[-1]=='Conclusion partielle',name
    conclusion=read(name).split(r'\section{Conclusion partielle}')[1].strip()
    assert len(re.split(r'\n\s*\n',conclusion))==1,(name,'conclusion en un paragraphe')
    assert ('chapitre suivant' in conclusion.lower() or 'discussion suivante' in conclusion.lower()),(name,'annonce du chapitre suivant')
discussion=read('discussion')
assert r'\chapter*{Discussion}' in discussion
assert r'\chapter*{Conclusion générale et perspectives}' in discussion
assert discussion.index(r'\chapter*{Discussion}')<discussion.index(r'\chapter*{Conclusion générale et perspectives}')
main=(ROOT/'latex/main.tex').read_text(encoding='utf-8')
for name in ['introduction','concept','etat-art','methode','experiences','discussion']:
    assert '\\include{chapters/'+name+'}' in main
for name in ['garde','epigraphe','dedicace','remerciements','abbreviations','sumaires']:
    assert '\\include{frontmatter/'+name+'}' in main
assert r'\onehalfspacing' in main and r'12pt,a4paper' in main and r'margin=2.5cm' in main
counts={'figures':0,'tables':0,'equation_groups':0}
for name in ['introduction','concept','etat-art','methode','experiences','discussion']:
    s=read(name)
    assert not re.search(r'\\(?:chapter|section|subsection|subsubsection)\*?\{[^\n]*\\cite',s)
    for environment,key in [('figure','figures'),('table','tables'),('equation','equation_groups')]:
        for m in re.finditer(r'\\begin\{'+environment+r'\}[\s\S]*?\\end\{'+environment+r'\}',s):
            block=m.group();labels=re.findall(r'\\label\{([^}]+)\}',block)
            assert len(labels)==1,(name,environment,'un label requis')
            command='eqref' if environment=='equation' else 'ref'
            assert '\\'+command+'{'+labels[0]+'}' in s[:m.start()],(name,labels[0],'annonce absente')
            if environment=='table':
                content=[v for v in (block.find(r'\begin{tabular'),block.find(r'\input{generated/')) if v>=0]
                assert block.index(r'\caption')<min(content),(name,'légende sous le tableau')
            if environment=='figure':assert block.index(r'\caption')>block.index(r'\includegraphics')
            counts[key]+=1
abbreviations=(ROOT/'latex/frontmatter/abbreviations.tex').read_text(encoding='utf-8')
assert re.findall(r'^(GAT|GNN|MSE|QP|ReLU|SiLU) &',abbreviations,re.M)==['GAT','GNN','MSE','QP','ReLU','SiLU']
result={'status':'passed','numbered_chapters':4,'intro_required_sections':7,'related_studies':3,
        'required_subheadings_per_study':4,'discussion_and_conclusion_separate':True,
        'existing_directories_preserved':['latex/chapters','latex/frontmatter'],
        'layout':'12 pt, interligne 1,5, marges de 2,5 cm conservés',**counts}
(ROOT/'tasks/evidence/022-professor-plan.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=True))
