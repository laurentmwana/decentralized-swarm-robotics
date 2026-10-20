# Explications détaillées du travail CeNN-Inspired Swarm Robotics

Ce document fournit une explication approfondie du travail présenté dans le mémoire "Contrôle décentralisé d'un essaim de robots par réseaux de neurones cellulaires - CeNN-Inspired Swarm Robotics", en mettant l'accent sur l'aspect CeNN (Cellular Neural Network) inspiré.

## 1. Introduction aux réseaux de neurones cellulaires (CeNN)

### 1.1 Définition et origine
Les **réseaux de neurones cellulaires** (Cellular Neural Networks - CeNN) ont été introduits par Leon O. Chua et Lin Yang en 1988 comme un paradigme de calcul parallèle pour le traitement d'images et le calcul spatio-temporel. Contrairement aux réseaux de neurones traditionnels où les connexions sont globalement structurées, les CeNN sont composés d'un réseau régulier de cellules identiquement structurées où chaque cellule interagit uniquement avec ses voisins immédiats dans une topologie définie.

### 1.2 Équation fondamentale d'un CeNN
La dynamique d'un CeNN classique est décrite par l'équation différentielle suivante pour chaque cellule \(C_{ij}\) dans un réseau 2D :

\[
\frac{dx_{ij}(t)}{dt} = -x_{ij}(t) + \sum_{C_{kl} \in \mathcal{N}(r_{ij})} A_{(ij)(kl)} y_{kl}(t) + \sum_{C_{kl} \in \mathcal{N}(r_{ij})} B_{(ij)(kl)} u_{kl}(t) + I
\]

où :
- \(x_{ij}(t)\) est l'état de la cellule \(C_{ij}\) au temps \(t\)
- \(y_{ij}(t) = f(x_{ij}(t)) = \frac{1}{2}(|x_{ij}(t)+1| - |x_{ij}(t)-1|)\) est la sortie (fonction d'activation piecewise linéaire)
- \(\mathcal{N}(r_{ij})\) est le voisinage de la cellule \(C_{ij}\) dans un rayon \(r\)
- \(A\) et \(B\) sont les matrices de rétroaction et de contrôle respectivement
- \(u_{ij}(t)\) est l'entrée externe à la cellule \(C_{ij}\)
- \(I\) est un biais seuil

### 1.3 Propriétés clés des CeNN
1. **Localité** : Chaque cellule interagit uniquement avec ses voisins dans un rayon fini
2. **Homogénéité** : Toutes les cellules ont la même structure et les mêmes paramètres (dans la version classique)
3. **Parallelisme inhérent** : Toutes les cellules peuvent être mises à jour simultanément
4. **Stabilité** : Sous certaines conditions sur les matrices A et B, le système est globalement stable
5. **Capacité de traitement d'images** : Les CeNN excellent pour des tâches comme la détection de contours, l'érosion, la dilatation, etc.

## 2. L'inspiration CeNN dans ce travail

### 2.1 Pourquoi CeNN pour la robotique essaim ?
Les essaims de robots partagent plusieurs caractéristiques fondamentales avec les CeNN :
- **Décentralisation** : Aucun contrôle central, chaque agent prend des décisions basé sur l'information locale
- **Interaction locale** : Les robots interagissent principalement avec leurs voisins proches
- **Homogénéité fonctionnelle** : Tous les robots exécutent généralement le même algorithme de contrôle
- **Émergence de comportements globaux** : Des comportements complexes émergent d'interactions locales simples
- **Robustesse** : La décentralisation rend le système résilient aux défaillances individuelles

### 2.2 Analogies entre CeNN traditionnel et notre approche
| Aspect | CeNN traditionnel | Notre approche (CeNN-inspiré Swarm-GNN) |
|--------|-------------------|------------------------------------------|
| **Unité de base** | Cellule identique | Robot identique (contrôleur partagé) |
| **Topologie** | Grille régulière fixe | Graphe de communication dynamique basé sur la portée |
| **État** | \(x_{ij}(t)\) (état continu) | \(h_i^t\) (état latent du GNN) |
| **Sortie** | \(y_{ij}(t)\) (fonction piecewise linéaire) | \(u_i^t\) (commande de vitesse bornée) |
| **Mécanisme de mise à jour** | Équation différentielle continue | Passage de messages discret dans un GNN |
| **Champ de rétroaction** | Matrice A fixe | Poids du GNN partagés et appris |
| **Champ de contrôle** | Matrice B fixe | Encodage de la consigne cible |
| **Entrée externe** | \(u_{ij}(t)\) | Vitesse actuelle + géométrie cible + identifiant de forme |
| **Voisiange** | Rayon fixe \(r\) | Portée de communication \(r_c\) (dynamic graph) |
| **Stabilité** | Analyse Lyapunov | Contraintes de vitesse + perte d'entraînement |
| **Traitement** | Traitement d'images/spatio-temporel | Formation de formes par essaim de robots |

### 2.3 Différences et innovations apportées
Bien que notre approche soit inspirée des CeNN, elle introduit plusieurs innovations importantes :

1. **Topologie dynamique** : Contrairement au voisinage fixe des CeNN classiques, notre graphe de communication évolue en temps réel basé sur les positions des robots
2. **Apprentissage des paramètres** : Les poids du GNN sont appris via l'entraînement plutôt que fixé manuellement comme dans les CeNN classiques
3. **Multimodalité de l'entrée** : Chaque robot intègre plusieurs sources d'information (vitesse propre, consigne cible, géométrie locale, identifiant de forme)
4. **Passage de messages généralisé** : Notre mécanisme de message passing est plus flexible que la somme pondérée linéaire des CeNN
5. **Fonction de sortie bornée** : Nous utilisons une tangente hyperbolique pour borner les commandes, plutôt qu'une fonction piecewise linéaire
6. **Objectif d'apprentissage explicite** : Nous optimisons une fonction de perte composée plutôt que de concevoir manuellement les templates

## 3. Architecture du contrôleur CeNN-inspiré

### 3.1 Vue d'ensemble
Notre contrôleur suit le paradigme CeNN-inspiré suivant :
1. Chaque robot maintient un état latent qui représente son "état interne"
2. À chaque pas de simulation, les robots échangent des messages avec leurs voisins (dans la portée de communication)
3. Ces messages sont agrégés (par moyenne) pour mettre à jour l'état latent de chaque robot
4. L'état latent final est décodé en une commande de mouvement
5. Tout le processus utilise les mêmes paramètres partagés pour tous les robots

### 3.2 Équations du contrôleur GNN (équivalent CeNN-discret)
Les opérations de notre contrôleur peuvent être écrites sous une forme analogue aux équations de CeNN :

**Étape d'encodage (initialisation de l'état)** :
\[
h_i^0 = \phi_{enc}(v_i^t, g_i, e_i)
\]
où :
- \(v_i^t\) : vitesse actuelle du robot i
- \(g_i\) : vecteur vers le point de contour cible le plus proche
- \(e_i\) : embedding de l'identifiant de forme
- \(\phi_{enc}\) : réseau de neurones encodeur

**Étape de passage de messages (équivalent au terme de rétroaction du CeNN)** :
Pour \(\ell = 0\) à \(L-1\) (L tours de message passing) :
\[
m_{ij}^{\ell} = \phi_{msg}(h_i^{\ell}, h_j^{\ell}, p_j - p_i)
\]
\[
\bar{m}_i^{\ell} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} m_{ij}^{\ell}
\]
\[
h_i^{\ell+1} = \phi_{upd}(h_i^{\ell}, \bar{m}_i^{\ell})
\]
où :
- \(\mathcal{N}(i)\) : ensemble des voisins du robot i (dans portée \(r_c\))
- \(p_j - p_i\) : position relative (encodage géométrique de l'arête)
- \(\phi_{msg}\) : fonction de génération de message
- \(\phi_{upd}\) : fonction de mise à jour d'état

**Étape de décodage (équivalent au terme de sortie du CeNN)** :
\[
u_i^T = \phi_{dec}(h_i^T)
\]
\[
\text{avec } \|\cdot\| \text{ borné par } v_{max}
\]
où :
- \(u_i^T\) : commande de vitesse finale du robot i
- \(\phi_{dec}\) : réseau de neurones décodeur
- \(v_{max}\) : vitesse maximale autorisée

### 3.3 Comparaison avec un CeNN discret
Si nous considérons une version discrétisée en temps du CeNN traditionnel, nous obtenons :

\[
x_{ij}^{t+1} = (1-\alpha)x_{ij}^{t} + \alpha \left[ -\beta x_{ij}^{t} + \sum_{C_{kl} \in \mathcal{N}(r_{ij})} A_{(ij)(kl)} y_{kl}^{t} + \sum_{C_{kl} \in \mathcal{N}(r_{ij})} B_{(ij)(kl)} u_{kl}^{t} + I \right]
\]

Notre approche peut être vue comme une généralisation où :
- Le terme \(-\beta x_{ij}^{t}\) est remplacé par notre fonction de mise à jour \(\phi_{upd}\)
- La somme pondérée sur les voisins est remplacée par notre mécanisme de message passing agrégé
- Les entrées externes \(u_{kl}^{t}\) sont remplacées par l'encodage de la consigne cible et de la géométrie locale
- Les matrices \(A\) et \(B\) sont remplacées par des réseaux de neurones appris

## 4. Aspects d'apprentissage et d'optimisation

### 4.1 Apprentissage direct à travers la simulation
Contrairement aux CeNN classiques où les paramètres sont souvent choisis manuellement ou via des règles d'apprentissage spécifiques (comme l'apprentissage de Hebb), notre approche utilise :

- **Apprentissage supervisé par renforcement indirect** : Une seule perte par épisode calculée sur l'essaim entier
- **Rétropropagation à travers le temps** : La perte est rétropropagée à travers tous les pas de simulation grâce à la différentiabilité du simulateur
- **Optimisation par gradient** : Utilisation de l'optimiseur Adam pour mettre à jour les paramètres partagés

Cette approche permet d'apprendre des comportements complexes qui seraient difficiles à concevoir manuellement avec des templates CeNN traditionnels.

### 4.2 Fonction de perte et lien avec l'énergie des CeNN
Dans les CeNN classiques, la stabilité est souvent analysée via une fonction d'énergie ou de Lyapunov. Dans notre travail, nous optimisons explicitement une fonction de perte composée :

\[
\mathcal{L} = \lambda_f \cdot d_C(P,Q) + \lambda_c \cdot \mathcal{L}_{col} + \lambda_u \cdot \mathcal{L}_{move} + \lambda_r \cdot \mathcal{L}_{cov}
\]

Cette fonction peut être interprétée comme une généralisation de la fonction d'énergie des CeNN où :
- \(d_C(P,Q)\) (distance de Chamfer) encourage l'adéquation à la cible (analogue à un terme d'attraction)
- \(\mathcal{L}_{col}\) pénalise les états indésirables (proximités dangereuses)
- \(\mathcal{L}_{move}\) régularise la dynamique (évite les oscillations)
- \(\mathcal{L}_{cov}\) encourage une répartition uniforme (analogue à un terme d'entropie ou de régularisation)

## 5. Caractéristiques émergentes et analogies avec les phénomènes CeNN

### 5.1 Ondes de propagation d'information
Comme dans les CeNN où des signaux peuvent se propager sous forme d'ondes à travers le réseau, dans notre système :
- L'information se propage à travers le graphe de communication via le passage de messages
- Avec L tours de message passing, un robot peut intégrer de l'information provenant de voisins à L sauts
- Cette propriété permet la coordination à longue distance malgré les interactions locales uniquement

### 5.2 Motifs stables et formation de formes
Dans les CeNN, certains paramètres produisent des motifs stables comme des points, des lignes ou des ondulations. Dans notre système :
- Les formes cibles agissent comme des "attracteurs" pour l'essaim
- La perte de forme guide l'essaim vers des configurations correspondant à ces attracteurs
- Les différents termes de perte travaillent ensemble pour stabiliser des formations spécifiques

### 5.3 Robustesse aux perturbations
Les CeNN sont connus pour leur robustesse aux variations de paramètres et au bruit. Notre système hérite de certaines de ces propriétés :
- La décentralisation rend le système résilient à la perte de certains robots
- Le partage des paramètres permet une généralisation à différentes tailles d'essaim
- L'apprentissage à travers des scénarios variés (curriculum) améliore la robustesse

## 6. Contributions spécifiques liées à l'inspiration CeNN

### 6.1 Théoriques
1. **Formalisation du lien CeNN-GNN** : Nous avons établi une correspondance claire entre les opérations des CeNN classiques et celles des réseaux de neurones sur graphes appliqués à la robotique essaim
2. **Generalisation du paradigme CeNN** : Nous avons étendu le concept de CeNN au-delà des grilles régulières fixes vers des topologies dynamiques et des entrées multimodales
3. **Cadre d'apprentissage pour CeNN-inspiré** : Nous avons proposé une méthodologie d'apprentissage direct à travers la simulation pour des systèmes CeNN-inspirés

### 6.2 Pratiques
1. **Implémentation modulaire** : Séparation claire entre simulation, contrôle, entraînement et évaluation, facilitant l'expérimentation avec différentes variantes CeNN-inspirées
2. **Plateforme de validation** : Notre implémentation sert de banc d'essai pour tester différentes architectures de message passing inspirées des CeNN
3. **Visualisation des flux d'information** : Nous pouvons analyser comment l'information se propage dans l'essaim, analogue à l'étude de la propagation d'ondes dans les CeNN classiques

### 6.3 Applications potentielles au-delà de ce travail
Les enseignements de ce travail peuvent être appliqués à :
- Autres formes de robotique décentralisée (drônes, véhicules autonomes)
- Systèmes de capteurs distribués
- Calcul parallèle inspiré par les phénomènes biologiques
- Conception de matériels neuromorphiques pour le traitement spatio-temporel

## 7. Limites et perspectives futures liées à l'aspect CeNN

### 7.1 Limites actuelles
1. **Topologie de voisinage simplifiée** : Nous utilisons un graphe de communication basé uniquement sur la distance euclidienne, alors que les vrais CeNN peuvent avoir des topologies de voisinage plus complexes
2. **Homogénéité stricte** : Tous les robots utilisent exactement le même contrôleur, alors que certains variantes de CeNN permettent une hétérogénéité contrôlée
3. **Absence de dynamiques temporelles avancées** : Notre modèle ne capture pas pleinement les richesses dynamiques possibles dans certains CeNN (comme le comportement chaotique ou oscillatoire)

### 7.2 Directions de recherche future
1. **CeNN à topologie adaptative** : Permettre au graphe de voisinage de s'adapter non seulement basé sur la distance mais aussi sur l'historique d'interaction
2. **CeNN hétérogène** : Introduire des variations contrôlées dans les paramètres du contrôleur pour des rôles spécialisés dans l'essaim
3. **Intégration de modèles CeNN plus riches** : Explorer l'utilisation de modèles CeNN avec mémoires à long terme ou des dynamiques plus complexes
4. **Analyse de stabilité CeNN-formelle** : Appliquer les outils d'analyse de stabilité des CeNN classiques pour garantir des propriétés de notre système
5. **Implémentation matérielle** : Explorer la mise en œuvre de notre contrôleur sur des plateformes neuromorphiques inspirées des architectures CeNN

## 8. Conclusion sur l'aspect CeNN

Notre travail démontre que le paradigme des Cellular Neural Networks fournit un cadre théorique et pratique puissant pour concevoir des systèmes de contrôle décentralisé en robotique essaim. En interprétant notre contrôleur GNN comme une généralisation dynamique et apprenable des CeNN classiques, nous bénéficions de :

1. **Un fondement théorique solide** : S'appuyant sur des décennies de recherche sur les CeNN
2. **Des principes de conception éprouvés** : Localité, partage de paramètres, émergence
3. **Une voie claire pour l'extension** : En nous inspirant des variantes avancées des CeNN
4. **Un langage commun** : Facilitant la communication avec les communautés des CeNN et du calcul neuromorphe

L'approche CeNN-inspirée ne se limite pas à une simple analogie ; elle fournit un cadre structuré pour penser, concevoir et analyser des systèmes d'essaim robotique où l'intelligence émerge d'interactions locales simples, tout en bénéficiant des advances récentes en apprentissage profond et en réseaux de neurones sur graphes.

Cette perspective CeNN-inspirée enrichit considérablement notre compréhension du système et ouvre des voies prometteuses pour la recherche future à l'intersection de la robotique essaim, des réseaux de neurones cellulaires et du calcul distribué.