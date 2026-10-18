# 14 — Glossaire

- **Essaim** : ensemble de robots simples appliquant la même règle locale.
- **Décentralisé** : la décision de mouvement n'utilise que l'état local, les voisins à portée et la consigne ; pas d'état global.
- **Consigne de mission** : forme à atteindre, connue de tous avant l'exécution (identifiant, nuage de points, éventuellement vecteur vers le contour).
- **Graphe de communication** : robots = nœuds, liens = paires à portée ; change à chaque pas.
- **GNN (réseau de neurones sur graphes)** : réseau qui calcule des messages entre voisins puis les agrège.
- **Passage de messages / tour** : une étape d'échange avec les voisins ; L tours = information venant de L sauts.
- **Agrégation par moyenne** : moyenne des messages reçus, insensible à l'ordre et au nombre de voisins.
- **Paramètres partagés** : mêmes poids pour tous les robots.
- **Simulateur différentiable** : simulation dont on peut calculer la dérivée de la perte par rapport aux poids.
- **Distance de Chamfer** : distance entre deux ensembles de points, dans les deux sens, sans affectation.
- **FPS (échantillonnage par point le plus éloigné)** : choix itératif du point le plus éloigné des déjà choisis.
- **Curriculum** : entraînement progressif du simple au complexe, en gardant les cibles précédentes.
- **Oubli catastrophique** : perte des acquis anciens en apprenant du nouveau.
- **Seuil τ / succès** : seuil de distance définissant la réussite ; sensible, toujours affiché.
- **Pas de convergence** : premier pas où la distance passe sous τ.
- **Baseline équitable** : contrôleur recevant la même information que le GNN.
- **Ablation** : retrait d'un composant (communication, profondeur) pour mesurer son apport.
- **Smoke (vérification rapide)** : exécution courte du pipeline, sans valeur scientifique.
