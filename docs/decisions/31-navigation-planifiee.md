# 31 — Planifier le transport de toute la formation

Le filtre local protège les disques physiques mais ne calcule pas un trajet collectif. Sur la destination obstruée du carré, une place demandée se trouve dans un obstacle : entraîner plus longtemps ne rend pas cet objectif réalisable.

Nous ajoutons un planificateur classique séparé du GAT. Pour chaque place et obstacle, il construit le disque interdit correspondant dans l'espace des translations communes. Les limites de l'arène tiennent compte des extrémités de la formation. Une recherche sur grille de 0,10 m puis une simplification par visibilité vérifient chaque segment balayé. L'interpolation quintique s'arrête aux points de passage ; les durées limitent la vitesse de référence à 0,20 m/s. La destination originale n'est pas modifiée. Une destination occupée est signalée avant le lancement.

La marge planifiée de 0,35 m réserve de l'espace au suivi ; elle n'est pas une garantie sur la trajectoire réelle. Les métriques et contacts balayés restent calculés sur les positions simulées. Le GAT protégé du carré atteint une marge réelle minimale de 0,34348 m dans la régression 7700, sans contact. Les filtres, bornes physiques et critères finaux restent actifs.

Le manifeste exporte référence demandée, référence planifiée, source de carte, vitesse et marge. Le contrôleur neuronal ne reçoit pas cette carte ; il suit sa cible mobile avec ses observations locales antérieures. La navigation est donc hybride avec planification sur carte connue, pas une navigation neuronale autonome en environnement inconnu.

La tâche 015 fournit la correction et sa régression. La tâche 016 vérifie trois formes sur de nouveaux départs et deux parcours complémentaires. Les anciennes campagnes 009 restent inchangées et sont affichées comme historiques. Le modèle brut structuré 014 reste limité à la formation sans navigation.
