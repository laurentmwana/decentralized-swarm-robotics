# 009 — Navigation collective puis obstacles

Statut : implémentée et évaluée, avec limites explicites. 412 tests ciblés réussis ; audit de 40 trajectoires passé.

Pilote et validation : pour chaque contrôleur et chaque campagne, 6/6 épisodes nominaux réussis, 0/4 difficiles, aucune collision observée. Les échecs de maintien de forme sont conservés. GAT protégé par un filtre classique, repère mobile et anticipation de référence explicites ; aucun nouvel entraînement. Les poids restent inchangés.

Décision et reproduction : [25-navigation-obstacles](../docs/decisions/25-navigation-obstacles.md). Preuves : `tasks/evidence/009-navigation-{pilot,validation,audit}.json`. Prochaine étape : tâche 010, interface et exports ; la planification autonome autour des obstacles reste hors du périmètre livré.

## Base existante

Le catalogue, les places locales et le prototype GAT protégé sont disponibles. Les positions, vitesses, communications et temps sont conservés entre missions. Le GAT brut reste une comparaison expérimentale dont la sécurité n'est pas validée.

## Travail attendu

1. Définir une destination ou trajectoire commune de référence et une formation transportée, avec unités, vitesse et conditions d'arrivée explicites.
2. Commander chaque robot à partir de son état, de sa cible locale mobile et des informations autorisées des voisins. Déclarer toute nouvelle information de vitesse de cible avant son utilisation par le modèle.
3. Mesurer séparément erreur de formation et progression vers la destination ; une formation correcte restée immobile ne constitue pas une navigation réussie.
4. Ajouter ensuite des obstacles statiques avec géométrie physique et contrat d'observation explicite. Ne pas donner au contrôleur une vue globale cachée.
5. Détecter les contacts avec obstacles et robots, y compris les traversées entre états ; conserver les échecs de passage et les délais dépassés.
6. Comparer professeur et GAT protégé sur les mêmes départs, trajectoires et obstacles ; identifier le rôle des filtres classiques.
7. Tester puis valider sur de nouveaux scénarios avant intégration de l'interface 010.

La tâche ne doit pas modifier les poids à partir des tests neuronaux déjà examinés. Toute extension d'entrées ou nouvel apprentissage aura son propre schéma et de nouveaux ensembles réservés.
