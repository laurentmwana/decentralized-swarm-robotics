# Journal des décisions

À remplir au fil du projet. Une entrée par décision : date, décision, raison, alternatives écartées.

| Date | Décision | Raison | Alternatives |
|------|----------|--------|--------------|
| 2026-10-11 | Conserver provisoirement l'intitulé « Contrôle décentralisé d'un essaim de robots par réseaux de neurones cellulaires (CeNN-Inspired Swarm Robotics) » | Il correspond à l'intitulé déposé et l'inspiration cellulaire reste cohérente avec les interactions locales. | Renommer dès maintenant en « réseaux de neurones sur graphes » ; écarté tant que le professeur ne l'a pas validé. |
| 2026-10-11 | Utiliser un repère commun 2D pour tous les robots dans la première version du simulateur | Simplifie le cadrage et l'implémentation initiale ; cette hypothèse sera explicitement documentée comme une limite. | Repère propre à chaque robot, plus réaliste mais plus complexe. |
| 2026-10-11 | Utiliser l'option A : chaque robot reçoit un identifiant partagé de la forme cible | Point de départ simple, reproductible et compatible avec une politique décentralisée fondée sur des observations locales et une consigne commune. | Option B (consigne portée par quelques robots) et option C (signal conditionnant un réseau unique), réservées comme extensions. |
| 2026-10-11 | Utiliser une dynamique 2D à vitesse directe et une arène bornée pour le premier simulateur | Base minimale testable pour valider voisinage, pannes et règles locales avant d'ajouter l'inertie ou des modèles robotiques réalistes. | Dynamique avec inertie ; arène illimitée. |
| 2026-10-11 | Comparer d'abord le futur GNN à une règle locale de cohésion et répulsion | Cette règle est interprétable, ne dépend que du voisinage et valide le comportement minimal du simulateur. | Commencer directement par le GNN, sans référence comportementale. |
| 2026-10-11 | Générer les formes à 128×128 pixels et échantillonner 20 points par farthest-point sampling | Le format est lisible, reproductible et produit des nuages de points régulièrement répartis pour les premiers essais à 20 robots. | Échantillonnage purement aléatoire ; images à résolution variable. |
| | Méthode d'apprentissage (directe ou par renforcement) | | |
| | Nombre de robots à l'entraînement | | |
