# 08 — Configurations

Tout paramètre numérique est dans un fichier versionné ; l'exécution sauvegarde la configuration **effective** à côté des résultats.

| Fichier | Objet | Valeurs notables |
|---|---|---|
| shapes.yaml | jeu de formes standard | image 128 × 128, 20 points |
| shapes-complex.yaml | formes complexes | 50 points par forme |
| simulator.yaml | démonstration de la règle locale | portée, vitesse max |
| training-circle.yaml | entraînement du cercle | 20 robots, 80 itérations, graine 41 |
| curriculum.yaml | curriculum réduit | 4 niveaux de 30 itérations |
| curriculum-complex.yaml | campagne complexe | 50 robots, 60 pas, 3 tours |
| curriculum-complex-smoke.yaml | vérification rapide | pipeline uniquement |
| evaluation-pilot.yaml | protocole pilote | 10 épisodes, 4 retraits, 3 tailles |

Hyperparamètres : voir le tableau de `05-entrainement.md`.

## Règles
- Ne jamais écrire une valeur « en dur » qui change un résultat.
- Une nouvelle expérience = un nouveau fichier ou une surcharge explicite, jamais une modification silencieuse.
- Ajouter dans la configuration d'évaluation : nombre de répétitions, durée d'épisode, seuil τ.
