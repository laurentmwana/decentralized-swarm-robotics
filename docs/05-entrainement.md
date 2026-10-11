# Étape 5 — Entraînement

## Objectif
Faire apprendre au réseau à produire les formes, en l'entraînant dans le simulateur.

## Idée générale
On lance des simulations de quelques dizaines de pas avec des départs aléatoires. À la fin (ou à chaque pas), on mesure la qualité du résultat, et on ajuste le réseau pour qu'il fasse mieux la fois suivante. On répète jusqu'à stabilisation.

## Ce que l'objectif récompense (en mots)
1. **Forme** : les robots couvrent bien la forme cible, sans trou et sans robot hors forme.
2. **Régularité** : les robots sont répartis de façon régulière sur la forme.
3. **Sécurité** : pas de collisions, distance minimale respectée.
4. **Sobriété** : mouvements raisonnables, pas de tremblements.
5. **Rapidité** : la forme est atteinte vite.

Pondérer ces éléments par des coefficients de la configuration. Les ajuster un à un en observant les animations.

## Méthode d'apprentissage
Deux voies possibles, à choisir et justifier :
- **Apprentissage direct à travers la simulation** : si le simulateur est écrit de façon à pouvoir propager les corrections (calcul différentiable), le réseau est ajusté directement à partir de l'objectif. Plus rapide et plus stable quand c'est possible.
- **Apprentissage par renforcement multi-agents** : le réseau apprend par essais et récompenses. Plus général mais plus lent et plus instable.

Recommandation : commencer par la première si le simulateur le permet, garder la seconde comme alternative.

## Programme d'apprentissage progressif
1. **Niveau 1** : cercle seul, peu de robots, départs proches.
2. **Niveau 2** : formes géométriques variées.
3. **Niveau 3** : chiffres puis lettres.
4. **Niveau 4** : nombres.
5. **Niveau 5** : ajouter pannes pendant l'entraînement et nombre de robots variable.
Ne passer au niveau suivant que si le précédent est atteint de façon fiable.

## Paramètres à régler
- vitesse d'apprentissage
- taille des lots de simulations
- longueur d'un épisode
- nombre de robots à l'entraînement (par exemple entre 10 et 20 au début)
- coefficients de l'objectif
- graine aléatoire

## Suivi
- Courbe de la perte sur entraînement et validation.
- Animation d'un exemple fixe toutes les N itérations pour voir l'évolution.
- Sauvegarde régulière des modèles (le meilleur sur la validation et le dernier).
- Journal de chaque lancement : configuration, graine, durée, résultat.

## Problèmes fréquents et réponses
- **Les robots se collent tous au même endroit** : augmenter la sanction de collision.
- **Les robots tremblent** : sanctionner les changements brusques de mouvement.
- **Le réseau n'apprend rien** : revenir à un niveau plus simple, vérifier les données d'entrée, réduire la vitesse d'apprentissage.
- **Bon sur l'entraînement, mauvais sur le test** : plus de variété dans les départs et les formes, réduire la taille du modèle.

## À faire
- [ ] Fonction d'objectif et ses composantes séparées
- [ ] Boucle d'entraînement
- [ ] Programme d'apprentissage progressif
- [ ] Suivi et sauvegardes
- [ ] Un premier entraînement complet sur le cercle
- [ ] Entraînement sur l'ensemble des niveaux

## Critères de réussite
Sur le cercle, la forme est obtenue de façon visible à partir de départs aléatoires ; la courbe de validation s'améliore puis se stabilise.

## Première exécution validée

La première exécution utilise l'apprentissage direct à travers le simulateur différentiable, une perte de Chamfer symétrique pour la forme, une pénalité de collision et une pénalité de mouvement. La configuration `configs/training-circle.yaml` entraîne 20 robots sur le cercle avec la graine 41. Les sorties reproductibles sont écrites dans `results/training-circle/` ; l'apprentissage progressif des autres formes reste à réaliser.

## Pièges
- Régler les coefficients en regardant le jeu de test.
- Lancer de longues sessions avant d'avoir validé une version réduite.
- Ne pas noter les graines : résultats non reproductibles.
