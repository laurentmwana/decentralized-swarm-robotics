# Étape 2 — Simulateur d'essaim

## Objectif
Construire le monde 2D dans lequel les robots bougent, se perçoivent et peuvent tomber en panne. Tout le reste en dépend.

## Ce que le simulateur doit faire
1. **Représenter les robots** : chaque robot a une position, une vitesse, un identifiant, un état (actif ou en panne).
2. **Faire avancer le temps** : à chaque pas, chaque robot reçoit une commande de mouvement (direction et vitesse), limitée par une vitesse maximale, puis sa position est mise à jour.
3. **Calculer le voisinage** : deux robots sont voisins si leur distance est inférieure au **rayon de communication**. Ce voisinage forme le graphe, recalculé à chaque pas.
4. **Gérer les pannes** : possibilité de retirer des robots à un moment donné (au hasard ou dans une zone).
5. **Détecter les collisions** : compter les paires de robots trop proches.
6. **Visualiser** : image par pas de temps et animation.

## Paramètres (tous dans la configuration)
- nombre de robots
- taille de l'arène
- rayon de communication
- vitesse maximale
- pas de temps
- nombre de pas par épisode
- distance de sécurité pour les collisions
- bruit éventuel sur les mesures (pour tester la robustesse)

## Décisions à prendre et noter dans `decisions.md`
- **Référentiel** : les robots partagent-ils un repère commun (même axe nord pour tous), ou chacun mesure ses voisins dans son propre repère ? Le repère commun est plus simple ; le repère propre est plus réaliste. Commencer par le repère commun, noter l'hypothèse.
- **Arène** : bornée (murs) ou illimitée ?
- **Dynamique** : déplacement direct (le robot va où on lui dit) ou avec inertie ? Commencer simple.

## Références de comportement à coder ici (utilisées en étape 6)
- **Rassemblement par règles** : chaque robot se rapproche du centre de ses voisins.
- **Répulsion** : s'éloigner des voisins trop proches.
Ces deux règles servent de premier test du simulateur et de comparaison plus tard.

## Tests à écrire
- Deux robots éloignés ne sont pas voisins, deux robots proches le sont.
- Un robot ne dépasse jamais la vitesse maximale.
- Un robot en panne ne bouge plus et ne communique plus.
- Même graine aléatoire = même résultat.

## À faire
- [ ] Module de dynamique
- [ ] Module de voisinage
- [ ] Module de pannes
- [ ] Module de collisions
- [ ] Visualisation animée
- [ ] Règles simples de rassemblement et de répulsion
- [ ] Tests

## Critères de réussite
Avec les règles simples, on voit des robots dispersés se regrouper sans se superposer, et l'animation le montre clairement.

## Pièges
- Recalculer les voisins de façon lente (utiliser une structure de recherche spatiale dès que le nombre de robots grandit).
- Mélanger dans le même code la simulation et le modèle : garder les deux séparés.
