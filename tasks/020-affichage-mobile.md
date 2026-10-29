# 020 — Rendre la simulation visible sur petit écran

Statut : terminé. Vérification réelle à 64 robots : animation visible, temps de lecture qui progresse et accès au carré final. **17 tests ciblés réussis**, dont deux régressions de l'horloge de lecture.

Le serveur et le dessin fonctionnaient, mais la disposition mobile plaçait le long formulaire avant l'arène. La première vue n'affichait donc que les réglages. De plus, le calcul achevé laissait la lecture arrêtée à son premier état.

Une deuxième anomalie a été reproduite pendant le test : une image de lecture invalide interrompait le dessin, effaçait les robots et arrêtait la boucle d'animation. L'horloge utilise maintenant explicitement `performance.now()`, maintient un index borné et laisse la boucle disponible même après une erreur de dessin. Les tests couvrent un appel d'animation sans horodatage de rappel et un état de temps non fini.

- Navigation « Voir la simulation » et retour aux réglages.
- Positionnement automatique sur l'arène au lancement et au chargement du résultat.
- Message de calcul dans l'arène pendant l'attente.
- Lecture automatique à la fin du calcul, avec pause toujours disponible.
- Bouton « Résultat final » pour consulter immédiatement la forme obtenue, notamment lorsque l'attribution à 64 robots représente plusieurs minutes simulées.
- Physique, chronologie, préparation, métriques et poids inchangés. Le résultat final est un accès de lecture, pas une suppression de la trajectoire.

Guide actualisé : [GUIDE_EXPERIENCES.md](../docs/GUIDE_EXPERIENCES.md). Décision : [36-affichage-mobile](../docs/decisions/36-affichage-mobile.md).

Preuve : `evidence/020-affichage-mobile.json` et capture `outputs/demo-reference/020-affichage-64.png`. La fenêtre utilisateur reste positionnée sur le carré final à 64 robots.
