# 022 — Rétablir le plan du professeur et livrer entre 65 et 70 pages

Demande : conserver les dossiers du mémoire et le plan du professeur, avec un PDF de 65 à 70 pages. Cette correction remplace la version trop condensée de la tâche 021 pour la rédaction courante.

## Structure rétablie

- Fichiers existants conservés dans `latex/chapters/` et `latex/frontmatter/`.
- Introduction générale : contexte, problématique, hypothèses, objectifs, méthodes, contribution et plan sommaire.
- Quatre chapitres : Concepts de base ; État de l'art ; Méthodologie et solution proposée ; Implémentation, expérimentations et résultats.
- État de l'art : Introduction, Méthodes traditionnelles, Méthodes avancées, Quelques travaux connexes, Positionnement, Conclusion partielle.
- Trois travaux connexes, avec quatre rubriques chacun : contexte et questions, méthode, résultats, critique.
- Discussion puis Conclusion générale et perspectives, distinctes et non numérotées.
- Chaque chapitre numéroté commence par Introduction et finit par une Conclusion partielle en un paragraphe annonçant la suite.

## Développements et présentation

La rédaction développe les données et leur séparation, la réservation, les entrées du réseau, les profils, l'organisation du programme et la lecture des résultats. Elle détaille les dix-huit cas réservés et ajoute trois figures tirées du catalogue ou des archives. Les échecs et les limites de la version actuelle restent conservés. Aucun résultat ancien à commande de vitesse n'est réintroduit.

Les légendes des tableaux sont replacées au-dessus. Figures, tableaux et équations sont annoncés et référencés avant leur insertion. Les abréviations sont classées et développées dans leur langue d'origine. Les liens du sommaire de l'introduction et de la discussion sont corrigés.

## Vérification et livraison

**70 pages physiques**, page de garde, pages liminaires, table des matières et bibliographie comprises : 13 pages avant l'introduction et 57 pages numérotées en chiffres arabes. Police 12 points, interligne 1,5 et marges de 2,5 cm conservés. Pas de page blanche ajoutée pour atteindre la longueur.

Huit figures scientifiques, onze tableaux, onze références citées et seize groupes d'équations. Compilation pdfLaTeX/Biber réussie, sans débordement ni renvoi indéfini. Les 70 pages ont été rendues et examinées sur planches ; les figures ajoutées et la dernière page corrigée de méthode ont également été inspectées.

Preuves : `evidence/022-professor-plan.json`, `evidence/022-manuscript-audit.json`, `evidence/022-pdf-review.json`. Le script `latex/tools/check_professor_plan.py` vérifie le maintien du plan. La version de 45 pages est sauvegardée localement dans `outputs/thesis-redaction/version-45-pages/`.

PDF : `output/pdf/memoire-swarm-gnn-redaction.pdf`. Sources : `latex/main.tex` et les dossiers existants. Répartition détaillée : `latex/PAGINATION.md`.
