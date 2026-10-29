# 021 — Reprendre le mémoire sur l'implémentation actuelle

Demande : continuer le travail déjà commencé dans `latex/` comme rédacteur. Précisions : français naturel, mots simples, aucune incohérence et aucune illustration ayant l'apparence d'une image générée par IA.

## Travail réalisé

- Sauvegarde locale des sources préexistantes avant révision.
- Reprise de l'introduction, des notions, de l'état de l'art, de la méthode, des expériences, de la discussion, des résumés français/anglais et des sigles.
- Conservation de l'identité, des pages personnelles et de la présentation déjà choisie.
- Retrait du récit obsolète à commande de vitesse et des 9 600 épisodes de la version actuelle du texte.
- Description explicite du professeur, des données synthétiques, des partitions, de la MSE, de l'attribution et des deux modèles.
- Distinction entre formation structurée brute et navigation hybride ; échecs et limites conservés.
- Cinq figures scientifiques en noir et blanc tirées des fichiers du programme, deux tableaux générés et manifeste de provenance.
- Script de contrôle documentaire et guide `latex/REDACTION.md`.

## Vérification

Le contrôle documentaire passe : dix références bibliographiques citées, seize labels, six références d'images dont le logo existant, sept empreintes de sources et deux empreintes de modèles vérifiées. Les chiffres principaux sont contrôlés contre les rapports, notamment les 18 cas de formation et les huit cas de navigation par contrôleur.

Statut : terminé pour cette première reprise. PDF compilé avec pdfLaTeX et Biber : **45 pages**, cinq figures scientifiques, sept tableaux et dix références citées. Aucun débordement de boîte, renvoi indéfini ou avertissement d’accent en mode mathématique dans le journal final. Les 45 pages ont été rendues et examinées sur planches de contrôle ; les pages de résultats et les graphiques ont été vérifiés. Le premier compilateur intégré a expiré pendant son téléchargement initial ; le compilateur local a permis de terminer la vérification.

Livraison : `output/pdf/memoire-swarm-gnn-redaction.pdf`, également disponible dans `latex/main.pdf`. Preuves : `evidence/021-manuscript-audit.json` et `evidence/021-pdf-review.json`. Les consignes supplémentaires du directeur pourront faire l’objet d’une prochaine relecture.
