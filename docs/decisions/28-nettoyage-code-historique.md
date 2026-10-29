# 28 — Une seule base d’application active

Date : 2026-10-29. L’utilisateur autorise explicitement la suppression de l’ancienne base lorsqu’elle n’est plus utile.

Les imports du parcours à accélérations, de l’imitation, de la nouvelle interface et des tests associés ont été examinés. Les seuls liens vers l’ancien moteur provenaient des réexports des packages `src.common` et `src.simulator` ; ces packages sont remplacés par des exports des composants actuels. Les modules du cercle à accélérations restent nécessaires à la comparaison et à des fonctions partagées, et sont conservés.

Sont retirés les dossiers `src/controllers`, `src/models`, `src/training`, `src/evaluation`, `src/metrics`, `src/shapes` et `src/reports`, les anciens modules du simulateur à vitesse, l’ancienne configuration commune, l’ancienne mission, l’interface `demo/server.py`/`index.html`, ainsi que leurs configurations et tests exclusifs. L’inventaire comporte 96 fichiers source/configuration/test ; les caches des anciens dossiers sont retirés avec eux. Chaque chemin absolu est résolu et vérifié sous la racine du projet avant suppression.

Le Makefile et le README ne proposent plus les anciennes commandes. `make web` lance `demo.reference_server`, et `make test` exécute tous les tests conservés. Les dépendances FastAPI, Uvicorn, httpx, Matplotlib et Pillow, inutilisées par la base actuelle, sont retirées des dépendances obligatoires. Aucun paquet installé dans l’environnement n’est désinstallé.

Les données, modèles, sorties de recherche, preuves, décisions historiques, documents LaTeX et supports de soutenance sont préservés. Les modifications préexistantes de l’utilisateur ne sont pas réinitialisées. Aucun commit n’est créé ; les suppressions restent inspectables dans Git. Les références aux anciens chemins dans les décisions historiques décrivent l’état passé et ne désignent plus des composants actifs.

Après nettoyage : aucune référence d’import vers les anciens espaces de noms dans `src`, `demo` ou `tests`, aucune ancienne commande dans les entrées actives, et 578 tests réussis sur la suite complète. L’API catalogue de la démonstration répond toujours, avec les poids validés de l’interface disponibles. L’inventaire est conservé dans `tasks/evidence/013-legacy-cleanup-inventory.json`.
