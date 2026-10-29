# 26 — Démonstration actuelle et comparaisons finales

Date : 2026-10-29. Tâche 010, autorisée par « vas-y ».

Mise à jour après la tâche 013 : l’ancienne démonstration et ses dépendances inutilisées ont été supprimées sur autorisation de l’utilisateur. L’interface décrite ici est désormais la seule interface active ; voir la décision 28 pour le périmètre du nettoyage.

## Intégration

L’ancienne démonstration reste disponible pour les expériences historiques. `demo.reference_server` branche une nouvelle interface sur les épisodes à accélérations, l’allocation locale, le GAT à cible locale, les transitions continues et la navigation de 008–009. Le serveur utilise la bibliothèque standard Python ; NumPy, PyYAML et Torch restent les dépendances de simulation. Il écoute uniquement sur 127.0.0.1, port 8765 par défaut. `demo/start-reference.ps1` retrouve le runtime et les dépendances Windows de cette installation, ou utilise Python normalement.

Le catalogue expose les 56 consignes vectorielles. Les effectifs proposés sont 8, 12, 16 et 24 ; seul 12 correspond aux campagnes et à l’apprentissage multi-formes de référence. Les autres effectifs sont signalés comme exploratoires, et la faisabilité géométrique est vérifiée avant lancement. Le nombre de pas de préparation vaut max(400, 90 + N² + 200), laissant le rendez-vous, les tours d’allocation et au moins 20 s de contrôle ; à 12 robots, cela donne 43,4 s, contre 40 s dans les campagnes 008–009. Les essais interactifs ne sont donc pas présentés comme des répétitions identiques de ces campagnes.

Les scénarios sont une formation, une transition continue vers une autre forme, une navigation rectiligne, un contournement fourni et les deux familles difficiles de 009. Les routes et disques sont prédéfinis ; ils ne constituent pas une planification globale apprise. Changer de forme en navigation reste exploratoire. Les positions initiales sont strictement partagées entre les modes d’une comparaison, les poids sont chargés une fois, leur empreinte est vérifiée et aucun apprentissage n’est déclenché.

## Calcul et lecture

Un calcul est exécuté en arrière-plan, dans une file à un seul travailleur. Les états complets sont calculés avant la lecture. Pause, vitesse ×1/×2/×5/×10, curseur et réinitialisation agissent sur la lecture ; ils ne prétendent pas interrompre l’intégration numérique. L’étape courante distingue regroupement, freinage, répartition, construction et transport. Les contrôles de lancement sont désactivés pendant l’attente et les anciens résultats sont retirés afin de ne pas les confondre avec le nouveau calcul.

Le rendu Canvas montre les disques physiques, places locales de l’étape courante, communications réellement enregistrées, traces récentes et numéros facultatifs. L’axe temporel et les coordonnées proviennent directement des résultats, sans interpolation qui pourrait masquer un contact. Les mesures affichées concernent toujours l’expérience complète ; les transitions affichent le temps absolu de stabilisation de la dernière étape lorsque toute la séquence réussit.

La réussite d’un essai reste un résultat individuel. Le volet distinct « Campagnes conservées » affiche les taux et dénominateurs des évaluations antérieures. Les lignes de navigation nominales, difficiles et globales se recouvrent et ne doivent pas être additionnées ; les temps moyens sont calculés sur les seules réussites. Les stages de transition partagent deux séquences et les graines ne sont pas toutes indépendantes.

## Exports et traçabilité

Chaque calcul terminé conserve un dossier UUID dans `outputs/demo-reference/`. Un manifeste enregistre la demande, la physique, le départ exact, l’empreinte du modèle, le statut exploratoire d’effectif et les résultats. Par contrôleur : archive NPZ sans pickle, JSON de lecture avec les images physiques exactes, CSV temps/robot/position/vitesse. Les archives contiennent aussi les consignes locales et les traces de protocole disponibles. Un ZIP rassemble ces fichiers et le manifeste.

Les requêtes du navigateur ne peuvent fournir ni chemin de fichier, ni poids alternatifs, ni entraînement. Les endpoints de résultat n’acceptent que des identifiants générés et des noms fixes ; les chemins arbitraires sont refusés. La session est limitée à 20 demandes et deux calculs en attente ; les résultats physiques restent sur disque au redémarrage, mais l’historique interactif en mémoire n’est pas restauré automatiquement. Ce serveur local de recherche ne constitue pas un service de production multi-utilisateur.

`python -m demo.comparison_report --output <nouveau-dossier>` reconstruit 11 lignes de comparaison à partir des preuves 008d, 008e et 009. Le rapport CSV/JSON conserve l’empreinte SHA-256 de chaque source. Il inclut le GAT brut à 0/9 et 54 contacts sur les formes réservées, le GAT protégé à 9/9 sans contact sur ces mêmes formes, ainsi que les deux contrôleurs à 6/10 en navigation de validation, dont 0/4 sur les cas difficiles. Ces résultats ne sont pas remplacés par les essais interactifs.

## Vérification

423 tests ciblés réussis : les 412 vérifications précédentes et 11 tests supplémentaires de catalogue, paramètres publics, modèle absent, calcul réel, fidélité JSON/NPZ/CSV, ZIP, échange HTTP et conservation des échecs dans le rapport final. Aucun entraînement n’a lieu pendant les tests.

Vérification interactive dans le navigateur : carré, départ 7700, comparaison professeur/GAT protégé, deux réussites sans contact ; erreur finale GAT ≈ 0,011 m, stabilisation 31,2 s contre 29,1 s pour le professeur. Le curseur final, l’affichage des communications et numéros, la réinitialisation, la lecture et la pause sont exercés. Transition carré→B avec le GAT protégé : réussite sans contact, erreur finale ≈ 0,013 m, stabilisation absolue 76,4 s. Le rapport final régénéré est conservé sous `tasks/evidence/010-final-comparison.json`.

La destination obstruée est vérifiée dans l’interface : les deux contrôleurs échouent sans contact, avec des erreurs finales de forme d’environ 0,520 m et 0,566 m ; leurs temps réussis restent absents. L’erreur transitoire maximale du GAT protégé est ≈ 0,785 m. Le volet de comparaison affiche notamment 60 % sur l’ensemble de navigation de validation, et 0 % sur ses cas difficiles. L’audit des quatre dossiers complets présents pendant la vérification, soit six trajectoires, confirme l’identité JSON/NPZ des positions, vitesses et temps, les lignes CSV, les départs appariés, les mesures du manifeste et le contenu exact des ZIP. Preuve : `tasks/evidence/010-demo-export-audit.json`. Ces essais interactifs ne sont pas ajoutés aux taux des campagnes conservées.

Les résultats de sécurité concernent le prototype hybride. Le modèle seul n’a pas atteint l’objectif de contrôle sûr sans protection classique, et les obstacles difficiles ne sont pas résolus par cette interface. L’achèvement de l’interface ne transforme pas ces limites en succès scientifique.
