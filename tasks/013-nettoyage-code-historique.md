# 013 — Supprimer le code historique inutilisé

Statut : terminé à la demande explicite de l’utilisateur. 96 fichiers source/configuration/test obsolètes supprimés, après vérification des dépendances et des chemins. Suite complète conservée : 578 tests réussis.

Suppression : anciens contrôleurs, modèles, entraînements, évaluations, métriques et formes du simulateur à vitesse, anciens rapports exécutables, anciens modules physiques, ancienne interface FastAPI et configurations/tests correspondants. Les imports publics des packages commun et simulateur exposent désormais le parcours à accélérations. Les anciennes commandes du Makefile et du README sont remplacées par les commandes actuelles ; les dépendances de l’ancienne interface sont retirées.

Conservation : simulateur à accélérations, professeurs et métriques de référence, cercle et formes générales, imitation et comparaisons, interface actuelle, données et modèles, preuves, décisions scientifiques, mémoire et supports de soutenance. Le cercle à accélérations reste utile et ne fait pas partie du code historique supprimé.

Vérification : aucun import ni commande active du code conservé ne pointe vers les anciens modules ; tous les tests restants passent ; le serveur actuel répond et conserve le modèle validé. Aucun commit, réinitialisation Git ou suppression de résultat scientifique n’est effectué.

Inventaire : `tasks/evidence/013-legacy-cleanup-inventory.json`. Les décisions historiques sont conservées comme archives, pas comme commandes actuelles.
