# 36 — Affichage immédiat de la simulation

La page répondait et une simulation terminée à 64 robots était chargée, sans erreur JavaScript. Sur une fenêtre étroite, le panneau de réglages occupait cependant tout le premier écran : l'arène était plus bas. Le diagnostic visuel a confirmé ce décalage, plutôt qu'une panne du moteur ou du modèle.

La page offre désormais des liens vers l'arène et les réglages, et se positionne sur la simulation au calcul puis au chargement des données. La lecture démarre automatiquement après le calcul. Un bouton distinct affiche le dernier état pour éviter de devoir attendre plusieurs minutes de lecture à 64 robots ; le début, les étapes et les contacts restent consultables.

Ces changements concernent uniquement la présentation et la lecture. Aucun seuil, état physique, modèle ou temps de stabilisation n'est modifié. L'indication de calcul reste visible dans l'arène en attendant les données.

Le test de lecture automatique a également reproduit une exception de dessin à 64 robots, avec un état physique pourtant complet. Le rendu utilisait un index de lecture invalide et la boucle cessait de se programmer après l'exception. L'horloge est maintenant prise explicitement avec performance.now(), les temps et indices sont bornés, et le prochain rafraîchissement est programmé avant le dessin. Une erreur de dessin met la lecture en pause avec un message, sans désactiver définitivement la boucle. Deux tests JavaScript de régression vérifient l'absence d'horodatage de rappel et les temps non finis.

Vérification dans la fenêtre étroite : le calcul affiche son arène, puis l'animation avance au-delà de 180 s simulées ; le bouton de résultat final affiche le carré à 481,6 s. Les 17 tests ciblés de lecture et démonstration passent.
