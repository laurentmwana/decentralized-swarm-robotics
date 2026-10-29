# 34 — Adapter les dimensions avant de lancer une expérience

Une forme et un effectif ne déterminent pas seuls une configuration réalisable : un agrandissement sépare les places, mais peut couper les communications ou dépasser l'arène. Un simple remplacement du message d'erreur masquerait ce conflit. Nous calculons donc un intervalle de dimensions admissibles.

Les points sont générés à largeur unité. Leur distance minimale donne la borne basse d'échelle pour garder 0,40 m. La plus grande arête d'un arbre couvrant minimum donne la borne haute exacte pour la connectivité au rayon fixé. Les extrêmes des traits et des centres de référence donnent les bornes dues aux murs, avec réserve de 0,15 m en formation et 0,35 m pour le trajet de navigation. Toutes les contraintes sont vérifiées à nouveau par les validations existantes dans les tests.

Une largeur d'origine admissible reste inchangée. Sinon la largeur est bornée par cet intervalle, avec une petite réserve numérique. Lors d'un agrandissement, une séparation préférée de 0,50 m est recherchée dans cet intervalle et le temps de formation disponible est augmenté. Cette marge répond à un diagnostic conservé où la flèche à 24 robots restait au-dessus du seuil malgré une géométrie juste admissible. Les seuils scientifiques et physiques sont maintenus.

Le nouvel aperçu local fournit les options compatibles, les dimensions et les explications sans enqueuer de simulation. L'interface ignore les réponses périmées après un changement rapide de paramètres. Une option sélectionnée devenue incompatible reste visible avec explication et lancement désactivé, sans remplacement silencieux de la forme. Le serveur vérifie aussi le cas choisi pour les appels directs, les deux formes d'une transition, et la route avec obstacles.

Le manifeste garde les dimensions et leur éventuelle adaptation. Les dimensions des cas réservés du modèle structuré restent identiques. Les poids ne sont pas modifiés ; les autres dimensions ne sont pas présentées comme nouvellement validées pour le réseau. La réussite géométrique préliminaire ne remplace pas les métriques de l'épisode réel.
