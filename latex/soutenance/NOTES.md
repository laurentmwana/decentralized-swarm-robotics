# Notes de soutenance

18 diapositives principales et deux compléments. Durée indicative de 15 à 20 minutes, à ajuster après répétition.

## 1. Contrôle décentralisé d’un essaim de robots

Présenter le sujet. Il s’agit d’une étude en simulation 2D, avec apprentissage supervisé par imitation. Annoncer une présentation de 15 à 20 minutes, hors questions. Source : page de garde et introduction du mémoire.

## 2. Coordonner des décisions locales

Distinguer la commande locale du calcul de la mission partagée. La navigation utilise une carte connue et un planificateur classique. Ne pas présenter toute la chaîne comme apprise. Source : introduction et méthodologie.

## 3. Objectifs et démarche

Le résultat attendu est une évaluation mesurée, avec ses limites. Les tests du code et les essais du modèle répondent à deux questions différentes. Source : introduction, objectifs et méthodes.

## 4. Un simulateur dans un plan 2D

Aucun mouvement vertical ni modèle physique 3D. Les observations viennent des états simulés, sans caméra ni LiDAR réel. Figure du graphe calculée à partir du programme. Source : méthodologie, paramètres physiques ; latex/images/current_graph.png.

## 5. Préparer, attribuer, puis construire

Le cercle est programmé. Il n’est pas une décision spontanée du réseau. L’attribution classe les places par distance ; les identifiants départagent les conflits. Les numéros ne sont pas des entrées du réseau. Source : protocole, guide des expériences.

## 6. Le réseau propose une accélération

Un GAT est un Graph Attention Network. L’attention pondère les informations des voisins. Le modèle ne choisit pas les places à la place du protocole. Le planificateur de navigation est également classique. Source : méthodologie, architecture et commande finale.

## 7. Des données synthétiques pour l’imitation

Ces nombres concernent le GAT initial. Les fichiers d’entraînement sont 185, validation 20, test 9. Beaucoup d’observations d’une même trajectoire ne représentent pas autant d’essais indépendants. Le test initial réutilise un départ sur neuf formes. Source : run.json initial et méthodologie.

## 8. Entraîner ne garantit pas la mission

MSE signifie erreur quadratique moyenne. Pas de matrice de confusion pour cette sortie continue. Le GAT initial est retenu à l’époque 11, après arrêt à l’époque 23. Ne pas comparer directement l’échelle de la MSE à celle de la perte composite structurée. Source : journaux d’entraînement et Figure 3.2.

## 9. Des critères pour une mission complète

Formation : erreur de couverture bidirectionnelle entre robots et places, pas seulement une distance moyenne. Les contacts sont vérifiés sur les déplacements entre états enregistrés. Les durées sont simulées. Source : section 3.11 du mémoire.

## 10. Le filtre change le bilan du GAT initial

Résultats du rapport 008d-local-target-expanded-test.json. Le taux porte sur neuf cas, pas sur toutes les formes ni tous les départs. La préparation est classique. Le modèle protégé ne prouve pas la réussite du réseau seul.

## 11. Une figure précise peut avoir subi des contacts

Source : rapport 008d, cas geometries_ellipse graine 7300 et archives NPZ associées. Le graphe montre l’erreur et la séparation minimale. Le seuil de contact est 0,20 m entre centres. Le tracé logarithmique borne les très petites erreurs à 10⁻¹² m pour l’affichage seulement.

## 12. Le modèle structuré réussit les 18 cas réservés

Source : 014-vector-fresh-test.json et audit final. Six cas par effectif : trois formes fois deux graines. Les départs sont proches d’un rendez-vous contrôlé ; il ne s’agit pas de départs arbitraires. La préparation et l’attribution restent classiques.

## 13. À 64 robots, l’attribution coûte du temps

Source : résultats structurés 014, Figure de décomposition des durées. À 12 robots, stabilisation moyenne 29,43 s ; à 24, 85,23 s. Ne pas confondre temps simulé et temps d’attente réel devant l’ordinateur.

## 14. Navigation près de cinq à dix obstacles

Source : 019-near-targets-evaluation.json et audit. Les huit cas comprennent les nombres 5 à 10, un second départ à 10 et une régression à un obstacle. Le visuel illustre le cas à dix obstacles. Le GAT structuré brut n’est pas évalué en navigation.

## 15. Un catalogue plus large que les essais réservés

Source : catalogue du programme et Figure du mémoire. O et cercle partagent une géométrie ; V et la figure en V également. La génération d’une cible ne prouve pas la réussite du modèle sur cette cible.

## 16. Les limites de l’étude

Présenter ces limites clairement. Le graphe local n’annule pas les hypothèses de préparation et de mission partagée. Les résultats obtenus valent pour les conditions enregistrées. Source : discussion, validité interne, externe et de construit.

## 17. Contributions et suite du travail

Conclure sur ce que les essais établissent réellement. Le réseau structuré est prometteur dans son domaine contrôlé. Les perspectives ne sont pas des résultats déjà obtenus. Source : conclusion générale.

## 18. Démonstration et échanges

Avant la soutenance, lancer le serveur puis ouvrir http://127.0.0.1:8765/. Préparer un résultat enregistré en cas d’indisponibilité de l’application. La démonstration illustre les résultats ; elle ne remplace pas la campagne d’évaluation. Slides 19 et 20 sont des compléments pour les questions.

## 19. Complément — Réserver une place

Exemple oral : les robots 2 et 5 demandent la même place. Le 2 conserve la demande prioritaire ; le 5 choisit la suivante. Un robot qui traverse une place ne devient pas automatiquement son propriétaire. Le réseau apprend la commande, pas cette réservation. Source : protocole d’attribution et guide.

## 20. Complément — Régression et classification

Le rappel et la matrice de confusion conviennent à une tâche de classification avec catégories annotées. Ils ne sont pas nécessaires pour évaluer directement les accélérations continues de ce modèle. Succès/échec est un verdict de simulation, pas la sortie apprise. Source : concepts et critères d’évaluation.
