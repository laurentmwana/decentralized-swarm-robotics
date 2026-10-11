# Synthèse provisoire de l'état de l'art

## Ce que les approches classiques apportent

Les règles locales de flocking et les contrôleurs de formation géométrique constituent des baselines interprétables. Elles permettent de vérifier le simulateur, de produire un comportement de regroupement et de mesurer l'apport réel du modèle appris. Elles exigent toutefois une conception manuelle spécifique à la tâche et à la géométrie visée.

## Ce que les GNN apportent

Les GNN correspondent naturellement à un essaim où les robots sont des nœuds et les liens de communication sont des arêtes. Un même modèle peut être partagé par tous les robots ; une agrégation symétrique des voisins rend l'architecture compatible avec un ordre et un nombre de voisins variables. Les travaux de Tolstaya, Gama et Jiang montrent que ce cadre peut être employé pour des contrôleurs décentralisés et évalué avec des tailles d'équipe différentes.

## Point de vigilance scientifique

Une politique peut être entraînée de manière centralisée ou à partir d'un expert, mais elle doit rester décentralisée au moment de l'exécution. Le mémoire devra donc distinguer explicitement les informations autorisées pendant l'entraînement de celles accessibles par un robot dans le simulateur.

## Positionnement de travail

Le projet étudie une politique GNN unique et partagée, qui ne reçoit à l'exécution que des observations locales, des positions relatives de voisins et un identifiant de forme commun. Il sera évalué sur la qualité de formation de formes, les collisions, la perte de robots et le changement de taille de l'essaim, contre des règles locales écrites à la main et une variante sans échange de messages.

## Limite de cette synthèse

Cette synthèse reste préparatoire : elle doit être consolidée par 15 à 25 lectures intégrales, des fiches complètes et une vérification précise des protocoles expérimentaux avant d'être intégrée au mémoire.
