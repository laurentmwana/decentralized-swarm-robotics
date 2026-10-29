# 33 — Plusieurs obstacles et périmètre des modèles

L'effectif choisi ne détermine pas à lui seul le réseau : « GAT protégé » utilise les anciens poids appris avec 12 robots ; « GAT structuré brut » utilise un autre modèle appris avec 12, 24 et 64, validé seulement en formation contrôlée. Le texte de l'interface distingue ces cas et ne présente plus un contrôleur classique comme un réseau entraîné à 12.

La nouvelle scène accepte cinq à dix disques. Trois sont placés le long de la bande parcourue par la formation, les suivants autour ; tous sont transmis au planificateur sur carte déclarée, dessinés et exportés. Leur placement est fixe. Les perceptions du réseau restent locales. Les comptes sont validés côté serveur, y compris type et limites. Les anciens parcours gardent leur géométrie.

Un premier pilote à dix obstacles a révélé des contacts muraux pendant la préparation du GAT protégé. Le moteur bornait les positions, mais le filtre traitait robots et disques sans contrainte de mur. Quatre contraintes linéaires sont maintenant jointes à la même projection : distance intérieure h, condition h'' + 2h' + h ≥ 0, norme d'accélération inchangée. Une marge supplémentaire de 0,15 m précède le rayon physique. Les états réels et contacts balayés restent la référence ; aucune garantie universelle n'est déduite des contraintes si elles deviennent infaisables.

Cette protection seule peut laisser le réseau bloqué dans un passage pendant la construction de la forme. Pour le parcours multiple, nous retenons donc une préparation classique, déclarée dans l'interface et le manifeste, suivie d'un transport par le contrôleur choisi. Les deux variantes partagent la même préparation et le même trajet planifié ; le GAT protégé intervient pendant la navigation. Cette architecture hybride ne doit pas être présentée comme un apprentissage neuronal autonome des obstacles.

Le GAT structuré brut reste indisponible en navigation : ses preuves à 12/24/64 ne sont pas étendues artificiellement à ces nouvelles scènes. La campagne utilise les mêmes seuils de destination, formation, vitesse, maintien et zéro contact que les étapes précédentes. Les résultats du diagnostic échoué sont conservés sous outputs/navigation-multiple/.

La campagne finale réussit 8/8 par contrôleur, sans contact. Elle comprend les six comptes publics 5–10, un second départ à dix obstacles et une régression à un obstacle, tous avec le carré à 12 robots. Les audits indépendants vérifient seize trajectoires ainsi que l'identité des sept préparations classiques appariées. Ce résultat ne vaut pas pour des placements arbitraires, toutes les formes ni tous les effectifs.

La vérification navigateur a également révélé deux serveurs actifs : SO_REUSEADDR autorisait cette coexistence sous Windows, laissant répondre une ancienne API. Le serveur local utilise maintenant SO_EXCLUSIVEADDRUSE et refuse une seconde liaison au même port. L'ancienne instance a été arrêtée et la version courante relancée.
