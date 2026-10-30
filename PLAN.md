Voici le **plan d’implémentation de ton mémoire**, en reprenant nos choix : contrôle décentralisé, réseau GAT, apprentissage supervisé par imitation et quatre métriques essentielles.

## 1. Définir le système et les missions

L’application simulera un essaim de robots dans un **environnement 2D borné**.

Chaque robot disposera de ses observations et des informations de ses voisins situés dans sa portée de communication. Il calculera lui-même son déplacement.

Les missions seront développées progressivement :

1. Former un **cercle** à partir de positions dispersées.
2. Former une **ligne**, puis un **V**.
3. Passer d’une formation à une autre.
4. Déplacer la formation vers une destination, puis introduire des obstacles.

**Résultat attendu :** des missions précises, avec leurs conditions de réussite. La navigation avec obstacles constituera la dernière étape, après validation des formations.

## 2. Construire le simulateur

Implémenter :

- les robots : position, vitesse, rayon physique et accélération ;
- le terrain et ses limites ;
- les déplacements à chaque pas de temps ;
- la détection des collisions ;
- les observations locales ;
- l’enregistrement des trajectoires.

**Résultat attendu :** un environnement fonctionnel dans lequel les robots peuvent recevoir des commandes et se déplacer.

## 3. Définir les formations et la répartition des robots

Pour chaque formation, préciser sa taille, son orientation et les règles d’espacement.

Il faudra aussi résoudre la répartition des robots :

- pour le cercle, utiliser un objectif de rayon et d’espacement ;
- pour la ligne et le V, définir un mécanisme local de répartition ou d’affectation, puis vérifier qu’il évite les conflits de places.

**Résultat attendu :** une définition exploitable de chaque formation, compatible avec les informations accessibles aux robots.

**Ce point doit être résolu avant l’entraînement : le GAT ne crée pas, à lui seul, un mécanisme d’affectation.**

## 4. Programmer le contrôleur de référence

Créer un contrôleur classique qui combine les comportements nécessaires :

- rejoindre la formation ;
- maintenir l’espacement ;
- éviter les collisions ;
- ralentir pour stabiliser la formation ;
- suivre une destination lorsque cette mission sera ajoutée.

Il devra utiliser les mêmes informations locales que le futur contrôleur GAT.

**Résultat attendu :** un professeur fonctionnel, qui servira également de référence pour les comparaisons.

## 5. Générer les données d’apprentissage

Exécuter le contrôleur de référence sur différentes configurations et enregistrer :

| Entrées | Réponse attendue |
|---|---|
| Observations du robot, consigne de formation et informations des voisins | Commande d’accélération calculée par le professeur |

Séparer les données en ensembles d’**entraînement, de validation et de test**, en séparant les simulations complètes pour éviter de retrouver des situations presque identiques dans plusieurs ensembles.

**Résultat attendu :** un jeu de données permettant d’entraîner et d’évaluer le modèle.

## 6. Développer et entraîner le GAT

Architecture initiale proposée :

| Bloc | Composition |
|---|---|
| Encodage des observations | Deux couches denses |
| Échange entre voisins | Une couche GAT |
| Calcul du mouvement | Une couche dense et une couche de sortie |
| Sortie | Deux valeurs : accélérations horizontale et verticale |

Le modèle sera partagé par tous les robots. **Une seule couche GAT correspondra à un échange direct entre voisins.**

L’entraînement supervisé cherchera à réduire la **MSE entre les commandes prédites et celles du professeur**.

**Résultat attendu :** un modèle sauvegardé, capable de produire une commande à partir d’observations locales.

## 7. Intégrer le GAT dans la simulation

Remplacer les commandes du professeur par celles du modèle entraîné.

À chaque pas :

1. Construire le graphe des communications.
2. Préparer les observations locales.
3. Calculer les commandes avec le GAT.
4. Appliquer les limites physiques.
5. Mettre à jour les positions.

**Résultat attendu :** un essaim piloté par le GAT, sans intervention du professeur.

## 8. Évaluer les performances

Comparer le **GAT et le contrôleur classique** sur les mêmes scénarios avec les quatre métriques retenues :

| Métrique | Unité |
|---|---|
| Erreur de formation | Mètres |
| Temps de stabilisation | Secondes |
| Nombre de collisions | Nombre |
| Taux de réussite | Pourcentage |

Fixer à l’avance le seuil d’erreur acceptable, la durée de maintien et le délai maximal. Pour les essais qui échouent, signaler l’échec au lieu de leur attribuer artificiellement un temps de stabilisation.

Répéter les expériences avec différentes positions initiales, tailles d’essaim et conditions de communication.

**Résultat attendu :** des tableaux et graphiques montrant les performances, leur variabilité et les limites du modèle.

## 9. Construire l’interface de démonstration

L’interface permettra de :

- choisir le nombre de robots et la formation ;
- sélectionner le contrôleur classique ou le GAT ;
- lancer, mettre en pause et réinitialiser la simulation ;
- afficher les communications et les trajectoires ;
- consulter les quatre métriques ;
- exporter les résultats.

**Résultat attendu :** une application utilisable pour les expériences et la soutenance.

## 10. Organiser la réalisation

| Phase | Travail | Condition pour passer à la suite |
|---|---|---|
| **1** | Simulateur et formation circulaire avec le contrôleur classique | Comportement vérifié et métriques disponibles |
| **2** | Données, entraînement et contrôle par GAT sur le cercle | Évaluation sur de nouvelles simulations |
| **3** | Ligne, V et changements de formation | Répartition des robots et transitions validées |
| **4** | Déplacement collectif et obstacles | Navigation évaluée sans masquer les échecs |
| **5** | Comparaisons finales et interface | Résultats reproductibles et démonstration prête |

**La première étape concrète sera donc de construire le simulateur et de réussir la formation circulaire avec le contrôleur classique.** Cela fournira la base nécessaire aux données, au GAT et aux expériences du mémoire.