# Guide des expériences, contrôleurs et commandes

Ce guide décrit la démonstration locale actuelle. Les unités sont le mètre, la seconde, le mètre par seconde et le mètre par seconde carrée.

## 1. Démarrer et arrêter le serveur

Dans PowerShell, depuis le dossier du projet :

```powershell
cd D:\ProjectLabs\python\swarm-gnn
.\demo\start-reference.ps1
```

Ouvrir ensuite <http://127.0.0.1:8765/>. Garder le terminal ouvert. Pour arrêter le serveur, revenir dans ce terminal et appuyer sur **Ctrl+C**.

Le lanceur utilise l'environnement Python fourni avec Codex lorsqu'il est disponible. Avec un environnement Python personnel correctement installé :

```powershell
python -m pip install -r requirements.txt
python -m demo.reference_server
```

Un seul serveur peut utiliser le port 8765. Si le port est déjà occupé par cette démonstration, arrêter son instance avant de la relancer. Relancer le serveur après une modification du code Python ; recharger la page après une modification de l'interface.

## 2. Nombre de robots et modèle sélectionné

Le nombre de robots et le contrôleur sont deux réglages différents. Choisir 24 robots n'entraîne pas un nouveau réseau. Choisir un contrôleur classique signifie que les mouvements sont calculés par des règles, même si des modèles entraînés existent dans le projet.

| Choix | Ce qui calcule les mouvements | Effectifs et domaine |
|---|---|---|
| Contrôleur classique | Une commande mathématique et les protections classiques | 8, 12, 16 et 24 dans les scènes permises ; 64 en formation avec profil contrôlé |
| GAT protégé par le filtre classique | Le GAT de référence propose les accélérations ; un filtre peut les modifier | Entraîné avec 12 robots ; 8, 16 et 24 sont des utilisations exploratoires ; indisponible à 64 |
| GAT structuré brut · 12/24/64 | Le modèle structuré calcule les mouvements sans filtre pendant ses phases apprises | Entraîné avec 12, 24 et 64 ; formation seulement, départ contrôlé |
| Classique / GAT protégé | Deux simulations appariées avec les deux premières commandes | Même départ, même forme, même scène et mêmes critères |
| Classique / GAT structuré | Deux simulations appariées, classique et modèle structuré | Formation avec départ contrôlé, à 12, 24 ou 64 |

Il y a donc **trois contrôleurs et deux modes de comparaison**, pas cinq réseaux.

La réserve finale du GAT structuré contient le rectangle, la lettre C et le nombre 3, à deux départs pour chacun des trois effectifs. Les autres formes restent expérimentales même si leur géométrie est compatible.

Le modèle structuré ne couvre pas 8 et 16 robots. À ces effectifs, l'interface annonce son remplacement par le modèle de référence lorsqu'on quitte un profil structuré. À 64, si l'on utilisait le modèle de référence, l'interface privilégie le modèle structuré disponible et annonce ce changement. Un choix explicite du contrôleur classique reste classique.

L'interface indique toujours le contrôleur effectivement sélectionné et l'effectif courant. « Aucun réseau de neurones utilisé » décrit un choix classique ; ce n'est pas une conclusion sur tous les contrôleurs disponibles.

## 3. Pourquoi un cercle apparaît avant une autre forme

Le cercle initial est un **rendez-vous provisoire**, distinct de la forme finale demandée. Il organise une configuration de communication entre les robots.

Le déroulement est le suivant :

1. **Préparation : cercle provisoire.** Les robots rejoignent leurs positions de rendez-vous afin de faciliter les échanges entre voisins.
2. **Freinage.** Ils réduisent leur vitesse avant l'affectation.
3. **Répartition des places.** Ils échangent leurs revendications de places. Chaque robot classe les places selon sa distance ; un identifiant prioritaire tranche les conflits. Cette attribution est classique et distribuée.
4. **Construction de la forme choisie.** Les robots rejoignent leurs places finales.
5. **Transport**, si l'expérience demande un déplacement collectif.

Le cercle n'est pas un choix spontané du réseau. Il appartient au protocole programmé. Pour le modèle structuré, les positions initiales sont déjà proches de ce rendez-vous ; le numéro reproductible introduit seulement de petites variations autour de lui. Les autres parcours utilisent des départs plus dispersés.

Supprimer réellement cette phase modifierait l'algorithme et les conditions d'apprentissage. Les résultats acquis ne démontrent pas que cette suppression fonctionnerait. L'interface conserve la trajectoire complète afin que la démonstration et les métriques représentent les mouvements réellement calculés.

La répartition emploie actuellement jusqu'à N² tours locaux. À un tour de 0,1 s :

| Robots | Temps simulé consacré aux tours d'attribution |
|---:|---:|
| 8 | 6,4 s |
| 12 | 14,4 s |
| 16 | 25,6 s |
| 24 | 57,6 s |
| 64 | 409,6 s |

Ces durées sont des secondes de simulation, pas nécessairement la durée d'attente devant l'ordinateur. Le protocole actuel à 64 robots est coûteux ; ce tableau ne décrit pas un besoin universel en robotique.

## 4. Les sept expériences

### 4.1 Former l'essaim

**But :** construire une lettre A–Z, un nombre 1–10 ou une figure géométrique du catalogue.

Choisir la formation, l'effectif, le contrôleur et le numéro de départ, puis lancer. Les robots exécutent la préparation et construisent la forme. Avec le modèle structuré, l'arène et les dimensions suivent le profil 12/24/64.

**À regarder :** erreur finale, temps de stabilisation, contacts et réussite. Une forme visuellement proche peut échouer si la vitesse reste trop élevée ou s'il y a eu un contact auparavant.

### 4.2 Changer de formation

**But :** passer de la première forme à une seconde en conservant les positions et les vitesses entre les étapes.

Choisir la première formation puis **Formation suivante**. Chaque étape reçoit une affectation adaptée. L'essaim n'est pas téléporté dans la nouvelle figure.

**À regarder :** réussite de chaque étape et contacts sur la séquence entière. La carte d'erreur finale se rapporte à la dernière forme ; le tableau et l'export conservent les résultats de la séquence.

Le modèle structuré et l'effectif 64 ne sont pas proposés pour cette expérience dans la version actuelle.

### 4.3 Déplacer en ligne droite

**But :** transporter la formation d'un centre situé à −1 m vers un centre situé à +1 m, sans obstacle.

Une référence mobile commune déplace les places cibles. Le contrôleur de chaque robot reçoit sa cible locale et ses observations autorisées. La planification et l'anticipation de la référence sont classiques.

**À regarder :** maintien de la forme et arrivée réelle à destination. Une formation immobile ne réussit pas simplement parce qu'elle garde sa forme.

### 4.4 Contourner un obstacle

**But :** suivre un parcours de contournement comportant des points de passage.

Le parcours demandé passe notamment par une déviation verticale avant de rejoindre la destination. Le planificateur vérifie le passage de toutes les places et adapte le détour si nécessaire. Le contrôleur ne doit pas seulement éviter un contact : il doit aussi conserver la formation.

### 4.5 Obstacle sur la route · détour calculé

**But :** demander un trajet direct alors qu'un obstacle gêne le passage de certaines places.

Le planificateur classique cherche une translation commune qui évite l'obstacle pour toute la formation. Le centre de destination reste identique. Une interpolation progressive fixe la référence, avec une vitesse maximale de 0,20 m/s.

**À regarder :** le détour, l'erreur maximale pendant le déplacement et la distance finale à destination. Le réseau ne découvre pas seul une carte : le planificateur utilise la carte connue de la scène.

### 4.6 Naviguer avec plusieurs obstacles

**But :** transporter une formation avec **5, 6, 7, 8, 9 ou 10 obstacles**.

Choisir **Nombre d'obstacles**. Les obstacles sont désormais placés près des places finales du carré à 12 robots, sur plusieurs côtés, plutôt qu'aux bords éloignés de l'arène. Dans cette scène, le centre de chaque obstacle est à au plus 1 m d'une place finale du carré ; les rayons et marges de passage sont pris en compte. « Près de la cible » désigne ici une place finale de robot, pas uniquement le centre de l'essaim.

Le placement est fixe et reproductible. Ajouter des obstacles ne génère pas aléatoirement un nouvel environnement. Les autres formes ou effectifs passent d'abord par le contrôle de compatibilité ; leur réussite n'est pas garantie par les essais du carré.

**Architecture de ce parcours :** préparation classique de la formation, puis transport par le contrôleur choisi, avec planification classique et protection des déplacements. Le GAT protégé intervient pendant le transport ; il n'est pas réentraîné lorsque l'on change le nombre d'obstacles.

### 4.7 Vérifier une destination obstruée

**But :** montrer qu'une destination peut être géométriquement impossible.

Dans le cas prévu pour le carré, un obstacle occupe une place finale. L'aperçu refuse alors la configuration avant la simulation et explique le problème. Ce refus n'est ni un plantage ni une preuve que le réseau a mal appris : on demande une place occupée par un obstacle.

Choisir une autre formation ou un parcours compatible. L'application ne déplace pas silencieusement la destination et ne transforme pas ce refus en réussite.

## 5. Dimensions et choix incompatibles

Une largeur fixe ne convient pas à toutes les combinaisons. Trop de robots sur une petite forme peuvent donner des places trop proches ; trop peu de robots sur une grande forme peuvent couper les communications.

L'aperçu contrôle l'espacement, les communications et les murs, puis :

- garde les dimensions d'origine lorsqu'elles conviennent ;
- agrandit ou réduit la forme quand une taille compatible existe ;
- affiche la largeur effectivement choisie ;
- désactive les combinaisons irréalisables et explique comment changer le réglage.

Le minimum de sécurité reste 0,40 m entre places. Une marge préférée supérieure est recherchée pour les formes agrandies quand elle tient dans l'arène. Les seuils de réussite restent identiques. Une compatibilité géométrique ne prouve pas encore que le contrôleur réussira la simulation.

## 6. Numéro reproductible et commandes de lecture

| Réglage ou bouton | Utilité |
|---|---|
| Disposition initiale · numéro reproductible | Recréer les mêmes positions avec les mêmes réglages. Ce numéro n'est pas une entrée du réseau. |
| Lancer l'expérience | Calculer une nouvelle simulation, afficher sa zone et démarrer automatiquement la lecture. Aucun entraînement n'est lancé. |
| Lire / Pause | Lire ou suspendre l'animation d'une simulation déjà calculée. |
| Réinitialiser | Revenir au début de sa lecture. Ce bouton ne réentraîne pas le modèle. |
| Résultat final | Afficher directement le dernier état, sans attendre toute la préparation ou les tours d'attribution. La trajectoire complète et ses métriques sont conservées. |
| Voir la simulation | Descendre directement vers l'arène, particulièrement utile lorsque les réglages prennent toute la hauteur d'un petit écran. |
| Modifier les réglages | Revenir au formulaire de l'expérience. |
| ×1, ×2, ×5, ×10 | Changer la vitesse de lecture ; les équations physiques restent identiques. |
| Barre du temps | Examiner un instant précis de la trajectoire. |
| Résultat affiché | Basculer entre les contrôleurs dans une comparaison appariée. |
| Communications | Afficher les liens entre voisins dans la portée de communication. |
| Trajectoires | Afficher les déplacements enregistrés. |
| Places cibles | Afficher les positions cibles de l'étape en cours. Pendant la préparation, ce sont les places du rendez-vous. |
| Numéros des robots | Afficher leurs identifiants, utilisés notamment pour trancher les conflits de places. |
| Exporter les résultats | Télécharger le manifeste, les états physiques, les positions/vitesses et les métriques. |

La simulation est calculée entièrement avant sa lecture. La page se positionne sur l'arène au lancement et lorsque le résultat est chargé. Les valeurs finales restent visibles même quand la barre du temps montre le début de l'expérience.

## 7. Lire les métriques

| Métrique | Sens et condition |
|---|---|
| Erreur finale de formation | Écart de couverture entre robots et places ; seuil final 0,10 m. |
| Stabilisation | Instant où les conditions sont maintenues pendant 1 s avec des vitesses au plus égales à 0,05 m/s. |
| Contacts physiques | Contacts robot–robot ; en navigation, également obstacles et murs. Les passages entre deux états sont vérifiés. |
| Réussite de l'essai | Verdict pour une simulation complète, avec les critères de l'expérience. |
| Distance à destination | En navigation, écart du centre effectif à la destination ; seuil 0,10 m. |
| Déformation maximale en route | Plus forte erreur pendant le transport ; limite 0,20 m. |

Zéro contact ne suffit pas à réussir. L'essaim peut manquer la destination, dépasser le délai ou rester trop déformé. « Non atteinte » signifie que la stabilisation requise n'a pas été obtenue ; on ne lui attribue pas artificiellement un temps réussi.

Un **taux de réussite** exige plusieurs essais et un dénominateur. Les tableaux de campagnes indiquent les essais évalués. Les anciennes campagnes restent des archives : leurs résultats ne sont pas remplacés rétroactivement par ceux d'un nouveau modèle ou d'une scène modifiée.

Pendant l'entraînement supervisé, la **MSE** mesure l'écart quadratique moyen entre l'accélération prédite et celle du professeur. Le réseau effectue une régression, pas une classification : une matrice de confusion et le rappel ne sont pas les métriques adaptées à cette sortie continue. Une faible MSE ne remplace pas la vérification des mouvements réels en simulation.

## 8. Commandes de vérification et de campagnes

Ces commandes supposent un Python disposant des dépendances du projet. Chaque dossier de sortie doit être nouveau pour conserver les résultats précédents.

### Tests du logiciel

```powershell
python -m pytest tests -q -p no:cacheprovider
```

Les tests contrôlent le logiciel. Ils ne remplacent pas l'entraînement ni une campagne de simulations sur de nouveaux cas.

### Campagne de navigation planifiée

```powershell
python -m src.reference.evaluate_planned_navigation --output outputs/mes-essais-navigation
```

Compare classique et GAT protégé, puis audite les états et les métriques. Les noms de formes, effectifs et départs du protocole sont enregistrés dans le résultat.

### Campagne de cinq à dix obstacles proches des cibles

```powershell
python -m src.reference.evaluate_multiple_obstacles --output outputs/mes-essais-obstacles --seed 8901
```

Évalue chaque nombre 5–10, un second départ à dix obstacles et une régression à un obstacle. Compare les deux contrôleurs, vérifie les trajectoires, les métriques et la préparation commune.

### Exporter les comparaisons historiques sans simuler

```powershell
python -m demo.comparison_report --output outputs/mes-comparaisons-historiques
```

Cette commande relit les preuves archivées ; elle ne refait pas les expériences.

### Consulter les options d'entraînement

```powershell
python -m src.imitation.local_target_train --help
python -m src.imitation.vector_experiment --help
```

Le premier entraîne le GAT de référence à partir d'un dataset d'imitation. Le second rassemble les expériences d'augmentation, d'entraînement et d'évaluation du GAT structuré. Pour reproduire précisément un entraînement, suivre ses fichiers de configuration et sa décision technique : les poids du modèle de référence et ceux du modèle structuré ne sont pas interchangeables.

Les entraînements sont locaux. Les commandes de lecture de la démonstration n'appellent pas une API de modèle pour entraîner les robots.

## 9. Formulation pour la soutenance

« Notre système combine une attribution distribuée des places, des commandes classiques et un réseau de neurones entraîné par imitation supervisée pour prédire les accélérations. La formation passe par une préparation de communication. En navigation, un planificateur classique sur carte connue fournit le détour. Nous évaluons les trajectoires avec l'erreur de formation, la stabilisation, les contacts et l'arrivée. Les résultats sont associés aux conditions réellement testées. »

Ne pas présenter la démo comme un robot réel équipé de caméras, une découverte autonome des obstacles inconnus, un réseau entraîné avec tous les effectifs dans tous les parcours, ou une garantie de réussite universelle. Le modèle structuré validé à 12/24/64 concerne la formation contrôlée ; son usage en navigation reste une autre étape.
