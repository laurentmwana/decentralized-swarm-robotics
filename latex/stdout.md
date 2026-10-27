4.13 Lien entre agrégation et résultats observés
L'agrégation par moyenne rend le contrôleur compatible avec des voisinages de taille variable. Cela compte particulièrement dans les expériences de passage à l'échelle, puisque les robots du centre de l'arène peuvent recevoir plus de messages que ceux du bord quand leur nombre augmente. Une somme brute introduit une dépendance au nombre de voisins, sans croissance forcément monotone de sa norme, car les messages peuvent se compenser. La moyenne supprime le facteur multiplicatif du degré, mais ne garantit ni une distribution stable des messages ni la qualité de la formation.
À 40 et 100 robots, l'architecture ne s'effondre pas quand le nombre de noeuds change. C'est ce qu'on attend du partage des paramètres et de la moyenne des messages. Restons prudents, pourtant : le meilleur succès à 100 robots peut aussi venir de la plus grande densité de points disponibles sur un même contour. Pour attribuer précisément l'amélioration à la stratégie d'agrégation, il faudrait une ablation comparant moyenne, somme normalisée et absence de messages.
Le lien avec le flocking est limité mais instructif. La règle manuelle de regroupement s'appuie sur une interaction locale simple et produit de la cohésion sans représenter explicitement le cercle. Le GNN reçoit une consigne en plus et apprend ses réponses à partir d'exemples simulés. La comparaison historique montre donc la différence entre un comportement de rassemblement et un comportement conditionné par une cible, et elle n'oppose pas de façon définitive méthodes manuelles et méthodes apprises.
4.14 Analyse des métriques
La distance de Chamfer est la métrique principale, car elle répond directement à la question de la proximité entre la formation et la cible. Une valeur faible signifie qu'en moyenne les robots sont proches d'un point du contour et que les points du contour sont proches d'un robot. Elle ne décrit pas la dynamique, en revanche. Deux épisodes peuvent finir à la même distance alors que l'un atteint la cible vite et que l'autre oscille jusqu'au dernier pas. Le temps de convergence est donc une mesure complémentaire prévue pour les expériences finales.
Le taux de collisions demande lui aussi de la précaution. Ici, il est calculé à partir d'un seuil de sécurité et des positions simulées. Il ne mesure pas une collision mécanique avec contact, car le simulateur n'a ni masse, ni friction, ni rebond. Son rôle est de repérer les trajectoires qui placent trop souvent deux robots à une distance incompatible avec une marge de sécurité définie. Un taux plus bas est favorable, mais il ne compense pas une formation qui manquerait largement la cible.
La régularité des distances au plus proche voisin complète les deux mesures précédentes. Une formation peut être proche de la cible tout en laissant une partie du contour très dense
52

et une autre presque vide. Cette métrique en détecte une partie, sans mesurer parfaitement la couverture. Dans la campagne finale, il sera utile de présenter les métriques ensemble dans un tableau par condition, plutôt que de retenir celle qui donne le résultat le plus favorable.
4.15 Menaces à la validité
La première menace tient au faible nombre de répétitions. Dix épisodes par condition suffisent pour vérifier que les scripts fonctionnent et pour repérer un écart important entre deux méthodes. Ils ne suffisent pas à estimer précisément une moyenne, à comparer des distributions ou à établir un intervalle de confiance solide. Nous retenons cinquante répétitions par condition pour la suite, ce qui rendra les agrégats plus informatifs sans remplacer une analyse statistique complète.
La deuxième est l'écart entre le simulateur et un système réel. Les robots du projet connaissent un repère commun, échangent des messages exacts et sont modélisés par une vitesse appliquée directement. Sur un robot physique, les estimations de position sont bruitées, les roues ont de l'inertie et les communications peuvent être retardées. Ces hypothèses rendent l'étude historique plus contrôlable, mais limitent la portée des conclusions.
La troisième concerne les formes elles-mêmes. Les nombres et les lettres du curriculum réduit sont peu nombreux et servent pendant l'entraînement. Savoir former A, B ou C ne prouve pas qu'on généralise à toutes les lettres. Une expérience de généralisation devra réserver des formes entières avant l'entraînement, puis mesurer la performance sur ces cibles inconnues sans toucher aux poids du modèle.
4.16 Plan d'expériences pour la campagne finale
Le plan qui suit décrit des expériences à compléter, pas l'inventaire des résultats déjà obtenus. Des entraînements et une campagne comparative ont été conduits depuis, et leur analyse reste à écrire.
La campagne finale doit s'organiser autour de questions simples et mesurables. La première porte sur la qualité de formation. Pour chaque catégorie de formes, quelle est la distance à la cible, quelle est la régularité de la répartition et quelle proportion d'épisodes franchit le seuil de succès ? La deuxième porte sur l'apport de la communication. Que se passe-t-il quand on retire les messages des voisins en gardant la même consigne de forme ? La troisième porte sur la robustesse. Comment évoluent les mesures quand des robots disparaissent, quand le rayon de communication diminue ou quand les positions sont bruitées ?
53

Chaque condition doit être fixée avant de consulter les résultats. Une expérience de panne peut par exemple combiner trois tailles d'essaim, quatre fractions de retrait et cinquante graines. Les mêmes positions initiales peuvent être réutilisées pour comparer deux contrôleurs, afin que l'écart observé ne vienne pas d'une série de départs plus favorable à l'une des méthodes. Les données brutes sont ensuite conservées pour recalculer les moyennes, écarts-types et graphiques.
Une présentation complète associera un tableau chiffré et une figure. Le tableau donne les valeurs précises et leur dispersion, tandis que la figure montre la tendance quand un facteur varie, par exemple le pourcentage de pannes. Les images finales et les animations jouent un troisième rôle, car elles aident à comprendre la nature spatiale d'un échec. Cette complémentarité évite de réduire un comportement collectif à un seul nombre.
4.17 Interprétation prudente des résultats pilotes
Le pilote est utile parce qu'il a vérifié l'enchaînement complet des scripts, de la production des épisodes à l'agrégation des résultats et à la génération des graphiques, sans intervention manuelle. Il a aussi fait apparaître un écart important entre le GNN conditionné par la cible et la règle de regroupement. Cet écart justifie de poursuivre les expériences, mais il ne suffit ni à conclure à la supériorité générale d'une méthode, ni à valider l'implémentation actuelle.
La première raison est l'asymétrie d'information déjà discutée, la deuxième le nombre de répétitions, et la troisième le fait que les résultats portent surtout sur une cible circulaire. Une méthode peut très bien réussir un contour régulier et convaincre moins sur une lettre faite d'angles ou de traits éloignés. La campagne finale vise donc à remplacer une affirmation générale par un ensemble de résultats précis, chacun associé à une forme, une taille d'essaim et une perturbation données.
Cette prudence sert le mémoire. Un résultat négatif, comme une lettre difficile à couvrir ou un effondrement quand la portée devient trop faible, n'est pas un échec de rédaction. Il renseigne sur les limites de la politique et aide à décider quels mécanismes ajouter. Dans un travail sur les systèmes distribués, expliquer pourquoi une méthode cesse de fonctionner est souvent aussi utile que montrer ses meilleurs exemples.
4.18 Conclusion partielle
Nous avons présenté la chaîne d'implémentation et les résultats du prototype pilote. La perte diminue. Le cercle est mieux approché que par la règle de regroupement, dans le protocole retenu. Le même modèle s'exécute à différentes tailles d'essaim, et la distance moyenne varie peu lors des retraits simulés. Ce sont les données d'un pilote à dix graines par condition, marqué par une asymétrie d'information entre contrôleurs, un seuil de
54

succès sensible et des formations symboliques approximatives. La robustesse générale n'est donc pas établie. L'implémentation actuelle a depuis été entraînée et évaluée, mais ses chiffres ne figurent pas dans ce chapitre, et les résultats du pilote ne peuvent pas la valider.
Ce que le pilote établit, et ce qui reste à confirmer par des campagnes complémentaires, est résumé au Tableau 4.8.
Tableau 4.8 - Résultats établis par le pilote et validations restant à réaliser.

Établi par le pilote
La chaîne expérimentale fonctionne : simulateur, contrôleur, pertes, optimisation et rapports. Le GNN entraîné atteint une distance de Chamfer inférieure à celle de la règle de regroupement sur cible circulaire (20 robots, 10 graines). Le même modèle s'exécute avec 40 et 100 robots sans modification des poids.
La distance moyenne varie peu entre 0 et 20 % de retraits simulés.
Le curriculum réduit laisse poursuivre l'optimisation lorsque de nouvelles cibles sont ajoutées. Les propriétés de permutation, de bornage et de calcul des mesures sont testées.

À confirmer
L'analyse des campagnes conduites depuis reste à écrire dans ce mémoire.
La comparaison reste limitée à une cible circulaire et à une baseline ne recevant pas la même consigne.
La qualité de formation à grande échelle et l'effet de la densité du graphe ne sont pas isolés. Le protocole ne démontre pas une robustesse générale, avec dix épisodes par condition et sans intervalle de confiance. Les formes symboliques (lettres, nombres) restent approximatives et peu lisibles.
L'apprentissage réel d'une politique utile et sa généralisation à des formes non vues restent à valider.

Reste la discussion, non numérotée, qui replacera ces observations dans le contexte des travaux antérieurs et de leurs limites. La conclusion générale, elle aussi non numérotée, dégagera ensuite les enseignements et les perspectives du travail.

55

Discussion
Introduction
Revenons sur les résultats du pilote historique. Le chapitre 2 servira de repère, nous examinerons ensuite ce qui fragilise l'étude, puis ce qu'on peut en tirer pour concevoir des contrôleurs décentralisés. Le périmètre d'abord : il s'agit bien du pilote, pas de l'implémentation actuelle, dont les entraînements et la campagne comparative ne sont pas encore analysés dans ce mémoire. Implications pratiques et suite expérimentale closent le chapitre, avant la conclusion générale.
Interprétation des résultats
Le pilote montre d'abord que la chaîne historique tient debout. Le simulateur, le contrôleur à messages, la fonction de perte et l'optimisation fonctionnent ensemble et produisent une configuration circulaire plus proche de la cible que l'état initial. Rien là-dedans qui dise quoi que ce soit d'une convergence asymptotique, que nous ne pouvons pas affirmer. Les paramètres sont partagés, donc le même réseau s'applique à chaque robot, et l'agrégation locale s'accommode d'un nombre variable de voisins. Le protocole, lui, indique que l'essaim s'ajuste mieux au cercle que la règle de regroupement implémentée.
Il ne faut pas pour autant faire dire plus à cette observation. La baseline de regroupement ne reçoit pas la même consigne que le GNN, et elle n'est ni apprise ni adaptée à la tâche du cercle. L'écart de distance de Chamfer ne permet donc pas de savoir si le GNN fait mieux à information égale. La question que le pilote tranche est plus modeste : dans la configuration retenue, la politique entraînée se rapproche davantage du cercle qu'une règle qui se contente de regrouper les robots.
Un détail de protocole change la lecture des résultats. Le pilote aurait évalué le modèle issu du dernier niveau du curriculum réduit, et non un modèle entraîné uniquement sur le cercle. Aucun artefact conservé ne permet aujourd'hui de l'établir, et le seul poids dont la provenance est enregistrée provient d'un entraînement sur cible circulaire. Nous voulions évaluer le contrôleur réellement destiné à la démonstration. Si l'attribution au curriculum est exacte, les résultats mêlent l'effet du curriculum et celui de l'architecture.
56

Une campagne définitive devra les distinguer, en évaluant à la fois le modèle spécialisé sur le cercle et celui du curriculum, sur les mêmes épisodes.
Quand on retire 20 % des robots, la distance moyenne se dégrade peu. C'est un signal à creuser, pas une preuve de robustesse générale. Le taux de succès, nul à ce niveau de pannes avec le seuil retenu, rappelle d'ailleurs que la moyenne de Chamfer ne dit pas tout. À vingt robots, le premier franchissement du seuil est rare. Or le champ historique appelé « pas de convergence » n'exige pas de rester sous le seuil ensuite, et il ne dit pas, à lui seul, si la formation s'est stabilisée. La mesure actuelle, qui exige de rester sous le seuil jusqu'à la fin, suit une autre convention et se limite à un horizon fini.
Comparaison avec les travaux antérieurs
Situons maintenant ces résultats par rapport au chapitre 2, en distinguant ce qui relève de l'architecture de ce qui relève du protocole.
Tolstaya et al. montrent qu'un contrôleur GNN peut être entraîné pour des tâches de robotique en essaim sur un graphe de communication mobile [15]. Notre travail est de la même famille, mais la tâche est différente : il s'agit de converger vers une géométrie de cible donnée, non de suivre une trajectoire ni d'éviter des obstacles. Former une figure oblige à couvrir tout le contour, ce qui n'a pas d'équivalent direct en navigation.
Le pilote confirme l'intuition qui sous-tend les contrôleurs GNN décentralisés. L'architecture ne s'effondre pas quand le nombre de robots change. Là où nous nous démarquons, c'est sur l'apprentissage. Plusieurs travaux forment le contrôleur par imitation d'un expert, et il faut alors disposer d'une référence. Nous n'en avons pas. Le signal vient d'une perte de forme évaluée en fin d'épisode, rétropropagée à travers une simulation différentiable. Rien à imiter, donc, mais un horizon de rétropropagation aussi long que l'épisode.
Jiang, Huang et Guo proposent une formation décentralisée de bout en bout fondée sur un GNN [7]. Le rapprochement est direct, mais les hypothèses de perception diffèrent. Leurs observations viennent de capteurs embarqués, transformées en cartes locales, alors que notre prototype travaille sur des états simulés, avec une consigne géométrique fournie avant l'épisode. Comparer les performances n'aurait de sens qu'à hypothèses de perception comparables.
Les contrôleurs écrits à la main pour générer des formes [5, 9] restent la référence à atteindre. Ils sont lisibles et n'exigent aucun entraînement, mais il faut les concevoir pour chaque famille de formes. Nous ne prétendons pas les remplacer. Nous montrons simplement qu'une loi apprise, exécutée localement, peut être entraînée de manière reproductible sur des formes variées. La comparaison équitable, avec une règle manuelle qui reçoit une consigne de contour comparable, a depuis été conduite. Ses résultats restent à analyser et ne sont pas reportés ici. Les perspectives y reviennent.
57

Menaces à la validité
Validité interne
La première limite est le nombre de répétitions. Avec dix épisodes par condition, on peut valider une chaîne de mesure et repérer de gros écarts, mais pas estimer précisément une moyenne, encore moins comparer des distributions. S'y ajoute un couplage entre architecture et protocole. La provenance du modèle évalué n'est pas établie, ce qui brouille l'attribution des résultats au seul passage de messages.
La deuxième limite ne porte plus sur l'existence d'une baseline apprise sans communication. Cette variante a été entraînée et évaluée, et l'écart historique entre le GNN et le regroupement garde son interprétation, un système connaît sa cible et l'autre non. La limite actuelle vient des graines. Le contrôleur avec messages ne dispose que d'un seul entraînement, quand la variante sans communication en compte plusieurs. L'écart mesuré entre les deux ne peut donc pas encore être attribué aux seuls échanges, et les résultats décrivent un entraînement particulier, pas la performance attendue de l'architecture.
Validité externe
Dans le simulateur, les robots partagent un repère commun, connaissent leurs positions exactes et communiquent sans délai ni perte. Les collisions sont pénalisées par une fonction continue, sans dynamique de contact. Un robot en panne disparaît proprement du graphe, alors qu'un robot défaillant réel peut rester immobile, gêner les autres ou transmettre des valeurs aberrantes. L'arène est finie et ses bords modifient les trajectoires : rien ne s'extrapole donc à un espace ouvert. Enfin, les épisodes sont très courts (vingt pas dans le pilote), bien en deçà d'une mission réelle.
Ces hypothèses rendent l'étude contrôlable, mais elles en limitent la portée. Les résultats établissent une faisabilité en simulation 2D, pas un transfert vers des plateformes physiques.
Validité de construit
Les mesures elles-mêmes comportent des approximations. La distance de Chamfer prend en compte les deux ensembles de points mais ignore la lisibilité d'un symbole : deux configurations peuvent avoir des distances voisines alors que l'une se lit et l'autre pas. Le seuil de succès, fixé à 0,5 dans le pilote, est un choix opérationnel dont dépend directement le pourcentage de réussite. Le taux de collisions compte les rapprochements sous une distance de sécurité, pas des chocs mécaniques. Quant à la régularité des voisins, elle repère les amas sans garantir que tout le contour est couvert.
Inutiles, ces mesures ne le sont pas. Elles imposent juste de les présenter ensemble, avec leur définition, au lieu de retenir celle qui donne le résultat le plus flatteur.
58

Implications pour la conception de contrôleurs décentralisés
Trois enseignements se dégagent du travail, sous réserve des limites qui précèdent. Le premier : le prototype qui combine consigne et messages a produit des formations non triviales dans les scénarios étudiés. Laquelle des deux contributions pèse le plus, nous ne le savons pas encore, et les données disponibles ne permettent pas de les séparer. C'est l'objet de l'ablation sans communication, déjà entraînée et évaluée, et de la comparaison entre identifiant seul et géométrie, qui reste à conduire. Deuxième point, la moyenne : elle évite que la somme des messages grossisse mécaniquement avec le nombre de voisins. Elle ne garantit pas pour autant que leurs distributions restent identiques quand la densité change. Les résultats historiques à 40 et 100 robots sont compatibles avec ce choix de conception, sans en établir l'effet causal. Une attention à poids normalisés peut aussi former une moyenne pondérée, et ses propriétés comme sa performance méritent d'être évaluées plutôt que écartées d'emblée. Reste la progression cumulative, censée limiter l'oubli. Le pilote n'isole pas son gain par rapport à un entraînement non progressif, et conserver les anciennes cibles ne dispense pas de mesurer leur performance à chaque niveau, sur des épisodes appariés. Le tableau des pertes par cible du chapitre 4 est un premier pas dans cette direction.
Considérations pratiques et sociales
On justifie souvent les essaims robotiques par leur faible coût unitaire et leur redondance, et des plateformes comme les Kilobot montrent qu'on peut déployer des centaines de robots simples [13]. De quoi changer la manière d'évaluer un contrôleur. Un système qui fonctionne en simulation mais repose sur des hypothèses fragiles de repère ou de communication peut imposer, à l'échelle du groupe, des coûts de recalibrage considérables.
La simulation, elle, allège la facture. Entraîner et valider numériquement avant tout déploiement physique réduit le nombre d'essais sur le matériel, donc la consommation et les pertes qui en découlent. La reproductibilité des campagnes, assurée ici par des configurations et des graines explicites, va dans le même sens : rejouer une expérience coûte du temps de calcul, alors que refaire des robots coûte des matières.
Enfin, une formation de forme par un essaim est un dispositif visible dans l'espace public. Toute extension vers des plateformes réelles devra tenir compte du bruit, de l'occupation de l'espace et du consentement des personnes concernées, ainsi que des usages ambivalents d'un tel dispositif. Nous n'abordons pas ces questions de façon expérimentale dans ce mémoire. Elles font partie des limites de son périmètre.
59

Suite expérimentale
Les campagnes préparées prévoient au moins 50 départs par condition et cinq graines d'entraînement, avec une séparation nette des deux sources de variabilité. Elles incluent du bruit d'observation, la suppression aléatoire d'arêtes et la réduction de portée. Supprimer des arêtes n'équivaut pas à bruiter le contenu des messages. La comparaison prioritaire oppose la politique sans communication à son équivalent avec messages, à consigne identique. Les modules distinguent désormais proximité, couverture, régularité et sécurité, et un questionnaire en aveugle prépare l'étude de lisibilité. Mais leur existence ne remplace ni l'exécution des campagnes ni la collecte de réponses humaines. Le Tableau C.1 récapitule, contrôleur par contrôleur, l'information disponible et le statut de chaque comparaison.
Tableau C.1 - Information reçue par les contrôleurs actuels (voisinage, identifiant et géométrie). Cette matrice décrit ce que chaque contrôleur reçoit, pas des performances.

Contrôleur

Voisins ID forme Géométrie de la cible

Poids appris

Regroupement

Oui

Non Aucune

Non

(flocking)

Règle manuelle

Oui

Non Déplacement vers le point le plus

Non

proche (direction et distance)

GNN identifiant

Oui

Oui Aucune

Oui

seul

GNN avec géométrie

Oui

Oui Direction unitaire vers le point le plus

Oui

proche

GNN sans

Non

Oui Direction unitaire vers le point le plus

Oui

communication

proche

Les poids appris sont les paramètres à optimiser. Le pilote n'a évalué que l'ancien GNN et le regroupement. Deux variantes ont depuis été entraînées et évaluées, la règle manuelle à consigne comparable et la politique sans communication. Le contrôleur avec messages n'a encore qu'un seul entraînement. Le GNN sans communication garde ses 2 tours, mais son graphe et ses agrégats sont nuls.

Commençons par l'agrégation. Comparer la moyenne à une agrégation tronquée ou pondérée permettrait de mesurer la sensibilité du contrôleur à un voisin défaillant. Vient ensuite la profondeur. Comparer un, deux et trois tours à budget d'entraînement égal évalue l'effet conjoint de la portée informationnelle, de la profondeur et de la capacité paramétrique. Les distinguer demandera des contrôles supplémentaires.
À plus long terme, il faudra remplacer le repère commun par des observations relatives, une localisation distribuée ou une balise mesurable. Passer d'une dynamique à vitesse directe à un modèle cinématique ou dynamique plus réaliste permettra ensuite d'évaluer le transfert vers des robots physiques.

60

Conclusion partielle
Cette discussion a replacé les observations du pilote historique dans la littérature et précisé leurs limites : une consigne asymétrique, peu de répétitions, la dépendance à une seule graine d'entraînement et une simulation simplifiée. Elle a distingué l'exécution à taille variable de la généralisation mesurée, et la proximité géométrique de la lisibilité. La conclusion générale reprend les apports de l'étude et les validations expérimentales qui restent à conduire.
61

Conclusion générale et perspectives
Ce travail a mis en place une chaîne expérimentale reproductible pour étudier la formation de formes par un essaim piloté par GNN. Le système réunit un simulateur 2D, des formes représentées par des nuages de points, une politique locale à paramètres partagés et un apprentissage différentiable. Le cercle a donné une première observation, visuelle et chiffrée, du fonctionnement du prototype historique. Elle ne valide toutefois pas les performances de l'implémentation actuelle. Génération de données, entraînement, évaluation et visualisation sont des modules séparés, si bien que les expériences se rejouent à partir de leurs configurations.
Le résultat principal n'est pas encore une démonstration sur robots physiques. Il est plus précis. Dans le simulateur retenu, une politique GNN partagée peut recevoir des informations locales, échanger des messages avec ses voisins et produire des mouvements qui rapprochent l'essaim d'une cible. Le modèle peut aussi s'exécuter avec un nombre de robots différent de celui de l'entraînement. C'est une conséquence du partage des paramètres et de l'agrégation des messages ; la qualité obtenue, elle, doit être vérifiée expérimentalement pour chaque scénario.
Le protocole pilote suggère que le contrôleur appris s'ajuste mieux à une cible circulaire qu'une règle locale de regroupement, y compris dans les scénarios de retrait étudiés. Nous restreignons volontairement cette conclusion : nous n'avons que dix épisodes par condition, une seule graine d'entraînement et une baseline qui ne reçoit pas la même consigne. Pour les formes symboliques, la couverture reste approximative. La chaîne de traitement accepte plusieurs types de contours, mais on ne peut affirmer qu'elle forme des lettres ou des nombres lisibles et généralisables.
Les premières perspectives sont expérimentales. Il faut augmenter le nombre de répétitions, fixer un jeu de formes inconnues, mesurer des intervalles de confiance et conserver les épisodes défavorables autant que les favorables. Il faut aussi poursuivre avec les variantes déjà entraînées et évaluées, la politique sans communication et la règle manuelle à consigne comparable. Leur analyse reste à conduire, avec plusieurs graines d'entraînement pour le contrôleur avec messages et d'autres profondeurs de passage de messages. La consigne de la règle manuelle reste plus riche, puisqu'elle conserve la longueur du déplacement vers la cible. L'équivalence d'information doit donc être explicitée, et non supposée. Ces comparaisons permettront d'interroger l'apport propre des messages.
62

Les perspectives techniques touchent au réalisme du simulateur. Des bruits de position et de communication, des délais, des obstacles et une dynamique de robot non holonome rapprocheraient le cadre de situations réelles. À plus long terme, la consigne géométrique pourrait venir d'une perception locale, d'une balise ou d'une carte distribuée. Passer à une plateforme physique demanderait alors une estimation de pose, une stratégie de sécurité et une validation progressive, d'abord sur quelques robots, puis sur un essaim plus grand.
Le projet répond donc en partie à la question initiale. Il montre qu'un contrôleur GNN local peut former des formes en simulation, et il fournit les outils d'une évaluation plus rigoureuse. La suite devra établir, avec des expériences mieux contrôlées, dans quelles conditions cette approche apporte un avantage réel sur les méthodes classiques de contrôle distribué.
63

Bibliographie
[1] Filipe de Avila Belbute-Peres et al. « End-to-end differentiable physics for learning and control ». In : Advances in Neural Information Processing Systems. T. 31. 2018.
[2] Manuele Brambilla et al. « Swarm robotics : a review from the swarm engineering perspective ». In : Swarm Intelligence 7 (2013), p. 1-41. doi : 10.1007/s11721-0120075-2.
[3] Leon O. Chua et Yang Yang. « Cellular neural networks : Theory ». In : IEEE Transactions on Circuits and Systems 35.10 (1988), p. 1257-1272. doi : 10.1109/31. 7600.
[4] Justin Gilmer et al. « Neural Message Passing for Quantum Chemistry ». In : ICML. 2017, p. 1263-1272.
[5] M. Ani Hsieh, Vijay Kumar et Luiz Chaimowicz. « Decentralized controllers for shape generation with robotic swarms ». In : Robotica 26.5 (2008), p. 691-701. doi : 10.1017/S0263574708004323.
[6] Ali Jadbabaie, Jie Lin et A. Stephen Morse. « Coordination of groups of mobile autonomous agents using nearest neighbor rules ». In : IEEE Transactions on Automatic Control 48.6 (2003), p. 988-1001. doi : 10.1109/TAC.2003.812781.
[7] Chao Jiang, Xinchi Huang et Yi Guo. « End-to-end decentralized formation control using a graph neural network-based learning method ». In : Frontiers in Robotics and AI 10 (2023). doi : 10.3389/frobt.2023.1285412.
[8] James Kirkpatrick et al. « Overcoming catastrophic forgetting in neural networks ». In : Proceedings of the National Academy of Sciences 114.13 (2017), p. 3521-3526. doi : 10.1073/pnas.1611835114.
[9] Guannan Li et al. « Decentralized progressive shape formation with robot swarms ». In : Autonomous Robots 43.6 (2019), p. 1505-1521. doi : 10.1007/s10514-0189807-5.
[10] Ryan Lowe et al. « Multi-agent actor-critic for mixed cooperative-competitive environments ». In : Advances in Neural Information Processing Systems. T. 30. 2017, p. 6379-6388.
64

[11] Craig W. Reynolds. « Flocks, Herds, and Schools : A Distributed Behavioral Model ». In : SIGGRAPH. 1987, p. 25-34. doi : 10.1145/37401.37406.
[12] Stéphane Ross, Geoffrey Gordon et Drew Bagnell. « A reduction of imitation learning and structured prediction to no-regret online learning ». In : Proceedings of the 14th International Conference on Artificial Intelligence and Statistics. T. 15. Proceedings of Machine Learning Research. PMLR, 2011, p. 627-635.
[13] Michael Rubenstein et al. « Kilobot : a low cost robot with scalable operations designed for collective behaviors ». In : Robotics and Autonomous Systems 62.7 (2014), p. 966-975. doi : 10.1016/j.robot.2013.08.006.
[14] Franco Scarselli et al. « The graph neural network model ». In : IEEE Transactions on Neural Networks 20.1 (2009), p. 61-80. doi : 10.1109/TNN.2008.2005605.
[15] Ekaterina Tolstaya et al. « Learning Decentralized Controllers for Robot Swarms with Graph Neural Networks ». In : CoRL. 2020, p. 671-682.
[16] Petar Velickovi et al. « Graph attention networks ». In : International Conference on Learning Representations (ICLR). 2018.
[17] Manzil Zaheer et al. « Deep sets ». In : Advances in Neural Information Processing Systems. T. 30. 2017, p. 3391-3401.
65

