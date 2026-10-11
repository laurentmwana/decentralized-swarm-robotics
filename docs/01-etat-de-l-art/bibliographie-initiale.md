# Bibliographie initiale — étape 1

Cette liste est une **présélection vérifiée par les notices bibliographiques et les résumés**. Elle constitue l'ordre de lecture initial ; elle ne remplace pas les fiches de lecture complètes exigées par l'étape 1.

## Articles prioritaires

| Référence | Axe | Apport attendu pour le mémoire | Statut |
|---|---|---|---|
| Brambilla, Ferrante, Birattari et Dorigo (2013), *Swarm robotics: a review from the swarm engineering perspective*, *Swarm Intelligence*, 7, 1–41. DOI: [10.1007/s11721-012-0075-2](https://doi.org/10.1007/s11721-012-0075-2) | Robotique en essaim | Définitions, principes d'ingénierie d'essaim et limites de déploiement. | À lire |
| Olfati-Saber (2006), *Flocking for Multi-Agent Dynamic Systems: Algorithms and Theory*, *IEEE Transactions on Automatic Control*, 51(3), 401–420. DOI: [10.1109/TAC.2005.864190](https://doi.org/10.1109/TAC.2005.864190) | Contrôle distribué | Référence de comparaison pour les règles de flocking, cohésion et séparation. | À lire |
| Fax et Murray (2004), *Information Flow and Cooperative Control of Vehicle Formations*, *IEEE Transactions on Automatic Control*, 49(9), 1465–1476. DOI: [10.1109/TAC.2004.834433](https://doi.org/10.1109/TAC.2004.834433) | Formation de groupe | Rôle de la topologie de communication dans la stabilité des formations. | À lire |
| Reynolds (1987), *Flocks, Herds, and Schools: A Distributed Behavioral Model*, *SIGGRAPH '87*, 25–34. DOI: [10.1145/37401.37406](https://doi.org/10.1145/37401.37406) | Baseline comportementale | Origine des règles d'alignement, cohésion et séparation à adapter comme référence simple. | À lire |
| Scarselli et al. (2009), *The Graph Neural Network Model*, *IEEE Transactions on Neural Networks*, 20(1), 61–80. DOI: [10.1109/TNN.2008.2005605](https://doi.org/10.1109/TNN.2008.2005605) | Fondements GNN | Cadre historique pour les calculs sur graphes. | À lire |
| Gilmer et al. (2017), *Neural Message Passing for Quantum Chemistry*, ICML, 1263–1272. [Article](https://proceedings.mlr.press/v70/gilmer17a.html) | Échange de messages | Formalisation de l'agrégation de messages, directement transposable au contrôleur partagé. | À lire |
| Battaglia et al. (2018), *Relational Inductive Biases, Deep Learning, and Graph Networks*. [arXiv:1806.01261](https://arxiv.org/abs/1806.01261) | GNN et généralisation | Justification conceptuelle de l'induction relationnelle et des architectures graphe. | À lire |
| Kipf et Welling (2017), *Semi-Supervised Classification with Graph Convolutional Networks*. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907) | GNN | Référence d'architecture d'agrégation de voisinage. | À lire |
| Hamilton, Ying et Leskovec (2017), *Inductive Representation Learning on Large Graphs*, NeurIPS. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216) | GNN inductif | Éclairage sur la généralisation à de nouveaux nœuds et tailles de graphe. | À lire |
| Tolstaya et al. (2019), *Learning Decentralized Controllers for Robot Swarms with Graph Neural Networks*. [arXiv:1903.10527](https://arxiv.org/abs/1903.10527) | GNN pour essaims | Contrôleur local partagé, graphe de communication dynamique et expérimentation de flocking. | À lire en priorité |
| Gama, Tolstaya et Ribeiro (2020), *Graph Neural Networks for Decentralized Controllers*. [arXiv:2003.10280](https://arxiv.org/abs/2003.10280) | GNN pour contrôle décentralisé | Propriétés d'équivariance, stabilité et communication retardée. | À lire en priorité |
| Gama et al. (2021), *Synthesizing Decentralized Controllers with Graph Neural Networks and Imitation Learning*. [arXiv:2012.14906](https://arxiv.org/abs/2012.14906) | Apprentissage par imitation | Comparaison utile avec l'apprentissage direct prévu dans le projet ; traite flocking et planification multi-agent. | À lire en priorité |
| Hsieh, Kumar et Chaimowicz (2008), *Decentralized Controllers for Shape Generation with Robotic Swarms*, *Robotica*, 26(5), 691–701. DOI: [10.1017/S0263574708004323](https://doi.org/10.1017/S0263574708004323) | Formation de formes | Contrôleurs locaux pour des courbes planes fermées, évitement de collisions et propriétés de convergence. | À lire en priorité |
| Li et al. (2018), *Decentralized Progressive Shape Formation with Robot Swarms*, *Autonomous Robots*, 43(6), 1505–1521. DOI: [10.1007/s10514-018-9807-5](https://doi.org/10.1007/s10514-018-9807-5) | Formation de formes | Contrôle de formation progressive fondé sur des décisions locales. | À lire en priorité |
| Jiang, Huang et Guo (2023), *End-to-End Decentralized Formation Control Using a Graph Neural Network-Based Learning Method*, *Frontiers in Robotics and AI*, 10:1285412. DOI: [10.3389/frobt.2023.1285412](https://doi.org/10.3389/frobt.2023.1285412) | GNN pour formation | GNN décentralisé entraîné par démonstrations, formation triangulaire et essai de passage à l'échelle. | À lire en priorité |

## Positionnement provisoire

Le mémoire combinera les règles de contrôle distribué classiques (cohésion, séparation, formation) avec un contrôleur GNN unique et partagé. La contribution expérimentale visée est la formation de formes cibles, puis l'évaluation de la résistance aux pannes et du passage à l'échelle. Les travaux de Tolstaya et de Gama sont les références les plus proches du contrôleur décentralisé par GNN, mais leurs tâches principales sont le flocking et la planification, plutôt que la formation de lettres, chiffres et nombres avec le protocole de robustesse retenu ici.

## Prochaine action de lecture

Pour chaque article prioritaire, créer une fiche dans le format défini par `docs/01-etat-de-l-art.md` : problème, méthode, données ou simulateur, résultats, limites et utilité pour ce mémoire. La liste doit ensuite être étendue à 15–25 sources réellement lues.
