# 35 — Rendre le contrôleur et la préparation compréhensibles

La mention figée du profil classique à 64 robots et les remplacements automatiques de commande rendaient ambigu le rôle du réseau. L'interface affiche désormais la commande réellement choisie avec l'effectif courant. Lorsqu'un changement est requis, il est annoncé. Le modèle structuré couvre 12/24/64 en formation contrôlée ; le modèle de référence reste entraîné à 12, avec 8/16/24 exploratoires. Un choix explicite classique est respecté.

Le cercle de rendez-vous sert à la préparation des communications avant la répartition distribuée des places. Le retirer changerait le protocole et le domaine évalué. Nous expliquons donc la phase dans l'interface et le guide ; nous ne masquons pas sa trajectoire ni ses contacts éventuels pour donner l'apparence d'un départ direct.

La scène de cinq à dix obstacles ne place plus ses obstacles supplémentaires aux murs éloignés : ils sont proches des places cibles du carré final à 12 robots. Les centres sont à au plus 1 m d'une de ces places. Les disques et marges restent considérés par le planificateur et le filtre, et les placements finaux ne sont pas déplacés pour masquer un conflit.

La nouvelle campagne utilise les départs 8901/8902, un protocole enregistré avant exécution et les mêmes poids, seuils et audits. Les seize trajectoires réussissent sans contact : sept cas multiples et une régression simple par contrôleur. Cela valide cette scène avec le carré à 12 ; les autres formes/effectifs et les environnements inconnus ne sont pas couverts par ce résultat.

Le guide Markdown fait partie du livrable et peut être ouvert depuis la démo. Il distingue entraînement, tests logiciels et campagnes physiques, ainsi que commande neuronale, attribution classique, préparation classique et planification sur carte connue.
