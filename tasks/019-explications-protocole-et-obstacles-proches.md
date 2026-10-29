# 019 — Expliquer les profils et le cercle, rapprocher les obstacles, livrer un guide

Statut : terminé. Suite complète : **634 tests réussis**, puis 32 tests ciblés incluant l'accès au guide. Demande utilisateur : message correct pour 8/16/24/64, comprendre le cercle initial, obstacles près des cibles et guide Markdown des expériences et commandes.

- Le texte de commande classique affiche l'effectif courant, sans phrase figée « à 64 » pour les autres effectifs.
- Un passage du modèle de référence vers 64 privilégie le GAT structuré disponible, en annonçant le changement. Un choix explicite classique est conservé. Les retours vers 8/16 annoncent le recours au modèle de référence, entraîné à 12 et exploratoire à ces effectifs.
- Le rendez-vous circulaire initial reste physiquement exécuté. Sa fonction est expliquée et la phase devient « Préparation : cercle provisoire ». Les métriques ne cachent pas cette partie de la trajectoire.
- Nouvelle scène multiple : les dix positions possibles sont sur les côtés proches des places finales du carré à 12 robots. Chaque centre d'obstacle est à au plus 1 m d'une place finale. Les anciens obstacles lointains aux murs sont retirés de cette scène.
- Guide [GUIDE_EXPERIENCES.md](../docs/GUIDE_EXPERIENCES.md), accessible aussi par `/guide.md` depuis la démonstration : sept expériences, trois contrôleurs, deux comparaisons, effectifs, protocole, métriques, lecture, exports, démarrage, tests, campagnes et options d'entraînement.

Campagne enregistrée sur la nouvelle scène : graines 8901/8902, comptes 5–10 plus une régression simple. Résultat : 8/8 par contrôleur sans contact. Audits de seize trajectoires, métriques et sept préparations appariées passés. Poids et seuils inchangés. Les anciens résultats 017 restent des archives de la disposition antérieure.

Décision : [35-protocole-et-guide](../docs/decisions/35-protocole-et-guide.md). Archives : `outputs/navigation-near-targets/validation-8901-8902/`.

Preuves versionnées : `evidence/019-near-targets-{evaluation,audit}.json`. Les messages classiques ont été vérifiés dans le navigateur aux quatre effectifs demandés ; le basculement explicite vers le GAT structuré à 64 a aussi été vérifié.
