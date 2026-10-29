# 32 — Évaluation des détours et preuves visibles

Après la régression 7700 de la correction 015, le protocole a enregistré avant simulation les départs 8600/8601 et trois formes pour le parcours avec obstacle, avec deux parcours complémentaires du carré. Le réseau et les paramètres de planification n'ont pas été ajustés pendant cette campagne.

Les seize trajectoires réussissent : huit classiques et huit GAT protégées, sans contact. Les critères sont identiques à ceux des anciens essais ; seule la route et son temps de parcours dépendent désormais du planificateur. Les points de départ/d'arrivée de chaque référence sont conservés. Une destination obstruée reste refusée et n'est pas comptée comme réussite de navigation.

L'audit physique existant recalcule les contacts balayés et vérifie les équations d'intégration, limites, horloges et départs appariés. Un second audit repart des positions, vitesses et places cibles archivées : couverture bidirectionnelle, distance à destination, déformation maximale et fenêtre de maintien à l'instant annoncé. Des tests volontairement altérés vérifient qu'une erreur publiée incorrecte et une arrivée prématurée sont rejetées.

La synthèse de l'interface est calculée depuis les huit cas attendus et n'est affichée qu'avec les deux audits sans divergence et l'empreinte du modèle de référence. Les campagnes 009 gardent leurs échecs et sont étiquetées comme archives antérieures à cette correction. Le tableau courant n'est pas une nouvelle preuve de généralisation à toutes les formes, à 24/64 robots, aux obstacles inconnus ni au GAT brut structuré.

La distance minimale réelle sur l'expérience complète est 0,137991 m, préparation comprise. La marge du planificateur concerne la géométrie de la référence ; le suivi et la préparation continuent d'être évalués à partir des états physiques et des contacts. Aucun certificat global de sécurité n'est revendiqué.
