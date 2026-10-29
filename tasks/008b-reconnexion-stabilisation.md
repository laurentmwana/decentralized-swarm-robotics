# 008b — Reconnexion locale et stabilisation du professeur

Statut : réalisation et validation ciblée terminées ; 314 tests réussis. Les neuf défauts de stabilisation et le départ déconnecté sont corrigés sur les scénarios conservés. Suite : contrat et collecte 008c, avec contrôle de qualité propre.

## Livraison et résultat

- Rendez-vous de 8 s et freinage de 1 s, commandes basées sur état propre, identifiant et voisins directs.
- Réinitialisation locale de l'affectation après rendez-vous ; 144 tours réellement simulés, soit début du contrôle de forme à 23,4 s dans le même horizon de 40 s.
- Détour activé uniquement lorsque la commande viole une contrainte de sécurité ; version antérieure conservée par défaut.
- Comparaison sur 7001–7002 : 47/112 précédemment, 56/112 avec seul détour corrigé, 112/112 avec reconnexion complète ; zéro collision dans ces comparaisons filtrées.
- Validation figée sur 7003–7005 : 168/168 réussites, zéro collision observée, stabilisation moyenne 29,396 s. Trois départs distincts, chacun partagé par 56 formes.
- Audit : limites physiques respectées, vitesse nulle et graphe inchangé pendant l'affectation. Le filtre a enregistré 3 815 infaisabilités robot/pas avec repli, explicitement conservées ; aucune garantie universelle n'est affirmée.

Décision : `docs/decisions/21-reconnexion-et-stabilisation.md`. Preuves : `tasks/evidence/008b-{risk-detours-evaluation,reconnection-evaluation,reconnection-validation,reconnection-validation-audit}.json`.

Les constats ci-dessous décrivent le point de départ, conservé pour comparaison.

## Constats conservés

- Départ dispersé 7001 : réseau en deux composantes (11 robots + 1 isolé). Le protocole stationnaire refuse de déclarer une affectation complète ; les 56 épisodes échouent et restent dans le rapport.
- Départ dispersé 7002 : affectation résolue pour les 56 formes, 47 réussites, aucune collision ; neuf défauts de stabilisation persistent.
- Le filtre de sécurité a rencontré 10 infaisabilités robot/pas sur le pilote en anneau et 4 sur le départ connecté dispersé. L'absence de collision observée ne suffit pas à établir une garantie.

## Travail attendu

1. Définir une phase de reconnexion avec état propre, consigne commune et voisins seulement, sans utiliser la connaissance globale des composantes dans les commandes.
2. Compter son temps, les échanges et les collisions ; préserver toutes les limites physiques.
3. Définir la transition vers l'affectation sans supposer gratuitement un graphe statique connecté pendant un mouvement.
4. Examiner les trajectoires des neuf échecs de stabilisation et traiter les blocages locaux avec une régression reproductible.
5. Comparer sur des conditions conservées puis valider sur de nouveaux départs, en gardant les échecs et sans modifier les seuils pour les masquer.

## Références

`docs/decisions/19-affectation-locale-professeur-formes.md`, `20-filtre-local-securite-formes.md`.
Rapports : `tasks/evidence/008-shape-teacher-{pilot,safety-pilot,safety-validation}.json` et `008-connectivity-audit.json`.

Le catalogue de 56 formes reste inchangé. Aucun apprentissage neuronal ne doit être présenté comme réussi avant vérification du professeur et définition des entrées/partitions multi-formes.
