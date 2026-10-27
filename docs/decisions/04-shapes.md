# 04 — Formes cibles : repère, consigne, découpage

Daté : 2026-10-21. Fige les points §0.5 « repère commun 2D et consigne de forme » et « découpage ».

## Repère commun 2D
- Toutes les formes sont normalisées dans le **cadre commun [-1, 1]²** : centrage sur le centroïde des pixels, puis mise à l'échelle pour que le **rayon de cible** (plus grande distance centroïde→point) soit exactement `1.0`.
- Une seule échelle pour tous les types (géométries, lettres, chiffres, nombres) → distances comparables entre formes.
- Rendu intermédiaire : **128×128**, géométries tracées (matplotlib), caractères et nombres rendus par police (Pillow, Arial) ; binarisation au **milieu de la plage de gris observée** `(min+max)/2`, jamais à un seuil arbitraire.
- Échantillonnage : **point le plus éloigné** (farthest-point), premier point tiré par la graine de l'échantillonneur.
- **Une figure = un nom** (`letters_A`, `numbers_314`), sans résolution. On demande aux robots de dessiner une figure ; ses **points de référence** (`reference_points: 50` dans `config/shapes.yaml`, un seul nombre pour tout le projet) servent uniquement à mesurer la distance essaim → figure, de façon comparable entre toutes les expériences. Décision du 2026-10-28, remplace les résolutions multiples (20, 40, 50) ; le cercle d'entraînement passe de 40 à 50 points de référence.
- **Contrainte absolue (anomalie §0.6)** : le nombre de points distincts d'une cible doit être égal au nombre de points de référence demandé ; sinon `ShapeSampler` lève une erreur. Aucun `np.resize`, aucun remplissage par duplication, jamais.

## Consigne de forme (partagée, décentralisée)
- Consigne par défaut : **identifiant de forme** (`shape_id`, entier partagé entre tous les robots) → embedding appris par le GNN (phase 4).
- Variante séparée : **vecteur unitaire vers le point de contour le plus proche**, recalculé à chaque pas par chaque robot (local, sans affectation globale). Elle est activable par la clé de config `use_target_geometry` et ses **résultats sont rapportés séparément** (plan phase 10) ; jamais mélangée aux runs « identifiant seul » (AGENT.md §6).
- La cible complète (nuage de points) n'est jamais une entrée du contrôleur pendant l'exécution : seule la consigne ci-dessus est partagée.

## Découpage entraînement / validation / test — FIGÉ
Sert à la sélection de modèle (validation) et jamais au réglage. Le test ne sert qu'à la généralisation (phases 10–11).

| Type | Entraînement | Validation | Test (réservé, généralisation) |
|---|---|---|---|
| Géométries (7) | cercle, carré, triangle, ellipse, croix | pentagone | étoile |
| Chiffres (10) | 0,1,2,3,4,5 | 6,7 | 8,9 |
| Lettres (26) | A–K, M–U (20 lettres) | V, W, X | Y, Z, et L réservée (3) |
| Nombres (9–999) | 12,45,77,100,108,121,144,200,222,333,404,500,707 | 99,111,555 | 9,314,606,789,999 |

- **Correction du 2026-10-27 (fuite de test)** : le nombre `9` est le même glyphe que le chiffre `9` (test). Il était en entraînement, ce qui exposait une forme de test ; il est déplacé en test. `ConfigLoader` refuse désormais tout nombre à un chiffre placé dans un autre split que le chiffre identique.
- **Correction du 2026-10-27 (rendu)** : les nombres à 2 et 3 chiffres débordaient du canevas (« 314 » rendu « 31 »). Le canevas est maintenant dimensionné sur le texte ; catalogue à régénérer (`make shapes`), checkpoints antérieurs incompatibles avec le nouvel index.

- L'index `outputs/shapes/index.csv` porte une ligne par forme avec sa colonne `split` ; le découpage est **verrouillé par hash** : `ShapeSplit` rejette toute tentative d'échantillonner une forme d'un split non demandé.
- Le cercle seul est utilisé en phases 3–8 ; les autres formes entraînement à partir du curriculum (phase 9).
