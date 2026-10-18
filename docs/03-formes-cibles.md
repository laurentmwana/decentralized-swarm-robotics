# 03 — Formes cibles

## Chaîne de génération (déterministe, séparée de l'entraînement)
1. **Rendu** d'une image 128 × 128 (police de caractères ou tracé géométrique).
2. **Binarisation** au milieu de la plage de gris (pas de seuil à régler par forme).
3. **Échantillonnage par point le plus éloigné** : le premier point vient de la graine, chaque point suivant est le pixel le plus éloigné des points déjà retenus ; répartit les points sur tout le contour, sans grappes ni trous.
4. **Normalisation** dans un cadre commun centré sur l'arène, à l'échelle du rayon de cible configuré (une lettre étroite et un cercle large ont la même amplitude).
5. Sauvegarde : tableau de points + image d'origine + ligne d'index ; **planche de contrôle** de toutes les formes, à inspecter avant tout entraînement.

## Résolutions
- Jeu standard : **20 points** (rayon de cible 2,0).
- Campagne complexe : **50 points** (rayon de cible 2,8).
Le nombre de points est lié au nombre de robots : si la forme a plus de détails que de robots, certains détails ne peuvent pas être rendus. À 20 points, les nombres (010, 123) ne sont pas lisibles.

## Contenu du jeu
7 formes géométriques (cercle, carré, triangle, …), 26 lettres, 10 chiffres, nombres à trois caractères (extension prévue : 9 à 999).

## Consigne fournie aux robots
- Identifiant de forme (toutes campagnes).
- Campagne complexe : en plus, **vecteur vers le point de contour le plus proche** (requête géométrique locale, de type carte de mission). Information plus riche : ne pas comparer ces campagnes sans le préciser.

## Séparation des données
Définir explicitement les fichiers d'entraînement / validation / test avant de parler de généralisation à des formes non vues.
