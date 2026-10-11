# Étape 3 — Jeu de formes cibles

## Objectif
Produire les formes que l'essaim doit apprendre à former, sous deux formats : **images** (pour les visualiser et les archiver) et **nuages de points** (ce que le modèle utilise vraiment).

## Contenu du jeu de formes, par niveaux de difficulté
1. **Formes géométriques** : cercle, carré, triangle, ligne, croix, étoile, anneau.
2. **Alphabet** : lettres A à Z.
3. **Chiffres** : 0 à 9.
4. **Nombres** : de 10 à 999, obtenus en juxtaposant des chiffres.

Cet ordre sert aussi d'**apprentissage progressif** (étape 5) : on commence par le plus simple.

## Génération des images
- Dessiner chaque forme en noir sur fond blanc, résolution fixe (par exemple 128 par 128 pixels), centrée, avec marges.
- Pour les lettres et chiffres : utiliser une police épaisse et lisible pour que la forme reste reconnaissable avec peu de points.
- Pour les nombres : placer les chiffres côte à côte avec un espacement constant, puis recentrer l'ensemble ; adapter la largeur de l'image au nombre de chiffres.
- Ajouter des variantes : rotation légère, changement d'échelle, plusieurs polices. Cela aide le modèle à généraliser.
- Nommer les fichiers de façon systématique, par exemple `shapes/geom_circle.png`, `shapes/alpha_A.png`, `shapes/digit_7.png`, `shapes/number_042.png`.

## Conversion en nuages de points cibles
1. Charger l'image et repérer les pixels noirs (le contenu de la forme).
2. Choisir combien de points représentent la forme : en pratique **autant que de robots** (ou un peu plus, pour tester).
3. Répartir ces points de façon régulière sur la forme (éviter qu'ils s'agglutinent). Une méthode simple : échantillonner puis écarter les points trop proches.
4. Normaliser les coordonnées pour qu'elles tiennent dans l'arène du simulateur.
5. Sauvegarder (fichier `.npy` ou `.json`) à côté de l'image.

## Question importante à trancher : comment le robot connaît-il la forme ?
Cette décision change la nature décentralisée du système. À noter dans `decisions.md`. Options :
- **A. Forme décrite à tous** : chaque robot reçoit une description compacte de la forme (par exemple un identifiant ou un petit vecteur) et sa position dans un repère commun. Simple, c'est le point de départ recommandé.
- **B. Forme portée par quelques robots** : seuls quelques robots connaissent la forme, les autres l'apprennent par leurs voisins. Plus décentralisé, plus difficile.
- **C. Forme émergente** : un seul réseau qui doit en produire plusieurs selon un signal. Variante de A.

Commencer par A, annoncer B comme extension.

## Découpage des données
- **Entraînement** : une grande partie des formes.
- **Validation** : quelques formes pour régler les paramètres.
- **Test** : formes jamais vues (par exemple certaines lettres ou nombres gardés de côté) pour mesurer la généralisation.
Ne jamais regarder le test pendant le réglage.

## Livrable : archive zip
Un fichier `shapes.zip` contenant les images PNG organisées en dossiers (`geometry/`, `alphabet/`, `digits/`, `numbers/`), les nuages de points correspondants, et un fichier `index.csv` listant chaque forme, sa catégorie, son découpage (entraînement, validation, test) et le nombre de points.

## À faire
- [ ] Générateur de formes géométriques
- [ ] Générateur de lettres et chiffres
- [ ] Générateur de nombres 10 à 999
- [ ] Conversion en nuages de points
- [ ] Découpage entraînement, validation, test
- [ ] Création de `shapes.zip` et de `index.csv`
- [ ] Tableau d'aperçu (planche contact) pour vérifier visuellement

## Critères de réussite
On reconnaît chaque forme à l'œil, y compris avec le nombre de points utilisé pour l'essaim ; le zip se régénère par une seule commande.

## Pièges
- Lettres trop fines : avec peu de points, elles deviennent illisibles. Épaissir les traits.
- Nombres à 3 chiffres trop larges pour l'arène : prévoir une mise à l'échelle.
- Fuite de données : mêmes formes dans l'entraînement et le test.
