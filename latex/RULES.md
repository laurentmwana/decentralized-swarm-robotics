# Remarques pour tous

Consignes de rédaction et de mise en forme du mémoire (Prof. Dr Selain KASEREKA KABUNGA).

---

## 1. Structure des chapitres

- Le **premier point** de chaque chapitre est **Introduction**.
- Le **dernier point** de chaque chapitre est **Conclusion partielle**.
- Dans la conclusion partielle, il faut :
  - résumer le chapitre en **un paragraphe** ;
  - **annoncer le chapitre suivant**.

---

## 2. Figures, tableaux et équations

### 2.1 Annonce et référencement

- Toutes les figures, tableaux et formules doivent être **annoncés avant** d'être insérés.
  - Exemple : *La Figure \ref{fig:paysage} illustre le paysage africain sur mesure qui permet de comprendre cette situation.*
- Quand on fait référence à une figure, un tableau ou une équation, écrire le mot avec la **première lettre en majuscule** : *Figure*, *Tableau*, *Équation*.
  - Exemple : *… comme illustré sur la Figure \ref{fig:archi}.*
- Pour les équations, utiliser `\eqref{eq:mathequation}` : cela insère directement les parenthèses.

### 2.2 Légendes (captions)

| Élément | Position de la légende |
|---|---|
| Tableau | **Au-dessus** du tableau |
| Figure | **En dessous** de la figure |

- Pour toute figure ou tout tableau **qui ne vient pas de vous**, mettre la référence dans la légende.
  - Exemple : *Figure 1.1 – Cloud Computing `\cite{refKey}`*
  - `\cite{refKey}` récupère automatiquement le numéro de la référence dans la bibliographie.

### 2.3 Lisibilité

- **Agrandir la taille des images**, sinon elles ne seront pas bien visibles après impression sur papier.

---

## 3. Pages de garde

- Il y a des fautes sur les pages de garde : bien les vérifier.
- Écrire toujours **Mathématiques** (avec un « s » à la fin).
- Mentions à respecter :

```text
FACULTE DES SCIENCES ET TECHNOLOGIES

Mémoire présenté et défendu en vue de l'obtention du Grade de
Licencié en Sciences et Technologies.

Filière : Informatique

Directeur : Prof. Dr Selain KASEREKA KABUNGA
```

---

## 4. Pages liminaires

### 4.1 Épigraphe

- Écrire **ÉPIGRAPHE** et non « EPIGRAPHIE ».

### 4.2 Dédicace et Remerciements

- À la fin (en dessous), écrire votre **nom complet** : c'est une signature, c'est important.

### 4.3 Liste des abréviations

- Par **ordre alphabétique**.
- Les premières lettres des mots qui expliquent le sigle sont en **majuscule**.
  - Exemple : `RDC : République Démocratique du Congo`
- Détailler une abréviation **n'est pas la traduire en français** : donner la forme originale.
  - Exemple : ~~KNN : K-plus Proches Voisins~~ → **`KNN : K-Nearest Neighbors`**

---

## 5. Ordre du contenu du mémoire

1. Épigraphe
2. Dédicace
3. Remerciements
4. Liste des abréviations
5. Liste des figures
6. Liste des tableaux
7. Résumé + mots clés
8. Abstract + Keywords
9. **Introduction générale**
   - Contexte de l'étude
   - Problématique
   - Hypothèses
   - Objectifs poursuivis
   - Méthodes et techniques utilisées
   - Contribution de l'étude
   - Plan sommaire / Subdivision du travail
10. **Chapitre 1** : Préliminaires // Généralités // Concepts de base // Concepts théoriques de base
11. **Chapitre 2** : Étude bibliographique // Étude des travaux connexes // État de l'art
12. **Chapitre 3** : Méthodologie // Solution proposée
13. **Chapitre 4** : Implémentation, Expérimentations et résultats

---

## 6. Structure détaillée du Chapitre 2 : État de l'art

1. **Introduction**
2. **Méthodes traditionnelles**
3. **Méthodes avancées**
4. **Quelques travaux connexes**
   - **Titre du travail 1** `[Réf]`
     - Contexte de l'étude et questions de recherche
     - Méthodologie adoptée
     - Aperçu sur les résultats
     - Critique
   - **Titre du travail 2** `[Réf]`
     - Contexte de l'étude et questions de recherche
     - Méthodologie adoptée
     - Aperçu sur les résultats
     - Critique
   - **Titre du travail 3** `[Réf]`
     - Contexte de l'étude et questions de recherche
     - Méthodologie adoptée
     - Aperçu sur les résultats
     - Critique
5. **Positionnement**
6. **Conclusion partielle**

---

## 7. Le résumé

Le résumé doit contenir, dans l'ordre :

1. **Une petite contextualisation**
2. **Quelques lacunes** trouvées dans les travaux de la littérature
   - Exemple : *Ces dernières années, plusieurs travaux ont été menés dans …, cependant, la plupart de ces travaux se sont plus focalisés sur … (ont présenté des lacunes dans …).*
3. **Comment le problème a été résolu**
   - Exemple : *Dans ce mémoire, nous proposons …*
4. **Quelques résultats**
   - Exemple : *Les résultats obtenus montrent que …*
5. **L'apport du travail**
   - *Ce travail fournit …*

> Ajouter **au moins 6 mots clés** après le résumé.

---

## 8. Références et orthographe

- Ne **pas** mettre les références dans les titres : les mettre **dans le texte**.
- Il y a beaucoup de mentions **`[?]`**, surtout pour les références : bien vérifier la compilation et les clés de citation.
- Il y a beaucoup de fautes d'orthographe.
  - **Conseil** : donner à Copilot ou ChatGPT les paragraphes **un à un**, en demandant de corriger les fautes d'orthographe **sans changer le texte**.
  - **Attention** : la relecture portera sur le **FOND** et non sur les fautes.

---

## 9. Check-list rapide

- [ ] Chaque chapitre commence par *Introduction* et finit par *Conclusion partielle*
- [ ] Conclusion partielle = résumé en un paragraphe + annonce du chapitre suivant
- [ ] Toute figure, tout tableau et toute équation sont annoncés avant d'apparaître
- [ ] *Figure*, *Tableau*, *Équation* avec majuscule ; `\eqref` pour les équations
- [ ] Légende du tableau au-dessus, légende de la figure en dessous
- [ ] Source citée dans la légende pour tout élément externe (`\cite{...}`)
- [ ] Images agrandies pour l'impression
- [ ] Pages de garde corrigées (Mathématiques, faculté, filière, directeur)
- [ ] ÉPIGRAPHE correctement écrit
- [ ] Nom complet en signature à la fin de la Dédicace et des Remerciements
- [ ] Abréviations par ordre alphabétique, en anglais d'origine, majuscules aux mots
- [ ] Plus aucun `[?]`
- [ ] Orthographe relue paragraphe par paragraphe
- [ ] Pas de références dans les titres
- [ ] Ordre du contenu respecté
- [ ] Résumé complet + au moins 6 mots clés