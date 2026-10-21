"""Empêche le retour des anciens identifiants et clés français."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml


FORBIDDEN_NAMES = {
    "commande", "cible", "modele", "adjacence", "reinitialiser", "vecteur_contour", "n_tours", "n_parametres",
    "graine", "graines", "formes", "valeurs", "lignes", "nuage", "nuages", "sortie", "chemin", "dossier",
    "pertes", "profils", "regles", "controleur", "controleurs", "synthese", "campagne", "meilleur", "dernier",
    "distance_finale", "reussite", "courbe_distance", "taux_collisions", "regularite_rapportee", "composantes_finale",
}


def test_python_identifiers_are_english():
    for root in (Path("src"), Path("tests")):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                identifiers = []
                if isinstance(node, (ast.Name, ast.Attribute, ast.arg)):
                    identifiers = [node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else node.arg]
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    identifiers = [node.name]
                for identifier in identifiers:
                    assert identifier.isascii() and identifier not in FORBIDDEN_NAMES, (path, node.lineno, identifier)


def test_configuration_keys_are_english():
    def validate(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, str):
                    assert key.isascii() and key not in FORBIDDEN_NAMES, key
                validate(child)
        elif isinstance(value, list):
            for child in value:
                validate(child)

    for path in Path("config").glob("*.yaml"):
        validate(yaml.safe_load(path.read_text(encoding="utf-8")))
