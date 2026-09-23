"""
Calculatrice de Carl.

Un modèle de langage prédit des mots, il ne calcule pas : même les gros
modèles se font aider par un outil. Carl apprend (data/calculs.py) à écrire
« [calc: 17*23 = » au lieu d'inventer un résultat ; chat.py repère ce motif
pendant la génération, calcule ici, insère « 391] » et le laisse continuer.
À l'affichage, « [calc: 17*23 = 391] » devient « 391 ».
"""

from __future__ import annotations

import ast
import operator
import re

# Motif en cours d'écriture : « [calc: <expression> = » en fin de texte.
APPEL = re.compile(r"\[calc:\s*([^\]=]+?)\s*=\s*$")
# Motif complet, à remplacer par son résultat à l'affichage.
COMPLET = re.compile(r"\[calc:\s*[^\]=]+?=\s*([^\]]*)\]")

_OPERATEURS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(noeud):
    if isinstance(noeud, ast.Constant) and isinstance(noeud.value, (int, float)):
        return noeud.value
    if isinstance(noeud, ast.BinOp) and type(noeud.op) in _OPERATEURS:
        gauche, droite = _eval(noeud.left), _eval(noeud.right)
        if isinstance(noeud.op, ast.Pow) and abs(droite) > 100:
            raise ValueError("exposant trop grand")
        return _OPERATEURS[type(noeud.op)](gauche, droite)
    if isinstance(noeud, ast.UnaryOp) and type(noeud.op) in _OPERATEURS:
        return _OPERATEURS[type(noeud.op)](_eval(noeud.operand))
    raise ValueError("expression non autorisée")


def formater(x: float) -> str:
    """Nombre à la française : 391 ; 12,5 ; 0,3333."""
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    if isinstance(x, int):
        return str(x)
    return f"{x:.4f}".rstrip("0").rstrip(".").replace(".", ",")


def calculer(expression: str) -> str:
    """Évalue une expression arithmétique (+ - * / ^ % et parenthèses), rien d'autre."""
    expr = expression.replace(",", ".").replace("^", "**").replace("×", "*").replace("x", "*").replace(":", "/")
    try:
        return formater(_eval(ast.parse(expr, mode="eval").body))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError):
        return "erreur"


def afficher(texte: str) -> str:
    """Remplace chaque « [calc: expr = résultat] » par le résultat."""
    return COMPLET.sub(lambda m: m.group(1).strip(), texte)
