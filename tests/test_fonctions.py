"""Les fonctions de l'ordinateur (fonctions.py), à date fixe."""

from datetime import date

from fonctions import calcul, conversion, dates, minuteur, repondre

JOUR = date(2026, 9, 25)  # un vendredi


def test_dates():
    assert dates("quel jour tombe Noël ?", JOUR).texte == "Noël tombe un vendredi (vendredi 25 décembre 2026)."
    assert "dans 91 jours" in dates("dans combien de jours est Noël ?", JOUR).texte
    assert dates("le 14 juillet 2027 tombe quel jour ?", JOUR).texte.startswith("Le 14 juillet tombe un mercredi")
    assert dates("dans 10 jours on sera quel jour ?", JOUR).texte == "Dans 10 jours, nous serons le lundi 5 octobre 2026."
    assert dates("Qui a gagné le 14 juillet ?", JOUR) is None  # une date, mais pas une question de calendrier


def test_conversions():
    assert conversion("5 miles en km").texte == "5 miles = 8,05 km."
    assert conversion("100 °F en °C").texte == "100 °F = 37,78 °C."
    assert conversion("3 pieds en mètres").texte == "3 pieds = 0,91 mètres."
    assert conversion("5 kg en km") is None  # pas la même grandeur


def test_calculs_poses():
    assert calcul("combien font 1 250 plus 3 780 ?").texte == "1 250 + 3 780 = 5030."
    assert calcul("Calcule 15 % de 80").texte == "15 % de 80 = 12."
    assert calcul("et 16 fois 18 ?").texte == "16 × 18 = 288."  # la relance d'un calcul
    assert calcul("J'ai 20 euros, j'achète 3 cahiers à 4 euros. Combien me reste-t-il ?") is None  # un problème : Carl


def test_minuteur_sans_le_lancer():
    r = minuteur("préviens-moi dans 10 minutes pour sortir le gâteau")
    assert r.texte == "C'est noté : je te préviens dans 10 minutes (sortir le gâteau). Laisse cette fenêtre ouverte."
    assert r.action is not None  # lancé par chat.py seulement
    assert minuteur("combien de minutes dans une heure ?") is None


def test_rien_pour_le_reste():
    for m in ["Quelle est la capitale de la France ?", "Raconte-moi une histoire", "C'est quoi un mile ?"]:
        assert repondre(m) is None, m
