"""Règles de génération de chat.py : relances, anti-boucle, calculatrice."""

from chat import a_montrer, dans_un_calcul, est_relance, trigrammes_interdits
from outils import afficher, calculer


def test_relances():
    assert est_relance("et de la France ?")
    assert est_relance("Et celle de l'Allemagne ?")
    assert est_relance("pourquoi ?")
    assert est_relance("en quelle année ?")          # courte : trois mots
    assert not est_relance("Quelle est la capitale du Japon ?")
    assert not est_relance("qui est tu ?")            # s'adresse à Carl : jamais de mémoire
    assert not est_relance("merci beaucoup !")        # politesse


def test_memoire_seulement_pour_une_relance():
    conv = [{"role": "user", "content": "Capitale de l'Espagne ?"},
            {"role": "assistant", "content": "Madrid."}]
    relance = conv + [{"role": "user", "content": "et de la France ?"}]
    nouvelle = conv + [{"role": "user", "content": "Quelle est la capitale du Japon ?"}]
    assert a_montrer(relance, None) == relance
    assert a_montrer(nouvelle, None) == nouvelle[-1:]
    assert a_montrer(nouvelle, 1) == nouvelle         # --memoire impose


def test_trigrammes_interdits():
    # « 1 2 3 ... 1 2 » : écrire 3 recréerait la suite « 1 2 3 ».
    assert trigrammes_interdits([1, 2, 3, 4, 1, 2]) == [3]
    assert trigrammes_interdits([1, 2, 3]) == []
    assert trigrammes_interdits([1, 2, 3, 1, 2], 0) == []


def test_calculatrice():
    assert calculer("17*23") == "391"
    assert calculer("10/3") == "3,3333"
    assert calculer("import os") == "erreur"
    assert afficher("17 × 23 = [calc: 17*23 = 391].") == "17 × 23 = 391."
    # Pendant l'appel, l'anti-boucle est suspendu (Carl y recopie l'opération).
    assert dans_un_calcul("4827 + 3196 = [calc: 4827+3196")
    assert not dans_un_calcul("= [calc: 1+2 = 3]. Et")
