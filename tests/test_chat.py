"""Règles de génération de chat.py : relances, anti-boucle, calculatrice."""

from chat import a_montrer, couper, dans_un_calcul, est_relance, trigrammes_interdits
from outils import afficher, calculer


def test_relances():
    assert est_relance("et de la France ?")
    assert est_relance("Et celle de l'Allemagne ?")
    assert est_relance("pourquoi ?")
    assert est_relance("en quelle année ?")          # courte : trois mots
    assert not est_relance("Quelle est la capitale du Japon ?")
    assert not est_relance("qui est tu ?")            # s'adresse à Carl : jamais de mémoire
    assert not est_relance("merci beaucoup !")        # politesse
    assert est_relance("et où ?")
    # Courtes, mais elles nomment leur sujet : des questions complètes.
    assert not est_relance("nantes pays ?")
    assert not est_relance("population lyon")
    assert not est_relance("capital du portigal ?")
    assert not est_relance("ou est nee marie curie")  # « ou » : « où » tapé vite
    for relance in ["sa population ?", "quel âge ?", "son âge ?", "et lui ?", "né où ?", "la capitale ?",
                    "depuis quand ?", "combien d'habitants ?", "où est-il né ?", "quand est-il mort ?",
                    "il est né quand ?", "qui l'a écrit ?", "ils parlent quelle langue ?", "Elle a quel âge ?"]:
        assert est_relance(relance), relance
    for autonome in ["qui a réalisé Elle ?", "Quelle heure est-il ?", "y a-t-il une capitale ?"]:
        assert not est_relance(autonome), autonome


def test_memoire_seulement_pour_une_relance():
    conv = [{"role": "user", "content": "Capitale de l'Espagne ?"},
            {"role": "assistant", "content": "Madrid."}]
    relance = conv + [{"role": "user", "content": "et de la France ?"}]
    nouvelle = conv + [{"role": "user", "content": "Quelle est la capitale du Japon ?"}]
    assert a_montrer(relance, None) == relance
    assert a_montrer(nouvelle, None) == nouvelle[-1:]
    assert a_montrer(nouvelle, 1) == nouvelle         # --memoire impose


def test_memoire_saute_les_politesses():
    conv = [{"role": "user", "content": "Quelle langue parle-t-on en Argentine ?"},
            {"role": "assistant", "content": "L'espagnol."},
            {"role": "user", "content": "merci !"},
            {"role": "assistant", "content": "Avec plaisir !"},
            {"role": "user", "content": "Et au Brésil ?"}]
    assert a_montrer(conv, None) == conv[:2] + conv[-1:]
    merci_bien = conv[:2] + [{"role": "user", "content": "cool merci bien !"}, conv[3], conv[4]]
    assert a_montrer(merci_bien, None) == conv[:2] + conv[-1:]
    sur = conv[:2] + [{"role": "user", "content": "tu es sûr ?"}, conv[3], conv[4]]
    assert a_montrer(sur, None) == conv[:2] + conv[-1:]
    salut = [{"role": "user", "content": "Hello Carl !"}, {"role": "assistant", "content": "Bonjour !"},
             {"role": "user", "content": "et de la France ?"}]
    assert a_montrer(salut, None) == salut[-1:]


def test_couper_a_la_derniere_phrase():
    assert couper("Victor Hugo est né à Besançon en 1802, et il a écrit des romans. Il a aussi écrit des poè") \
        == "Victor Hugo est né à Besançon en 1802, et il a écrit des romans."
    # La dernière phrase est finie : rien à couper.
    assert couper("Canberra est la capitale. Elle compte 450 000 habitants.") \
        == "Canberra est la capitale. Elle compte 450 000 habitants."
    # Presque tout serait perdu : on garde le texte, marqué comme coupé.
    assert couper("Oui. Il a écrit des romans, des poèmes, des pièces et des discours très longs sur") \
        .endswith("sur…")
    assert couper("Une seule phrase qui ne finit jam") == "Une seule phrase qui ne finit jam…"


def test_trigrammes_interdits():
    # « 1 2 3 ... 1 2 » : écrire 3 recréerait la suite « 1 2 3 ».
    assert trigrammes_interdits([1, 2, 3, 4, 1, 2]) == [3]
    assert trigrammes_interdits([1, 2, 3]) == []
    assert trigrammes_interdits([1, 2, 3, 1, 2], 0) == []


def test_calculatrice():
    assert calculer("17*23") == "391"
    assert calculer("10/3") == "3,3333"
    assert calculer("import os") == "erreur"
    assert calculer("1 200*3") == "3600"             # séparateur de milliers à la française
    assert calculer("7 ÷ 2") == "3,5"
    assert calculer("12 3") == "erreur"               # pas un groupe de milliers : on ne colle pas
    assert afficher("17 × 23 = [calc: 17*23 = 391].") == "17 × 23 = 391."
    # Pendant l'appel, l'anti-boucle est suspendu (Carl y recopie l'opération).
    assert dans_un_calcul("4827 + 3196 = [calc: 4827+3196")
    assert not dans_un_calcul("= [calc: 1+2 = 3]. Et")
