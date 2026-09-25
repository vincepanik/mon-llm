import json
from pathlib import Path

import pytest

import faits
from outils import APPEL_FAIT, afficher


def base(tmp_path, monkeypatch):
    donnees = {
        "entites": {"Q29": {"nom": "Espagne", "type": "pays", "liens": 300},
                    "Q535": {"nom": "Victor Hugo", "type": "personne", "liens": 200},
                    "Q16": {"nom": "Canada", "type": "pays", "liens": 300}},
        "faits": {"Q29": {"capitale": ["Madrid"]}, "Q535": {"naissance": ["26 février 1802"]},
                  "Q16": {"langue": ["anglais", "français"]}},
    }
    chemin = tmp_path / "faits.json"
    chemin.write_text(json.dumps(donnees), encoding="utf-8")
    monkeypatch.setattr(faits, "CHEMIN", chemin)
    faits.vider_cache()


def test_chercher_tolere_article_accents_et_nom_partiel(tmp_path, monkeypatch):
    base(tmp_path, monkeypatch)
    assert faits.chercher("Espagne", "capitale") == "Madrid"
    assert faits.chercher("l'espagne", "Capitale") == "Madrid"
    assert faits.chercher("Espangne", "capitale") == "Madrid"
    assert faits.chercher("Hugo", "né") == "26 février 1802"
    assert faits.chercher("Canada", "langues") == "anglais et français"
    assert faits.chercher("Espagne", "monnaie") is None
    assert faits.chercher("Wakanda", "capitale") is None
    faits.vider_cache()


def test_appel_et_affichage():
    assert APPEL_FAIT.search("[fait: Espagne | capitale =").groups() == ("Espagne", "capitale")
    assert APPEL_FAIT.search("[fait: Espagne | capitale = Madrid]") is None
    assert afficher("[fait: Espagne | capitale = Madrid] La capitale de l'Espagne est Madrid.") == \
        "La capitale de l'Espagne est Madrid."


VRAIE_BASE = pytest.mark.skipif(not Path("data/big/wikidata/faits.json").exists(), reason="base Wikidata absente")


@VRAIE_BASE
def test_vraie_base_retrouve_malgre_les_fautes():
    faits.vider_cache()
    attendu = {
        ("Espagne", "capitale"): "Madrid", ("l'espagne", "capitale"): "Madrid", ("portugal", "capital"): "Lisbonne",
        ("Hugo", "naissance"): "26 février 1802", ("Napoléon", "naissance"): "15 août 1769",
        ("mont Everest", "altitude"): "8 849", ("ville de Lyon", "pays"): "France",
        ("nietzche", "décès"): "25 août 1900", ("slovaki", "capitale"): "Bratislava",
        ("romeul lukaku", "naissance"): "13 mai 1993",
        # Noms de famille qui sont aussi des villes.
        ("Lincoln", "naissance"): "12 février 1809", ("Darwin", "naissance"): "12 février 1809",
        ("Candide", "date"): "1759",  # « Candide, ou l'Optimisme »
        # Relevés par la relecture : chacun renvoyait une autre personne ou rien.
        ("Louis IX", "naissance"): "2 mai 1214",
        ("Catherine II", "naissance"): "2 mai 1729", ("Stalingrad", "fin"): "2 février 1943",
        ("Chine", "capitale"): "Pékin", ("Pays-Bas", "capitale"): "Amsterdam", ("Van Gogh", "décès"): "29 juillet 1890",
        ("Wolfgang Mozart", "naissance"): "27 janvier 1756", ("Ventoux", "altitude"): "1 910",
        ("Verdun", "début"): "21 février 1916", ("guerre des 6 jours", "fin"): "10 juin 1967",
        ("Cote divoire", "capitale"): "Yamoussoukro", ("Kennedy", "naissance"): "29 mai 1917",
        # Grâce aux autres noms officiels (alias Wikidata).
        ("Napoléon Bonaparte", "naissance"): "15 août 1769", ("Mona Lisa", "auteur"): "Léonard de Vinci",
        # Deuxième relecture.
        ("E.T.", "réalisateur"): "Steven Spielberg", ("Ulm", "date"): "20 octobre 1805",
        ("Jean-Sébastien Bcah", "décès"): "28 juillet 1750", ("CharlieChaplin", "naissance"): "16 avril 1889",
        ("Jaon", "population"): "123 802 000", ("Gana", "capitale"): "Accra", ("Gerld Ford", "naissance"): "14 juillet 1913",
        ("Hollande", "naissance"): "12 août 1954", ("Seconde Guerre mondiale", "date"): "1er septembre 1939",
        # Troisième relecture.
        ("Waterloo", "date"): "18 juin 1815", ("Montaigne", "naissance"): "10 mars 1533",
        ("Dante", "naissance"): "1265", ("Karl Merx", "naissance"): "5 mai 1818", ("New Yirk", "population"): "8 804 190",
        ("Napoléon 1er", "naissance"): "15 août 1769", ("1ère guerre mondiale", "début"): "28 juillet 1914",
    }
    for (entite, relation), valeur in attendu.items():
        assert faits.chercher(entite, relation) == valeur, (entite, relation)


@VRAIE_BASE
def test_vraie_base_se_tait_plutot_que_d_inventer():
    faits.vider_cache()
    # Chacun donnait une réponse fausse, que Carl recopiait avec l'autorité de l'outil.
    for entite, relation in [("Kevin Pacini", "naissance"), ("Lune", "date"), ("Apollo 11", "date"),
                             ("Carl", "naissance"), ("Carl", "fondateur"), ("Portugal", "naissance"),
                             ("Hamlet", "auteur"), ("Platon", "naissance"), ("Homère", "naissance"),
                             ("Halliday", "décès"),
                             # Deuxième relecture : chacun inventait.
                             ("Paris", "naissance"), ("Jésus", "naissance"), ("Wakanda", "capitale"),
                             ("Prusse", "capitale"), ("Groenland", "capitale"), ("Paris", "auteur"),
                             ("La Reine", "naissance"), ("ma mère", "naissance"), ("le général", "naissance"),
                             ("Jupiter", "naissance"), ("The King", "naissance"), ("Louis", "naissance"),
                             ("Pierre Macron", "naissance"), ("Paul Einstein", "naissance"), ("Révolution", "début"),
                             ("guerre", "fin"), ("nord", "capitale"), ("guerre de Troie", "date"),
                             # Troisième relecture.
                             ("Léa Martin", "naissance"), ("Dordogne", "population"), ("Spotify", "fondateur"),
                             ("Hiroshima", "date"), ("Henri", "naissance"), ("Audten", "naissance"),
                             # Ambigus : la base ne choisit pas au hasard de la notoriété.
                             ("Clinton", "naissance"), ("François Ier", "naissance"), ("Cinton", "naissance")]:
        assert faits.chercher(entite, relation) is None, (entite, relation)
