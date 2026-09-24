import json

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
    faits._base.cache_clear()


def test_chercher_tolere_article_accents_et_nom_partiel(tmp_path, monkeypatch):
    base(tmp_path, monkeypatch)
    assert faits.chercher("Espagne", "capitale") == "Madrid"
    assert faits.chercher("l'espagne", "Capitale") == "Madrid"
    assert faits.chercher("Espangne", "capitale") == "Madrid"
    assert faits.chercher("Hugo", "né") == "26 février 1802"
    assert faits.chercher("Canada", "langues") == "anglais et français"
    assert faits.chercher("Espagne", "monnaie") is None
    assert faits.chercher("Wakanda", "capitale") is None
    faits._base.cache_clear()


def test_appel_et_affichage():
    assert APPEL_FAIT.search("[fait: Espagne | capitale =").groups() == ("Espagne", "capitale")
    assert APPEL_FAIT.search("[fait: Espagne | capitale = Madrid]") is None
    assert afficher("[fait: Espagne | capitale = Madrid] La capitale de l'Espagne est Madrid.") == \
        "La capitale de l'Espagne est Madrid."
