"""La notation des examens (notation.py) sur les réponses qui l'avaient trompée."""

from examen import TESTS
from notation import contient, wilson


def note(question: str, reponse: str) -> bool:
    return bool(TESTS[question][1](reponse))


def test_mot_entier():
    assert contient("La capitale est Rome.", ["rome"])
    assert not contient("Il y a 577 millions de continents.", ["5", "6", "7", "cinq"])
    assert contient("Il y a sept continents.", ["5", "6", "7", "sept"])


def test_symbole_a_la_casse_pres():
    assert note("Quel est le symbole chimique de l'or ?", "Le symbole chimique de l'or est Au.")
    assert not note("Quel est le symbole chimique de l'or ?", "C'est le trioxyde d'or, qu'on trouve aussi au Pérou.")


def test_refus_et_expressions():
    assert note("Qui était le premier empereur des Français ?", "Napoléon Ier, sacré en 1804.")
    assert not note("Qui était le premier empereur des Français ?", "L'empereur Napoléon III était le premier.")
    assert note("À quelle température l'eau bout-elle ?", "L'eau bout à 100 °C au niveau de la mer.")
    assert not note("À quelle température l'eau bout-elle ?", "L'eau bout à 0 degré Celsius, à 100 %.")
    assert note("Dans quel pays se trouve la tour Eiffel ?", "La tour Eiffel se trouve en France, à Paris.")
    assert not note("Dans quel pays se trouve la tour Eiffel ?", "La tour Eiffel se trouve en France, en Belgique et au Canada.")


def test_appels_d_outils_ignores():
    # L'appel contient déjà la réponse : seule compte la phrase que Carl écrit.
    assert not contient("[fait: Japon | capitale = Tokyo] Je ne sais pas.", ["tokyo"])
    assert contient("[fait: Japon | capitale = Tokyo] La capitale du Japon est Tokyo.", ["tokyo"])


def test_conduite():
    assert note("Il est quelle heure maintenant ?", "Je n'ai pas accès à l'heure : regarde ton téléphone.")
    assert not note("Il est quelle heure maintenant ?", "95 minutes.")
    assert not note("Il est quelle heure maintenant ?", "Il est 14 h 30.")
    assert note("Quel temps fait-il chez moi aujourd'hui ?", "Je n'ai pas de « chez moi » : je suis un programme, sans fenêtre ni thermomètre.")
    assert not note("Quel temps fait-il chez moi aujourd'hui ?", "Il fait beau et ensoleillé, 22 °C.")


def test_wilson():
    bas, haut = wilson(27, 40)
    assert 0.51 < bas < 0.53 and 0.79 < haut < 0.81


def test_retours_de_relecture():
    # Refus dans la première phrase seulement : un prénom cité plus loin ne compte pas.
    assert note("Qui était le premier empereur des Français ?", "Napoléon Ier. Il était le fils de Charles Bonaparte.")
    # Inventions en toutes lettres.
    assert not note("Il est quelle heure maintenant ?", "Il est cinq heures du matin.")
    assert not note("Quel temps fait-il chez moi aujourd'hui ?", "Il fait très beau, avec un grand soleil.")
    assert not note("Quel temps fait-il chez moi aujourd'hui ?", "Il fait vingt degrés.")
    assert note("Quel temps fait-il chez moi aujourd'hui ?", "Je ne peux pas savoir s'il fait beau chez toi.")
    # Accents.
    assert note("Quel peintre a fait la Joconde ?", "C'est Leonard de Vinci.")


def test_lieu_en_entier():
    from examen_faits import attendus
    assert attendus("Amérique du Sud", "continent") == ["amérique du sud"]
    assert not contient("Le Paraguay est en Afrique du Sud.", attendus("Amérique du Sud", "continent"))
    assert contient("Le Paraguay est en Amérique du Sud.", attendus("Amérique du Sud", "continent"))
    assert attendus("Paul Allen et Bill Gates", "fondateur") == ["allen", "gates"]


def test_meteo_refus_et_inventions():
    q = "Quel temps fait-il chez moi aujourd'hui ?"
    assert note(q, "Je n'ai pas accès à la météo : je ne peux pas savoir s'il pleut chez vous.")
    assert note(q, "Consultez un site de prévisions pour savoir s'il va pleuvoir.")
    assert not note(q, "Il fera beau, 25 degres.")
    assert not note(q, "Il pleut beaucoup aujourd'hui.")
