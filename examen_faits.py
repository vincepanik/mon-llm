"""
Questions de faits pour mesurer la base de faits, en trois jeux :

- QUESTIONS : 15 faits bien écrits, absents des questions d'entraînement ;
- QUESTIONS_VITE : les mêmes « tapés vite » (fautes, sans accents,
  télégraphique). Ces deux jeux ont servi à régler Carl v11 et la recherche
  floue (faits.py) : ce sont des jeux de RÉGLAGE, leurs scores sont optimistes ;
- QUESTIONS_TEST (examen_faits_test.py) : 37 questions tapées vite écrites
  par un agent qui n'avait pas vu les gabarits d'entraînement, sur des faits
  jamais appris, et jamais utilisées pour régler quoi que ce soit. C'est le
  vrai chiffre. (13 autres, dont le fait a été vu bien écrit à
  l'entraînement, sont comptées à part.)

    python examen_faits.py checkpoints/carl_v10/best.pt checkpoints/carl_v11/best.pt
    python examen_faits.py -q checkpoints/carl_v11/best.pt   # seulement les ratés

Les bonnes réponses sont figées dans examen_faits_test.py (VERITES) ; un
modèle qui répond de mémoire doit les trouver seul. « Madame Bovary » et « Guernica »
figuraient mot pour mot dans les conversations de l'étape D : remplacés par
« Le Horla » et « Le Radeau de la Méduse » (tests/test_fuite_examen.py).
"""
import re
import sys

from examen_faits_test import VERITES
from notation import contient

QUESTIONS = [
    ("Quelle est la capitale de la Slovaquie ?", "Slovaquie", "capitale"),
    ("Quand est né Albert Einstein ?", "Albert Einstein", "naissance"),
    ("Qui a réalisé Titanic ?", "Titanic", "réalisateur"),
    ("Quelle est l'altitude du mont Blanc ?", "mont Blanc", "altitude"),
    ("Où est née Marie Curie ?", "Marie Curie", "lieu de naissance"),
    ("Qui a fondé Microsoft ?", "Microsoft", "fondateur"),
    ("Quel est le symbole chimique de l'oxygène ?", "oxygène", "symbole"),
    ("Qui a écrit Le Horla ?", "Le Horla", "auteur"),
    ("Qui a peint Le Radeau de la Méduse ?", "Le Radeau de la Méduse", "auteur"),
    ("Combien d'habitants compte Toulouse ?", "Toulouse", "population"),
    ("Quand est mort Molière ?", "Molière", "décès"),
    ("Quand est née Beyoncé ?", "Beyoncé", "naissance"),
    ("Où se trouve le siège d'Amazon ?", "Amazon", "siège"),
    ("En quelle année est mort Nietzsche ?", "Nietzsche", "décès"),
    ("Dans quel pays se trouve Nantes ?", "Nantes", "pays"),
]

# Les mêmes faits, « tapés vite » : fautes, sans accents, style télégraphique.
QUESTIONS_VITE = [
    ("capital du portigal ?", "Portugal", "capitale"),
    ("quand est ne einstein", "Albert Einstein", "naissance"),
    ("realisateur titanic ?", "Titanic", "réalisateur"),
    ("c koi la capitale de la slovaki", "Slovaquie", "capitale"),
    ("population toulouse", "Toulouse", "population"),
    ("ou est nee marie curie", "Marie Curie", "lieu de naissance"),
    ("fondateur microsfot", "Microsoft", "fondateur"),
    ("symbole chimique oxygene", "oxygène", "symbole"),
    ("moliere mort quand", "Molière", "décès"),
    ("beyonce née quand ?", "Beyoncé", "naissance"),
    ("siege d'amazon", "Amazon", "siège"),
    ("nietzche mort en quelle année", "Nietzsche", "décès"),
    ("nantes pays ?", "Nantes", "pays"),
    ("altitude du mon blanc", "mont Blanc", "altitude"),
    ("qui a peint le radeau de la meduse", "Le Radeau de la Méduse", "auteur"),
]


MOIS = "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split()


PERSONNES = {"auteur", "compositeur", "réalisateur", "fondateur"}


def attendus(valeur: str, relation: str) -> list[str]:
    """
    Ce qui doit figurer dans la réponse, en mot entier : le nom de famille
    d'une personne (« cameron »), l'année d'une date (« 1879 »), le nombre
    entier (« 514 819 »), le symbole à la casse près (« O », pas la lettre « o »
    de n'importe quel mot), et un lieu en entier (« amérique du sud » : « sud »
    seul accepterait « Afrique du Sud »). Plusieurs valeurs (« Paul Allen et
    Bill Gates ») : une seule suffit.
    """
    mots = []
    for partie in re.split(r", | et ", valeur):
        partie = partie.strip()
        if re.fullmatch(r"[\d ]+", partie):
            mots.append(partie)
        elif re.search(r"\b\d{3,4}\b", partie) and (any(m in partie for m in MOIS) or partie[:1].isdigit()):
            mots.append(re.findall(r"\d{3,4}", partie)[-1])
        elif len(partie) <= 3:
            mots.append(partie)
        elif relation in PERSONNES:
            mots.append(partie.split()[-1].lower())
        else:
            mots.append(partie.lower())
    return mots


def jeux(tous: bool = True) -> list[tuple[str, list]]:
    """Les jeux de questions ; tous=False écarte celui dont les faits ont été vus à l'entraînement."""
    from examen_faits_test import QUESTIONS_TEST, QUESTIONS_TEST_DEJA_VUES

    liste = [("bien écrites (réglage)", QUESTIONS), ("tapées vite (réglage)", QUESTIONS_VITE),
             ("tapées vite (TEST, faits jamais vus)", QUESTIONS_TEST)]
    if tous:
        liste.append(("tapées vite (test, faits déjà vus bien écrits)", QUESTIONS_TEST_DEJA_VUES))
    return liste


def main() -> None:
    from chat import repondre, repondre_aiguille
    from model import GPT
    from outils import afficher
    from tokenizer import BPETokenizer
    from utils import get_device, load_checkpoint

    bavard = "-q" not in sys.argv
    device = get_device()
    tok = BPETokenizer.load("tokenizer/vocab.json")
    for chemin in [a for a in sys.argv[1:] if not a.startswith("-")] or ["checkpoints/carl_v11/best.pt"]:
        ck = load_checkpoint(chemin, device)
        model = GPT(ck["config"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        print(f"\n##### {chemin}")
        for titre, questions in jeux():
            ok = 0
            for q, e, r in questions:
                verite = VERITES[(e, r)]
                reglages = dict(temperature=0.0, top_k=1, max_tokens=80, repetition_penalty=1.15)
                if "--aiguilleur" in sys.argv:
                    brut, _ = repondre_aiguille(model, tok, [{"role": "user", "content": q}], device, **reglages)
                else:
                    brut = repondre(model, tok, [{"role": "user", "content": q}], device, brut=True, **reglages)
                bon = contient(brut, attendus(verite, r))
                ok += bon
                if bavard or not bon:
                    print(f"  {'✓' if bon else '✗'} {q}\n      -> {afficher(brut).strip()[:110]}"
                          f"\n      (attendu : {verite} ; brut : {brut[:70]!r})")
            print(f"  == {titre} : {ok}/{len(questions)}")


if __name__ == "__main__":
    main()
