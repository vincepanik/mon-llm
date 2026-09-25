"""
Quinze questions de faits jamais vues à l'entraînement (ni dans
data/sft/faits.json, ni dans l'examen), pour mesurer la base de faits ; puis
les mêmes, « tapées vite » (fautes, sans accents, style télégraphique).

    python examen_faits.py checkpoints/carl_v8/best.pt checkpoints/carl_v10/best.pt
    python examen_faits.py -q checkpoints/carl_v11/best.pt   # seulement les ratés

La bonne réponse vient de la base elle-même (faits.chercher) ; un modèle qui
répond de mémoire doit la trouver seul.
"""
import sys

import faits
from chat import repondre
from outils import afficher
from model import GPT
from tokenizer import BPETokenizer
from utils import get_device, load_checkpoint

QUESTIONS = [
    ("Quelle est la capitale de la Slovaquie ?", "Slovaquie", "capitale"),
    ("Quand est né Albert Einstein ?", "Albert Einstein", "naissance"),
    ("Qui a réalisé Titanic ?", "Titanic", "réalisateur"),
    ("Quelle est l'altitude du mont Blanc ?", "mont Blanc", "altitude"),
    ("Où est née Marie Curie ?", "Marie Curie", "lieu de naissance"),
    ("Qui a fondé Microsoft ?", "Microsoft", "fondateur"),
    ("Quel est le symbole chimique de l'oxygène ?", "oxygène", "symbole"),
    ("Qui a écrit Madame Bovary ?", "Madame Bovary", "auteur"),
    ("Qui a peint Guernica ?", "Guernica", "auteur"),
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
    ("qui a peint guernica", "Guernica", "auteur"),
]


def main() -> None:
    bavard = "-q" not in sys.argv
    device = get_device()
    tok = BPETokenizer.load("tokenizer/vocab.json")
    for chemin in [a for a in sys.argv[1:] if not a.startswith("-")] or ["checkpoints/carl_v10/best.pt"]:
        ck = load_checkpoint(chemin, device)
        model = GPT(ck["config"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        print(f"\n##### {chemin}")
        for titre, questions in [("bien écrites", QUESTIONS), ("tapées vite", QUESTIONS_VITE)]:
            ok = 0
            for q, e, r in questions:
                verite = faits.chercher(e, r)
                cle = verite.split(" et ")[0].split(",")[0].split()[-1].lower()  # « 1879 », « cameron », « o »
                brut = repondre(model, tok, [{"role": "user", "content": q}], device, temperature=0.0, top_k=1,
                                max_tokens=80, repetition_penalty=1.15, brut=True)
                affiche = afficher(brut).strip()
                bon = cle in affiche.lower().replace(".", " ").split() or cle in affiche.lower()
                ok += bon
                if bavard or not bon:
                    print(f"  {'✓' if bon else '✗'} {q}\n      -> {affiche[:110]}\n      (attendu : {verite} ; brut : {brut[:70]!r})")
            print(f"  == {titre} : {ok}/{len(questions)}")


if __name__ == "__main__":
    main()
