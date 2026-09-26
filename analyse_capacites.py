"""
Analyse des capacités de Carl, domaine par domaine : ce qu'il sait faire, ce
qu'il ne sait pas faire. Complète les examens (qui mesurent chacun une chose)
par une vue d'ensemble, sur des demandes variées et jamais vues.

    python analyse_capacites.py checkpoints/carl_v15/best.pt            # Carl et l'aiguilleur
    python analyse_capacites.py checkpoints/carl_v15/best.pt --seul     # Carl seul

Les réponses sont gardées dans resultats/analyse_<version>.json. Là où c'est
possible, la notation est automatique (mots attendus) ; les réponses libres
(histoires, conseils, explications) sont marquées « à lire » et jugées à la main.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from notation import contient

# domaine -> [(demande, mots attendus ou None si à lire, mots refusés)]
BATTERIE = {
    "politesse": [
        ("Coucou Carl, tu vas bien ?", ["coucou", "salut", "bonjour", "ça va", "va bien", "tout fonctionne"], []),
        ("Bonne soirée à toi !", ["soirée", "bientôt", "revoir", "merci"], ["je suis carl"]),
        ("Merci, tu m'as bien aidé", ["plaisir", "rien", "content", "ravi"], ["je suis carl"]),
        ("Je dois filer, à la prochaine", ["bientôt", "revoir", "prochaine", "plus tard"], []),
    ],
    "identité": [
        ("Tu es qui exactement ?", ["carl"], []),
        ("Mais au fait, qui t'a construit ?", ["kevin"], []),
        ("T'es une version de ChatGPT ?", ["carl"], ["oui, je suis chatgpt"]),
        ("Tu es capable de faire quoi au juste ?", ["questions", "répondre", "calcul", "faits"], []),
    ],
    "faits (base)": [
        ("La capitale du Vietnam, c'est laquelle ?", ["hanoï", "hanoi"], []),
        ("Germinal, c'est un roman de qui ?", ["zola"], []),
        ("Quand est née Édith Piaf ?", ["1915"], []),
        ("Quel est le symbole chimique du fer ?", ["Fe"], []),
        ("Combien d'habitants à Marseille ?", ["re:\\d{3} \\d{3}"], []),
        ("Qui a réalisé Le Parrain ?", ["coppola"], []),
    ],
    "faits tapés vite": [
        ("capital de l'argentine", ["buenos"], []),
        ("qui a ecrit le petit prince", ["exupéry", "exupery"], []),
        ("einstien né quand", ["1879"], []),
        ("pop de lyon ?", ["re:\\d{3} \\d{3}"], []),
    ],
    "qui est": [
        ("Qui était Victor Schœlcher ?", ["homme politique", "abolition", "député", "journaliste"], []),
        ("c'est qui Serena Williams", ["tennis", "joueuse"], []),
        ("Qui est Hayao Miyazaki ?", ["réalisateur", "animat", "mangaka", "cinéaste"], []),
    ],
    "calcul": [
        ("Combien font 48 fois 25 ?", ["1200", "1 200"], []),
        ("Calcule 15 % de 80", ["12"], []),
        ("Combien font 1 250 plus 3 780 ?", ["5030", "5 030"], []),
        ("J'ai 20 euros, j'achète 3 cahiers à 4 euros. Combien me reste-t-il ?", ["8"], []),
    ],
    "temps réel": [
        ("Tu sais l'heure qu'il est ?", ["horloge", "pas accès", "ne peux pas", "ne sais pas"], []),
        ("C'est quel jour, aujourd'hui ?", ["horloge", "pas accès", "ne peux pas", "ne sais pas"], []),
        ("Tu crois qu'il pleuvra demain ?", ["pas accès", "ne peux pas", "météo", "internet"], []),
    ],
    "listes": [
        ("Cite-moi 4 capitales d'Afrique", ["re:\\(.*\\).*\\(.*\\).*\\(.*\\)"], []),
        ("Donne-moi trois pays d'Asie", ["chine", "inde", "japon", "indonésie", "corée", "vietnam", "thaïlande"], []),
        ("Tu peux me citer trois fleuves français ?", ["loire", "seine", "rhône", "garonne"], []),
    ],
    "culture générale (hors base)": [
        ("Un insecte, ça a combien de pattes ?", ["six", "6"], []),
        ("Quelle est la planète la plus grosse ?", ["jupiter"], []),
        ("Quel est l'animal le plus rapide ?", ["guépard", "faucon"], []),
        ("De quelle couleur est une banane mûre ?", ["jaune"], []),
        ("Combien y a-t-il de mois dans une année ?", ["douze", "12"], []),
    ],
    "explications": [
        ("Pourquoi la mer est salée ?", None, []),
        ("C'est quoi la gravité ?", None, []),
        ("Comment fonctionne un arc-en-ciel ?", None, []),
    ],
    "conseils": [
        ("C'est quoi la recette des crêpes ?", None, []),
        ("Tu as un conseil pour mieux se concentrer ?", None, []),
        ("Comment préparer un entretien d'embauche ?", None, []),
    ],
    "créativité": [
        ("Écris un poème de quatre vers sur l'automne", None, []),
        ("Raconte-moi une courte histoire de dragon", None, []),
        ("Tu connais une devinette ?", None, []),
    ],
    "raisonnement": [
        ("Un chat peut-il voler ?", ["non"], ["oui, un chat peut voler"]),
        ("Si j'ai 5 pommes et que j'en mange 2, combien m'en reste-t-il ?", ["3", "trois"], []),
        ("Paul est plus grand que Marie. Qui est le plus petit ?", ["marie"], []),
        ("Le feu est-il chaud ou froid ?", ["chaud"], []),
    ],
    "langue française": [
        ("Au pluriel, « cheval » devient quoi ?", ["chevaux"], []),
        ("Donne un synonyme de « rapide »", ["vite", "véloce", "prompt", "vif", "rapidement", "express"], []),
        ("Comment dit-on « dog » en français ?", ["chien"], []),
        ("Conjugue « manger » au futur avec « je »", ["mangerai"], []),
    ],
    "limites connues": [
        ("Écris une fonction Python qui additionne deux nombres", None, []),
        ("Résume-moi le dernier match du PSG", ["pas accès", "ne peux pas", "internet", "ne sais pas"], []),
    ],
}


def main() -> None:
    from chat import repondre, repondre_aiguille
    from model import GPT
    from outils import afficher
    from tokenizer import BPETokenizer
    from utils import get_device, load_checkpoint

    seul = "--seul" in sys.argv
    chemin = next((a for a in sys.argv[1:] if not a.startswith("-")), "checkpoints/carl_v15/best.pt")
    device = get_device()
    tok = BPETokenizer.load("tokenizer/vocab.json")
    ck = load_checkpoint(chemin, device)
    model = GPT(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    reglages = dict(temperature=0.0, top_k=1, max_tokens=150, repetition_penalty=1.15)
    resultats = []
    print(f"##### {chemin}" + (" (seul)" if seul else " (avec l'aiguilleur)"))
    for domaine, demandes in BATTERIE.items():
        notes = []
        for q, mots, refus in demandes:
            msg = [{"role": "user", "content": q}]
            if seul:
                brut, source = repondre(model, tok, msg, device, brut=True, **reglages), "carl"
            else:
                brut, source = repondre_aiguille(model, tok, msg, device, **reglages)
            reponse = afficher(brut).strip()
            note = None if mots is None else contient(brut, mots, refus)
            notes.append(note)
            resultats.append({"domaine": domaine, "demande": q, "reponse": reponse, "source": source, "juste": note})
            marque = "·" if note is None else ("✓" if note else "✗")
            print(f"  {marque} [{domaine}] {q}\n      -> {reponse[:160]!r}  ({source})")
        notees = [n for n in notes if n is not None]
        if notees:
            print(f"  == {domaine} : {sum(notees)}/{len(notees)}")
    sortie = Path("resultats") / f"analyse_{Path(chemin).parent.name}{'_seul' if seul else ''}.json"
    sortie.parent.mkdir(exist_ok=True)
    sortie.write_text(json.dumps(resultats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"(réponses gardées dans {sortie})")


if __name__ == "__main__":
    main()
