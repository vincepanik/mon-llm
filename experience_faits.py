"""
Expérience : un modèle de 125M peut-il apprendre des faits par l'entraînement ?

Le test du perroquet (étape D) : des questions-réponses, Carl récite les
questions vues et rien d'autre. D'après « Physics of Language Models »
(Allen-Zhu et Li, 2023), un modèle ne sait ressortir un fait que s'il l'a lu
sous de nombreuses formes. Protocole, sur 500 faits de Wikidata (aucun de
l'examen) :

- groupe A (250 faits) : ~20 phrases variées par fait + quelques questions ;
- groupe B (250 faits) : les phrases seulement, jamais une question ;
- test : des questions formulées autrement qu'à l'entraînement, outils
  interdits (sinon Carl lirait la réponse dans sa base de faits).

Réussir B, c'est avoir appris le fait et savoir le ressortir. Ne réussir que
A, c'est réciter.

    python experience_faits.py --donnees        # data/sft/exp_*.json
    python sft.py --checkpoint checkpoints/carl_v10/best.pt --data ... --out checkpoints/carl_exp
    python experience_faits.py checkpoints/carl_v10/best.pt checkpoints/carl_exp/best.pt
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

import faits  # noqa: E402
from faits_sft import cite_dans_examen, formes, interdits_examen, maj, pays_en  # noqa: E402

R = random.Random(7)
SORTIE = Path("data/sft")
PAR_RELATION = 100
PHRASES_PAR_FAIT = 20

# Relation : (catégorie, phrases déclaratives, questions d'entraînement, questions de test).
# Les questions de test n'apparaissent jamais à l'entraînement, sous aucune forme.
REL = {
    "capitale": ("pays", [
        "La capitale {de_X} est {V}.", "{V} est la capitale {de_X}.", "{maj_le_X} a pour capitale {V}.",
        "{V}, capitale {de_X}, accueille le gouvernement du pays.", "Le gouvernement {de_X} siège à {V}.",
        "Parmi les villes {de_X}, {V} est la capitale.", "{V} est la ville capitale {de_X}.",
        "En visitant {le_X}, on peut découvrir {V}, sa capitale.", "La ville de {V} est la capitale {de_X}.",
        "{maj_le_X} : capitale, {V}."],
        ["Quelle est la capitale {de_X} ?", "C'est quoi la capitale {de_X} ?"],
        ["Comment s'appelle la capitale {de_X} ?", "{maj_le_X} a pour capitale quelle ville ?"]),
    "auteur": ("livre", [
        "{X} est une œuvre de {V}.", "{V} a écrit {X}.", "{X} a été écrit par {V}.", "On doit {X} à {V}.",
        "L'auteur de {X} est {V}.", "Parmi les livres de {V}, on trouve {X}.", "{X}, de {V}, est un livre connu.",
        "C'est {V} qui a écrit {X}.", "{X} est signé {V}.", "{V} est l'auteur de {X}."],
        ["Qui a écrit {X} ?", "Qui est l'auteur de {X} ?"],
        ["{X}, c'est un livre de qui ?", "Quel écrivain a signé {X} ?"]),
    "réalisateur": ("film", [
        "Le film {X} a été réalisé par {V}.", "{V} a réalisé {X}.", "{X} est un film de {V}.",
        "Le réalisateur de {X} est {V}.", "On doit le film {X} à {V}.", "Parmi les films de {V}, il y a {X}.",
        "C'est {V} qui a réalisé {X}.", "{X}, réalisé par {V}, est un film connu.", "{V} est le réalisateur de {X}.",
        "Derrière la caméra de {X}, on trouve {V}."],
        ["Qui a réalisé {X} ?", "Qui est le réalisateur du film {X} ?"],
        ["Le film {X}, c'est de quel réalisateur ?", "Quel cinéaste a tourné {X} ?"]),
    "naissance": ("personne", [
        "{X} est né{e} en {V}.", "{X} a vu le jour en {V}.", "L'année de naissance de {X} est {V}.",
        "En {V} naît {X}.", "{X}, né{e} en {V}, est une personnalité connue.", "{X} est venu{e} au monde en {V}.",
        "La naissance de {X} date de {V}.", "C'est en {V} que {X} est né{e}.", "{X} (né{e} en {V}) est célèbre.",
        "{V} est l'année de naissance de {X}."],
        ["En quelle année est né{e} {X} ?", "Quand est né{e} {X} ?"],
        ["{X} est né{e} en quelle année ?", "Quelle est l'année de naissance de {X} ?"]),
    "pays": ("ville", [
        "{X} se trouve {en_V}.", "{X} est une ville située {en_V}.", "La ville de {X} est {en_V}.",
        "{en_V_maj}, on trouve la ville de {X}.", "{X} fait partie des villes de ce pays : {V}.",
        "Si tu vas à {X}, tu seras {en_V}.", "{X} est située {en_V}.", "C'est {en_V} que se trouve {X}.",
        "Parmi les villes situées {en_V}, il y a {X}.", "{X} : une ville {de_V}."],
        ["Dans quel pays se trouve {X} ?", "{X} est dans quel pays ?"],
        ["Dans quel pays est située la ville de {X} ?", "{X}, ça se trouve dans quel pays ?"]),
}
DEMANDES = ["Dis-moi quelque chose sur {X}.", "Parle-moi de {X}.", "Un fait sur {X} ?", "Que sais-tu de {X} ?",
            "Raconte-moi quelque chose.", "Apprends-moi quelque chose.", "Un fait intéressant ?", "{X} ?"]


def choisir() -> list[dict]:
    entites, donnees, _ = faits._base()
    examen = interdits_examen()
    choisis = []
    for relation, (categorie, *_rest) in REL.items():
        candidats = [q for q, e in entites.items() if e["type"] == categorie and relation in donnees.get(q, {})
                     and not cite_dans_examen(e["nom"], examen) and len(e["nom"]) <= 40]
        candidats.sort(key=lambda q: -entites[q]["liens"])
        # Des entités connues (Carl a pu les croiser au pré-entraînement), mais pas seulement les plus célèbres.
        pris = R.sample(candidats[:1500], min(PAR_RELATION, len(candidats[:1500])))
        for qid in pris:
            e = entites[qid]
            v = faits.chercher(e["nom"], relation)
            if not v or " et " in v:
                continue
            if relation == "naissance":
                v = v.split()[-1]  # l'année seulement
                if not v.isdigit():
                    continue
            choisis.append({"qid": qid, "nom": e["nom"], "relation": relation, "valeur": v})
    R.shuffle(choisis)
    for i, f in enumerate(choisis):
        f["groupe"] = "A" if i % 2 == 0 else "B"
    return choisis


def gabarit(f: dict) -> dict:
    entites, _, _ = faits._base()
    g = formes(f["nom"], entites[f["qid"]])
    g["V"] = f["valeur"]
    if "le_X" in g:
        g["maj_le_X"] = maj(g["le_X"])
    if f["relation"] == "pays":
        g["en_V"] = pays_en(f["valeur"])
        g["en_V_maj"] = maj(g["en_V"])
        g["de_V"] = pays_en(f["valeur"]).replace("en ", "de ", 1) if g["en_V"].startswith("en ") else f"de {f['valeur']}"
    return g


def cle(f: dict) -> str:
    """Le mot qui doit figurer dans la réponse : « bratislava », « 1879 », « zola »."""
    return f["valeur"].split()[-1].lower()


def donnees() -> None:
    choisis = choisir()
    decl, qa, test = [], [], []
    for f in choisis:
        _, phrases, questions, tests = REL[f["relation"]]
        g = gabarit(f)
        for i in range(PHRASES_PAR_FAIT):
            phrase = phrases[i % len(phrases)].format(**g)
            decl.append([{"role": "user", "content": R.choice(DEMANDES).format(**g)},
                         {"role": "assistant", "content": maj(phrase)}])
        if f["groupe"] == "A":
            for q in questions:
                qa.append([{"role": "user", "content": maj(q.format(**g))},
                           {"role": "assistant", "content": maj(phrases[0].format(**g))}])
        for q in tests:
            test.append({"question": maj(q.format(**g)), "cle": cle(f), "groupe": f["groupe"],
                         "relation": f["relation"]})
    for nom, contenu in [("exp_phrases", decl), ("exp_questions", qa), ("exp_test", test)]:
        (SORTIE / f"{nom}.json").write_text(json.dumps(contenu, ensure_ascii=False), encoding="utf-8")
    print(f"{len(choisis)} faits : {len(decl)} phrases, {len(qa)} questions (groupe A), {len(test)} questions de test")
    for c in R.sample(decl, 5) + R.sample(qa, 2):
        print(f"   {c[0]['content']}  ->  {c[1]['content']}")
    for t in R.sample(test, 3):
        print(f"   test {t['groupe']} : {t['question']}  [{t['cle']}]")


def tester(chemins: list[str]) -> None:
    from chat import repondre
    from model import GPT
    from tokenizer import BPETokenizer
    from utils import get_device, load_checkpoint

    test = json.loads((SORTIE / "exp_test.json").read_text(encoding="utf-8"))
    device = get_device()
    tok = BPETokenizer.load("tokenizer/vocab.json")
    for chemin in chemins:
        ck = load_checkpoint(chemin, device)
        model = GPT(ck["config"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        scores: dict[tuple[str, str], list[int]] = {}
        for t in test:
            r = repondre(model, tok, [{"role": "user", "content": t["question"]}], device, temperature=0.0, top_k=1,
                         max_tokens=40, repetition_penalty=1.15, avec_outils=False).lower()
            for cle_score in [(t["groupe"], "tout"), (t["groupe"], t["relation"])]:
                scores.setdefault(cle_score, []).append(t["cle"] in r)
        print(f"\n##### {chemin}")
        for groupe in "AB":
            ok = scores[(groupe, "tout")]
            detail = "  ".join(f"{rel} {sum(scores[(groupe, rel)])}/{len(scores[(groupe, rel)])}" for rel in REL)
            print(f"  groupe {groupe} : {sum(ok)}/{len(ok)} ({sum(ok) / len(ok):.0%})   {detail}")


if __name__ == "__main__":
    if "--donnees" in sys.argv:
        donnees()
    else:
        tester([a for a in sys.argv[1:] if not a.startswith("--")] or ["checkpoints/carl_v10/best.pt"])
