"""
Les évidences du quotidien : un chien a quatre pattes, le chat miaule, une
semaine a sept jours. Ce qu'un enfant de trois ans sait, et que Carl ne sait
pas : personne ne l'écrit, parce que tout le monde le sait (le biais de
l'évidence). Carl répondait « un insecte a 15 pattes ».

La recette de l'expérience des faits (groupe B, 96 %) : chaque fait écrit
sous 20 formes, en phrases seulement, jamais en question. Les questions
directes sans outil lui avaient appris à sauter sa base de faits (savoirs
27 -> 22) ; les phrases, non. Les faits sont dans data/claude/lot_evidences.txt
(écrits par Claude, gardés en local).

    python data/evidences.py        # écrit data/sft/evidences.json

L'examen à l'aveugle (examen_evidences.py) pose des questions que les
données ne contiennent jamais, sous aucune forme.
"""

from __future__ import annotations

import json
import random
import unicodedata
from pathlib import Path

RACINE = Path(__file__).resolve().parent
LOT = RACINE / "claude" / "lot_evidences.txt"
SORTIE = RACINE / "sft" / "evidences.json"
PHRASES_PAR_FAIT = 20
R = random.Random(21)

MOTS = {"0": "zéro", "1": "un", "2": "deux", "3": "trois", "4": "quatre", "5": "cinq", "6": "six", "7": "sept",
        "8": "huit", "10": "dix", "11": "onze", "12": "douze", "24": "vingt-quatre", "26": "vingt-six",
        "28": "vingt-huit", "60": "soixante", "100": "cent"}
# Mots en « l' » qui sont féminins (pour « il » ou « elle », « un » ou « une »).
FEMININS = {"abeille", "autruche", "herbe", "orange", "eau pure"}
VOYELLES = "aeéèêiîoôuûhœy"


def maj(s: str) -> str:
    return s[:1].upper() + s[1:]


def nom(gn: str) -> str:
    """« le chien » -> « chien », « l'âne » -> « âne »."""
    for a in ("le ", "la ", "les ", "un ", "une ", "des ", "l'"):
        if gn.startswith(a):
            return gn[len(a):]
    return gn


def feminin(gn: str) -> bool:
    return gn.startswith(("la ", "une ")) or (gn.startswith("l'") and nom(gn) in FEMININS)


def pluriel(gn: str) -> bool:
    return gn.startswith(("les ", "des "))


def de(gn: str) -> str:
    """« le chien » -> « du chien », « les yeux » -> « des yeux », « un vélo » -> « d'un vélo »."""
    if gn.startswith("le "):
        return "du " + gn[3:]
    if gn.startswith("les "):
        return "des " + gn[4:]
    if gn.startswith("des "):
        return "de " + gn[4:]
    if gn.startswith(("la ", "l'")):
        return "de " + gn
    if gn.startswith(("un ", "une ")):
        return "d'" + gn
    return ("d'" if gn[:1].lower() in VOYELLES else "de ") + gn


def a(gn: str) -> str:
    if gn.startswith("le "):
        return "au " + gn[3:]
    if gn.startswith("les "):
        return "aux " + gn[4:]
    return "à " + gn


def un(gn: str) -> str:
    """« le chien » -> « un chien », « l'abeille » -> « une abeille »."""
    if gn.startswith(("un ", "une ", "des ")):
        return gn
    if gn.startswith("les "):
        return "des " + gn[4:]
    return ("une " if feminin(gn) else "un ") + nom(gn)


def pronom(gn: str) -> str:
    return ("elles" if feminin(gn) else "ils") if pluriel(gn) else ("elle" if feminin(gn) else "il")


def que(s: str) -> str:
    return ("qu'" if s[:1].lower() in VOYELLES else "que ") + s


def en_saison(s: str) -> str:
    return "au printemps" if s == "le printemps" else "en " + nom(s)


def pluriel_nom(n: str) -> str:
    return n + ("x" if n.endswith("eau") else "" if n.endswith(("s", "x")) else "s")


def normaliser(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower().replace("œ", "oe"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def juste(reponse: str, cles: list[str]) -> bool:
    """Un mot de la réponse commence par une clé (« soign » -> « soigne ») ; un nombre doit être exact."""
    mots = [m for m in "".join(c if c.isalnum() else " " for c in normaliser(reponse)).split()]
    for cle in cles:
        c = normaliser(cle)
        if " " in c:
            if c in " ".join(mots):
                return True
        elif any(m == c if c.isdigit() else m.startswith(c) for m in mots):
            return True
    return False


def lire() -> list[dict]:
    faits = []
    for ligne in LOT.read_text(encoding="utf-8").splitlines():
        if not ligne.strip() or ligne.startswith("#"):
            continue
        rel, *ch = [c.strip() for c in ligne.split(";")]
        f = {"relation": rel}
        if rel == "pattes":
            x, u, n = ch
            f.update(X=x, sujet=x, un_X=u, N=n, mot=MOTS[n], cles=[n, MOTS[n]] + (["aucune", "pas de"] if n == "0" else []))
        elif rel == "cri":
            f.update(X=ch[0], sujet=ch[0], V=ch[1], cles=[ch[1][:4]])
        elif rel == "petit":
            f.update(X=ch[0], sujet=ch[0], P=ch[1], cles=[nom(ch[1])])
        elif rel == "famille":
            f.update(X=ch[0], sujet=ch[0], F=ch[1], cles=ch[2].split("|"))
        elif rel == "milieu":
            f.update(X=ch[0], sujet=ch[0], L=ch[1], cles=ch[2].split("|"))
        elif rel == "couleur":
            x, c = ch[0], ch[1]
            cles = ch[2].split("|") if len(ch) > 2 else [c]
            base = ch[2] if len(ch) > 2 and "|" not in ch[2] else c
            f.update(X=x, sujet=x, C=c, K=base, cles=cles)
        elif rel == "nombre":
            u, c, n, mot = ch
            f.update(U=u, X=c, sujet=c, N=n, mot=mot, cles=[n, mot])
        elif rel in ("sens", "outil"):
            f.update(A=ch[0], O=ch[1], sujet=ch[1], cles=ch[2].split("|"))
        elif rel == "propriete":
            f.update(X=ch[0], sujet=ch[0], P=ch[1], cles=ch[2].split("|"))
        elif rel == "contraire":
            b = ch[1].split("|")
            f.update(A=ch[0], sujet=ch[0], B=b[0], cles=b)
        elif rel == "origine":
            f.update(X=ch[0], sujet=ch[0], S=ch[1], cles=ch[2].split("|"))
        elif rel == "metier":
            f.update(X=ch[0], sujet=ch[0], V=ch[1], cles=ch[2].split("|"))
        elif rel == "saison":
            f.update(X=ch[0], sujet=ch[0], V=ch[1], cles=ch[2].split("|"))
        else:
            raise ValueError(f"relation inconnue : {ligne}")
        faits.append(f)
    return faits


def gabarit(f: dict) -> dict:
    g = dict(f)
    for cle in ("X", "O", "S", "A", "P", "L", "U"):
        if cle in f:
            v = f[cle]
            g[cle + "_maj"] = maj(v)
            g["de_" + cle] = de(v)
            g["a_" + cle] = a(v)
            g["un_" + cle] = un(v)
            g["Un_" + cle] = maj(un(v))
            g["il_" + cle] = pronom(v)
            g["nom_" + cle] = nom(v)
    if "un_X" in f:
        g["un_X"] = f["un_X"]
        g["Un_X"] = maj(f["un_X"])
        g["d_un_X"] = de(f["un_X"])
    if "K" in f:
        el = f["K"][:1] in VOYELLES
        g["le_K"] = ("l'" if el else "le ") + f["K"]
        g["le_K_maj"] = maj(g["le_K"])
        g["du_K"] = ("de l'" if el else "du ") + f["K"]
    if "X" in f:
        g["que_X"] = que(f["X"])
    if "N" in f:
        g["mot_maj"] = maj(f["mot"])
    if f["relation"] in ("sens", "outil"):
        pl = pluriel(f["O"])
        g["sert"], g["permet"] = ("servent", "permettent") if pl else ("sert", "permet")
        g["de_inf"] = ("d'" if f["A"][:1] in VOYELLES else "de ") + f["A"]
    if f["relation"] == "outil":
        g["de_inf_je"] = g["de_inf"].replace("de se ", "de me ")
    if f["relation"] == "propriete":
        g["P0"] = f["cles"][0]
    if f["relation"] == "famille":
        g["Fs"] = pluriel_nom(nom(f["F"]))
    if f["relation"] == "origine":
        g["vient"] = "viennent" if pluriel(f["X"]) else "vient"
        g["donne"] = "donnent" if pluriel(f["S"]) else "donne"
        g["de_nom_X"] = ("d'" if nom(f["X"])[:1] in VOYELLES else "de ") + nom(f["X"])
    if f["relation"] == "saison":
        g["en_X"] = en_saison(f["X"])
        g["En_X"] = maj(en_saison(f["X"]))
        g["que_V"] = que(f["V"])
    if f["relation"] == "contraire":
        g["de_A"] = ("d'" if f["A"][:1] in VOYELLES.replace("h", "") else "de ") + f["A"]
        g["de_B"] = ("d'" if f["B"][:1] in VOYELLES.replace("h", "") else "de ") + f["B"]
        g["B_maj"] = maj(f["B"])
    return g


# Relation -> phrases déclaratives (jamais une question).
PHRASES = {
    "pattes": ["{Un_X} a {N} pattes.", "{X_maj} marche sur {mot} pattes.", "{X_maj} possède {N} pattes.",
               "Si tu comptes les pattes {d_un_X}, tu en trouves {mot}.", "{X_maj} est un animal à {N} pattes.",
               "{mot_maj} pattes : c'est ce qu'a {un_X}.", "Les pattes {de_X} sont au nombre de {N}.",
               "{Un_X} se déplace sur {mot} pattes.", "Chaque {nom_X} a {N} pattes.", "{X_maj} a {mot} pattes."],
    "pattes0": ["{X_maj} n'a pas de pattes.", "{Un_X} n'a aucune patte.", "{X_maj} avance sans pattes, en rampant.",
                "Zéro patte : {X} n'en a pas.", "Chez {X}, pas de pattes : il rampe.", "{Un_X} rampe car il n'a pas de pattes.",
                "{X_maj} est un animal sans pattes.", "Aucune patte pour {X} : il glisse sur le sol."],
    "cri": ["{X_maj} {V}.", "On dit {que_X} {V}.", "Quand {X} fait du bruit, {il_X} {V}.",
            "{X_maj} est un animal qui {V}.", "Écoute {X} : {il_X} {V}.", "Le cri {de_X} : {il_X} {V}.",
            "Tout le monde sait {que_X} {V}.", "Dans la ferme ou ailleurs, {X} {V}.", "{Un_X}, ça {V}.",
            "C'est {X} qui {V}."],
    "petit": ["Le petit {de_X} s'appelle {P}.", "{P_maj} est le petit {de_X}.", "On appelle {P} le bébé {de_X}.",
              "Le bébé {de_X}, c'est {P}.", "{P_maj} grandira et deviendra {un_X}.", "Quand {X} a un petit, c'est {un_P}.",
              "{Un_P} est un jeune {nom_X}.", "Le nom du petit {de_X} est {P}.", "{X_maj} et son petit, {P}.",
              "Chez {X}, le petit s'appelle {P}."],
    "famille": ["{X_maj} est {F}.", "{Un_X} fait partie des {Fs}.", "{X_maj} appartient à la famille des {Fs}.",
                "Parmi les {Fs}, on trouve {X}.", "{X_maj}, c'est {F}.", "On range {X} parmi les {Fs}.",
                "{Un_X} est {F}, tout simplement.", "Il faut savoir {que_X} est {F}.", "Chez les {Fs}, il y a {X}.",
                "{X_maj} fait partie de la catégorie des {Fs}."],
    "milieu": ["{X_maj} vit {L}.", "On trouve {X} {L}.", "{X_maj} habite {L}.", "La maison {de_X}, c'est {L}.",
               "Si tu cherches {un_X}, regarde {L}.", "{Un_X} vit {L}.", "C'est {L} que vit {X}.",
               "{X_maj} passe sa vie {L}.", "L'endroit où vit {X} : {L}.", "Chez {X}, on vit {L}."],
    "couleur": ["{X_maj} est {C}.", "La couleur {de_X}, c'est {le_K}.", "{X_maj} est de couleur {K}.",
                "Si tu regardes {X}, tu vois {du_K}.", "{le_K_maj} est la couleur {de_X}.", "{X_maj}, c'est {K}.",
                "On reconnaît {X} à sa couleur : {K}.", "Tout le monde sait {que_X} est {C}.",
                "La couleur {de_X} est {le_K}.", "Pour dessiner {X}, prends {du_K}."],
    "nombre": ["{X_maj} a {N} {U}.", "{X_maj} compte {N} {U}.", "{X_maj} compte {mot} {U}.",
               "Le nombre {de_U} {de_X} est {N}.", "{X_maj} : {N} {U}.", "{X_maj} a exactement {mot} {U}.",
               "{mot_maj} {U}, c'est ce qu'a {X}.", "Retiens {que_X} a {N} {U}.", "Pour {X}, on compte {N} {U}.",
               "Dans {X}, il y a {N} {U}."],
    "sens": ["Pour {A}, on utilise {O}.", "{O_maj} {sert} à {A}.", "{O_maj} nous {permet} {de_inf}.",
             "C'est avec {O} qu'on peut {A}.", "Sans {O}, impossible {de_inf}.", "On se sert {de_O} pour {A}.",
             "Si on veut {A}, on a besoin {de_O}.", "{O_maj}, c'est pour {A}.", "Pour {A}, il faut {O}.",
             "Le corps utilise {O} pour {A}."],
    "outil": ["Pour {A}, on utilise {O}.", "{O_maj} {sert} à {A}.", "On se sert {de_O} pour {A}.",
              "Si tu veux {A}, prends {O}.", "{O_maj} {permet} {de_inf}.", "L'objet pour {A}, c'est {O}.",
              "Avec {O}, on peut {A}.", "Pour {A}, il faut {O}.", "{O_maj}, c'est pour {A}.",
              "Quand on veut {A}, on prend {O}."],
    "propriete": ["{X_maj} est {P}.", "{X_maj}, c'est {P0}.", "Tout le monde sait {que_X} est {P}.",
                  "Il faut savoir {que_X} est {P}.", "{X_maj} a la particularité d'être {P}.", "On sait {que_X} est {P}.",
                  "{X_maj} est toujours {P}.", "Retiens {que_X} est {P}."],
    "contraire": ["Le contraire {de_A} est {B}.", "{A_maj} est le contraire {de_B}.", "{A_maj} et {B} sont des contraires.",
                  "L'opposé {de_A}, c'est {B}.", "Le mot contraire de « {A} », c'est « {B} ».",
                  "{A_maj} ou {B} : ce sont deux contraires.", "Le contraire {de_B} est {A}.",
                  "« {A_maj} » s'oppose à « {B} ».", "{B_maj} est l'opposé {de_A}.", "L'inverse {de_A}, c'est {B}."],
    "origine": ["{X_maj} {vient} {de_S}.", "On obtient {X} à partir {de_S}.", "{S_maj} nous {donne} {X}.",
                "{X_maj}, ça {vient} {de_S}.", "C'est {S} qui nous {donne} {X}.", "Pour avoir {X}, il faut {S}.",
                "{X_maj} est fait à partir {de_S}.", "Sans {S}, pas {de_nom_X}.", "{S_maj} {donne} {X}.",
                "À l'origine {de_X}, il y a {S}."],
    "metier": ["{X_maj} {V}.", "Le travail {de_X} : il {V}.", "{Un_X} est quelqu'un qui {V}.", "Chaque jour, {X} {V}.",
               "On appelle {X} la personne qui {V}.", "C'est {X} qui {V}.", "Dans son métier, {X} {V}.",
               "{Un_X}, ça {V}.", "Le métier {de_X}, c'est simple : il {V}.", "Au travail, {X} {V}."],
    "saison": ["{En_X}, {V}.", "{X_maj} est la saison où {V}.", "Pendant {X}, {V}.", "C'est {en_X} {que_V}.",
               "On sait {que_V} {en_X}.", "{En_X}, en général, {V}.", "Quand vient {X}, {V}.",
               "{X_maj}, c'est la saison où {V}.", "Chaque année, {en_X}, {V}.", "Ce qui arrive {en_X} : {V}."],
}
# Pas de « Une brosse à dents ? » -> « Une brosse à dents, c'est pour... » : v14 plaquait ce moule sur
# « une idée de repas ? » (« Une idée de repas, c'est pour un repas »).
DEMANDES = ["Dis-moi quelque chose sur {sujet}.", "Un fait sur {sujet} ?", "Apprends-moi quelque chose.",
            "Raconte-moi quelque chose.", "Un fait intéressant ?", "Une chose à savoir sur {sujet} ?",
            "Apprends-moi un truc simple.", "Une évidence ?"]
# v15 : deux questions par fait, formulées autrement que l'examen. v14 n'avait que des phrases, et
# « Combien de pattes a un chien ? » -> six : la seule question de ce moule dans les données
# (« Combien de pattes a un insecte ? », dans les conversations de Claude) l'emportait.
QUESTIONS = {
    "pattes": ["Il a combien de pattes, {X} ?", "{X_maj} marche sur combien de pattes ?"],
    "pattes0": ["Il a combien de pattes, {X} ?", "{X_maj} marche sur combien de pattes ?"],
    "cri": ["Quel est le cri {de_X} ?", "Comment crie {X} ?"],
    "petit": ["Quel est le nom du petit {de_X} ?", "Comment on appelle un bébé {nom_X} ?"],
    "famille": ["{X_maj} fait partie de quelle famille ?", "Dans quelle catégorie on range {X} ?"],
    "milieu": ["Où habite {X} ?", "Dans quel endroit on trouve {X} ?"],
    "couleur": ["Quelle est la couleur {de_X} ?", "{X_maj} est de quelle couleur ?"],
    "nombre": ["{X_maj} compte combien {de_U} ?", "Il y a combien {de_U} dans {X} ?"],
    "sens": ["Qu'est-ce qui nous permet {de_inf} ?", "On utilise quoi pour {A} ?"],
    "outil": ["Quel objet sert à {A} ?", "Il faut quoi pour {A} ?"],
    "propriete": ["Comment on décrit {X} ?", "Qu'est-ce qui caractérise {X} ?"],
    "contraire": ["Quel mot s'oppose à « {A} » ?", "Donne-moi l'opposé {de_A}."],
    "origine": ["On fabrique {X} avec quoi ?", "{X_maj}, on l'obtient comment ?"],
    "metier": ["Quel est le métier {de_X} ?", "{Un_X}, ça sert à quoi ?"],
    "saison": ["Comment est la nature {en_X} ?", "Qu'est-ce qui arrive {en_X} ?"],
}

# Questions de l'examen à l'aveugle : aucune n'est dans les données, sous aucune forme.
TEST = {
    "pattes": ["Combien de pattes a {un_X} ?", "{un_X}, ça a combien de pattes ?"],
    "pattes0": ["Combien de pattes a {un_X} ?", "{un_X}, ça a combien de pattes ?"],
    "cri": ["Quel bruit fait {X} ?", "{X_maj}, il fait quoi comme cri ?"],
    "petit": ["Comment s'appelle le petit {de_X} ?", "Le bébé {de_X}, on l'appelle comment ?"],
    "famille": ["{X_maj}, c'est quel type d'animal ou de plante ?", "{Un_X}, c'est quoi comme être vivant ?"],
    "milieu": ["Où vit {X} ?", "{Un_X}, ça vit où ?"],
    "couleur": ["De quelle couleur est {X} ?", "{X_maj}, c'est quelle couleur ?"],
    "nombre": ["Combien {de_U} a {X} ?", "{X_maj}, ça fait combien {de_U} ?"],
    "sens": ["Avec quoi on peut {A} ?", "Quelle partie du corps sert à {A} ?"],
    "outil": ["Qu'est-ce qu'on utilise pour {A} ?", "J'ai besoin {de_inf_je}, je prends quoi ?"],
    "propriete": ["{X_maj}, c'est comment ?", "Comment est {X} ?"],
    "contraire": ["Quel est le contraire {de_A} ?", "C'est quoi l'inverse {de_A} ?"],
    "origine": ["D'où {vient} {X} ?", "{X_maj}, ça provient de quoi ?"],
    "metier": ["Que fait {un_X} ?", "C'est quoi le travail {de_X} ?"],
    "saison": ["Que se passe-t-il {en_X} ?", "{En_X}, comment c'est dehors ?"],
}


def cle_modele(f: dict) -> str:
    return "pattes0" if f["relation"] == "pattes" and f["N"] == "0" else f["relation"]


def donnees(faits: list[dict]) -> list[list[dict]]:
    convs = []
    for f in faits:
        g = gabarit(f)
        g["sujet_maj"] = maj(f["sujet"])
        phrases = PHRASES[cle_modele(f)]
        for i in range(PHRASES_PAR_FAIT):
            convs.append([{"role": "user", "content": maj(R.choice(DEMANDES).format(**g))},
                          {"role": "assistant", "content": maj(phrases[i % len(phrases)].format(**g))}])
        for q in QUESTIONS[cle_modele(f)]:
            convs.append([{"role": "user", "content": maj(q.format(**g))},
                          {"role": "assistant", "content": maj(R.choice(phrases).format(**g))}])
    R.shuffle(convs)
    return convs


def questions_test(faits: list[dict]) -> list[dict]:
    test = []
    for f in faits:
        g = gabarit(f)
        for q in TEST[cle_modele(f)]:
            test.append({"question": maj(q.format(**g)), "cles": f["cles"], "relation": f["relation"]})
    return test


def verifier(convs: list[list[dict]], test: list[dict]) -> None:
    """Aucune question de test dans les données, aucune question dans les réponses."""
    vues = {normaliser(m["content"]) for c in convs for m in c}
    for t in test:
        assert normaliser(t["question"]) not in vues, t["question"]
    assert not any("?" in c[1]["content"] for c in convs)


def main() -> None:
    faits = lire()
    convs = donnees(faits)
    test = questions_test(faits)
    verifier(convs, test)
    SORTIE.write_text(json.dumps(convs, ensure_ascii=False), encoding="utf-8")
    print(f"{len(faits)} faits, {len(convs)} conversations -> {SORTIE}  ({len(test)} questions d'examen à part)")
    for c in R.sample(convs, 12):
        print(f"   {c[0]['content']}  ->  {c[1]['content']}")


if __name__ == "__main__":
    main()
