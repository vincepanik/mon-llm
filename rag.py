"""
Recherche dans Wikipédia pour Carl (RAG : retrieval-augmented generation).

À 125M paramètres, Carl ne retient qu'environ la moitié des faits simples
(le modèle de base en complète 20 sur 40). Au lieu de lui demander de tout
savoir, on cherche le passage de Wikipédia qui parle de la question et on le
lui donne à lire : il n'a plus qu'à y trouver la réponse, ce qu'un petit
modèle sait bien faire. C'est ce que font les assistants modernes pour éviter
d'inventer.

    python rag.py --construire        # une fois : indexe data/big/raw/wikipedia_fr.txt
    python rag.py --vecteurs          # une fois (~1 h) : l'empreinte de sens de chaque passage
    python rag.py "Quelle est la capitale du Japon ?"

L'index porte sur le début de chaque article (titre + ~900 caractères) : c'est
là que Wikipédia met l'essentiel, et 1,36 million de passages tiennent en
quelques Go. Recherche par BM25 (mots-clés), avec les mots vides et la
racinisation du français.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import bm25s
import Stemmer

SOURCE = Path("data/big/raw/wikipedia_fr.txt")
INDEX = Path("data/big/rag/index")
SEPARATEUR = "<|endoftext|>"
# Caractères de texte par passage, en plus du titre : la même taille que dans
# les exemples de lecture (data/lecture.py). À 900, la capitale du Canada ou la
# langue du Brésil tombaient souvent juste après la coupure.
LONGUEUR = 1400


def passage(document: str) -> str | None:
    """Titre + début du texte, coupé à la fin d'une phrase."""
    titre, _, texte = document.strip().partition("\n")
    texte = re.sub(r"\s+", " ", texte).strip()
    if len(texte) < 200:
        return None
    if len(texte) > LONGUEUR:
        coupe = texte.rfind(". ", 0, LONGUEUR)
        texte = texte[: coupe + 1] if coupe > LONGUEUR // 2 else texte[:LONGUEUR]
    return f"{titre.strip()}\n{texte}"


def documents(source: Path = SOURCE):
    """Les articles un par un, en flux (le fichier fait 5 Go)."""
    lignes: list[str] = []
    with source.open(encoding="utf-8") as f:
        for ligne in f:
            if ligne.strip() == SEPARATEUR:
                if lignes:
                    yield "".join(lignes)
                lignes = []
            else:
                lignes.append(ligne)
    if lignes:
        yield "".join(lignes)


def construire() -> None:
    t0 = time.time()
    corpus = [p for d in documents() if (p := passage(d))]
    print(f"{len(corpus):,} passages en {time.time() - t0:.0f} s", flush=True)
    stemmer = Stemmer.Stemmer("french")
    jetons = bm25s.tokenize(corpus, stopwords="fr", stemmer=stemmer, show_progress=True)
    moteur = bm25s.BM25()
    moteur.index(jetons, show_progress=True)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    moteur.save(str(INDEX), corpus=corpus)
    print(f"index écrit dans {INDEX} en {(time.time() - t0) / 60:.0f} min")


# Mots de la question qui ne disent rien de son sujet. Sans eux, « Quelle est
# la capitale de l'Espagne ? » ramenait l'article « Quel », et « En quelle
# année... » l'article « Neuf Jours d'une année ».
INTERROGATIFS = """quel quelle quels quelles qui que quoi quand où comment combien pourquoi lequel laquelle
    est sont était sera fait font faire a ont y t il elle ils elles on-t-il ce c est-ce année ans an
    dis dites explique expliquer donne donner connais connaître sais savoir peux pouvez peut
    moi toi nous vous je tu me te plus grand grande petit petite premier première""".split()


@lru_cache(maxsize=1)
def _moteur():
    moteur = bm25s.BM25.load(str(INDEX), load_corpus=True, mmap=True)
    mots_vides = sorted(set(bm25s.stopwords.STOPWORDS_FRENCH) | set(INTERROGATIFS))
    return moteur, Stemmer.Stemmer("french"), mots_vides


TITRES = INDEX.parent / "titres.json"


def _normaliser(texte: str) -> str:
    texte = texte.lower().replace("’", "'")
    texte = re.sub(r"\b(l|d|qu)'", "", texte)
    return re.sub(r"\s+", " ", texte).strip(" ?!.,")


@lru_cache(maxsize=1)
def _titres() -> dict[str, int]:
    """Titre normalisé -> numéro du passage. Calculé une fois, puis gardé sur disque."""
    import json

    if not TITRES.exists():
        titres: dict[str, int] = {}
        with (INDEX / "corpus.jsonl").open(encoding="utf-8") as f:
            for i, ligne in enumerate(f):
                titre = _normaliser(json.loads(ligne)["text"].split("\n", 1)[0])
                titres.setdefault(titre, i)
        TITRES.write_text(json.dumps(titres, ensure_ascii=False), encoding="utf-8")
    return json.loads(TITRES.read_text(encoding="utf-8"))


def _titres_dans(question: str) -> list[tuple[int, int]]:
    """
    Articles dont le titre est un groupe de mots de la question : (passage,
    nombre de mots). Seulement les groupes qui contiennent un nom propre (une
    majuscule ailleurs qu'en début de phrase) : sinon « capitale », « océan »
    ou « symphonie », titres d'articles eux aussi, passaient devant « Espagne ».
    """
    # Mot par mot, en écartant la ponctuation isolée (« ... Misérables ? ») :
    # normalisée d'un bloc, la phrase n'avait plus le même nombre de mots et la
    # recherche par titre abandonnait en silence.
    originaux = [m for m in question.replace("’", "'").split() if re.search(r"\w", m)]
    normalises = [_normaliser(m) for m in originaux]
    propre = [i > 0 and re.sub(r"^(l|d|qu)'", "", m, flags=re.I)[:1].isupper() for i, m in enumerate(originaux)]
    titres = _titres()
    trouves = {}
    for n in range(min(4, len(normalises)), 0, -1):
        for i in range(len(normalises) - n + 1):
            if not any(propre[i : i + n]):
                continue
            groupe = " ".join(normalises[i : i + n])
            if groupe in titres:
                trouves.setdefault(titres[groupe], n)
    return list(trouves.items())


def _jetons(texte: str, stemmer, mots_vides) -> set[str]:
    t = bm25s.tokenize([texte], stopwords=mots_vides, stemmer=stemmer, show_progress=False, return_ids=False)
    return set(t[0]) if t and t[0] else set()


# Questions adressées à Carl (« Qui t'a conçu ? », « Tu appartiens à qui ? »)
# ou calculs : chercher dans Wikipédia ne peut que l'induire en erreur. Mesuré :
# avec la recherche, il répondait « Bob Ackerman » à « Qui t'a conçu ? » et
# citait la chanson « Je t'appartiens » ; identité 9/10 -> 2/10.
A_CARL = re.compile(r"\b(tu|te|toi|ton|ta|tes|vous|votre|vos|carl)\b|\bt'|\bt’", re.I)


# Ce qui se passe maintenant : aucun article ne le dit (« Il est quelle heure
# maintenant ? » -> « 95 minutes. », lu dans un passage au hasard).
MAINTENANT = re.compile(r"\b(maintenant|aujourd'hui|heure|météo|ce soir|demain|hier)\b", re.I)


def utile(question: str) -> bool:
    """Faut-il chercher dans Wikipédia pour cette question ?"""
    return not (A_CARL.search(question) or MAINTENANT.search(question) or re.search(r"\d", question))


def disponible() -> bool:
    return INDEX.exists()


def chercher(question: str, k: int = 1, candidats: int = 100, methode: str | None = None) -> list[tuple[str, float]]:
    """
    Les k passages les plus proches de la question, avec leur score.

    methode="mots" : BM25 ramène `candidats` passages, puis on reclasse : un
    article dont le titre est entièrement dans la question (« Espagne » pour
    « la capitale de l'Espagne ») est presque toujours le bon, même si son
    texte, long et général, a un moins bon score BM25 qu'un court article voisin.

    methode="sens" (par défaut quand les vecteurs existent) : les passages
    les plus proches par le sens (embeddings). Mesuré (examen_rag.py), bon
    passage en tête : 34/55, contre 27 par mots et 28 en hybride.

    methode="hybride" : les deux
    classements fusionnés par leurs rangs (RRF, « reciprocal rank fusion ») :
    un passage bien placé dans les deux passe devant. Aucun réglage, donc rien
    d'ajusté sur les questions de l'examen. Moins bon que le sens seul : le
    bruit de BM25 (« Miseration » pour « Les Misérables ») s'y retrouve.
    """
    if methode is None:
        methode = "sens" if VECTEURS.exists() else "mots"
    moteur, stemmer, mots_vides = _moteur()
    mots_question = _jetons(question, stemmer, mots_vides)
    if not mots_question:  # que des mots vides (« bonjour », « merci »...)
        return []
    texte = lambda i: moteur.corpus[i]["text"]  # noqa: E731
    if methode == "sens":
        return [(texte(i), sim) for i, sim in chercher_sens(question, k)]
    mots = _par_mots(question, moteur, stemmer, mots_vides, mots_question, candidats)
    if methode == "mots":
        return [(texte(i), score) for i, score in mots[:k]]
    fusion: dict[int, float] = {}
    for classement in (mots, chercher_sens(question, candidats)):
        for rang, (i, _) in enumerate(classement):
            fusion[i] = fusion.get(i, 0.0) + 1.0 / (60 + rang)
    return [(texte(i), score) for i, score in sorted(fusion.items(), key=lambda x: -x[1])[:k]]


def _par_mots(question, moteur, stemmer, mots_vides, mots_question, candidats) -> list[tuple[int, float]]:
    """Classement BM25 + bonus de titre : [(numéro du passage, score)], meilleur d'abord."""
    requete = bm25s.tokenize([question], stopwords=mots_vides, stemmer=stemmer, show_progress=False)
    ids, scores = moteur.retrieve(requete, k=candidats, show_progress=False, return_as="tuple")
    # Index chargé avec son corpus : le moteur rend les passages eux-mêmes ({"id", "text"}).
    base = {int(d["id"]) if isinstance(d, dict) else int(d): float(sc) for d, sc in zip(ids[0], scores[0])}
    # Titre exact dans la question : candidat d'office, avec un bonus d'autant
    # plus fort que le titre est long (« Le Petit Prince » > « Prince »).
    exacts = dict(_titres_dans(question))
    classes = []
    for i in set(base) | set(exacts):
        score = base.get(i, min(base.values(), default=0.0))
        titre = _jetons(moteur.corpus[i]["text"].split("\n", 1)[0], stemmer, mots_vides)
        if titre:
            score += 10.0 * len(titre & mots_question) / len(titre) * (1.0 if titre <= mots_question else 0.5)
        if i in exacts:
            score += 8.0 + 3.0 * exacts[i]
        classes.append((i, score))
    classes.sort(key=lambda x: -x[1])
    return classes


# --- Recherche par le sens (embeddings) ---
# BM25 compare des mots : « Qui a peint la Joconde ? » ne trouve pas un passage
# qui dit « tableau de Léonard de Vinci ». Un petit modèle d'embeddings
# (multilingual-e5-small, 118M paramètres, licence MIT) résume chaque passage
# en 384 nombres ; deux textes qui parlent de la même chose ont des vecteurs
# proches, même sans mot commun.

MODELE_SENS = "intfloat/multilingual-e5-small"
VECTEURS = INDEX.parent / "vecteurs"
TRANCHE = 50_000


@lru_cache(maxsize=1)
def _encodeur(en_ligne: bool = False):
    import torch
    from transformers import AutoModel, AutoTokenizer

    from utils import get_device

    device = get_device()
    # Hors ligne une fois téléchargé : le modèle est lu dans le cache local.
    tk = AutoTokenizer.from_pretrained(MODELE_SENS, local_files_only=not en_ligne)
    modele = AutoModel.from_pretrained(MODELE_SENS, local_files_only=not en_ligne,
                                       dtype=torch.float16 if device != "cpu" else torch.float32)
    return tk, modele.to(device).eval(), device


def encoder(textes: list[str], prefixe: str, longueur: int = 384, en_ligne: bool = False):
    """Vecteurs normalisés (float16) ; e5 attend « query: » ou « passage: » devant le texte."""
    import torch

    tk, modele, device = _encodeur(en_ligne)
    x = tk([prefixe + t for t in textes], max_length=longueur, truncation=True, padding=True,
           return_tensors="pt").to(device)
    with torch.no_grad():
        sortie = modele(**x).last_hidden_state
    masque = x["attention_mask"][..., None].to(sortie.dtype)
    v = (sortie * masque).sum(1) / masque.sum(1)
    return torch.nn.functional.normalize(v.float(), dim=-1).half().cpu()


def construire_vecteurs(lot: int = 128) -> None:
    """Par tranches de 50 000 passages, écrites au fur et à mesure : reprenable."""
    import json

    import numpy as np

    VECTEURS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with (INDEX / "corpus.jsonl").open(encoding="utf-8") as f:
        numero, tranche = 0, []
        for ligne in f:
            tranche.append(json.loads(ligne)["text"])
            if len(tranche) == TRANCHE:
                _tranche(numero, tranche, lot, np)
                numero, tranche = numero + 1, []
                print(f"  {numero * TRANCHE:,} passages ({(time.time() - t0) / 60:.0f} min)", flush=True)
        if tranche:
            _tranche(numero, tranche, lot, np)
    print(f"vecteurs écrits dans {VECTEURS} en {(time.time() - t0) / 60:.0f} min")


def _tranche(numero: int, textes: list[str], lot: int, np) -> None:
    chemin = VECTEURS / f"{numero:03d}.npy"
    if chemin.exists():
        return
    import torch

    # Triés par longueur : moins de remplissage dans chaque lot, deux fois plus rapide.
    ordre = sorted(range(len(textes)), key=lambda i: len(textes[i]))
    sortie = torch.empty(len(textes), 384, dtype=torch.float16)
    for i in range(0, len(ordre), lot):
        idx = ordre[i : i + lot]
        sortie[idx] = encoder([textes[j] for j in idx], "passage: ", en_ligne=True)
    np.save(chemin, sortie.numpy())


@lru_cache(maxsize=1)
def _vecteurs():
    import numpy as np
    import torch

    from utils import get_device

    tranches = sorted(VECTEURS.glob("*.npy"))
    return torch.from_numpy(np.concatenate([np.load(t) for t in tranches])).to(get_device())


def chercher_sens(question: str, k: int = 100) -> list[tuple[int, float]]:
    """(numéro du passage, similarité entre 0 et 1) des k passages les plus proches par le sens."""
    import torch

    v = _vecteurs()
    q = encoder([question], "query: ").to(v.device)
    scores = (v @ q[0]).float()
    meilleurs = torch.topk(scores, k)
    return list(zip(meilleurs.indices.tolist(), meilleurs.values.tolist()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--construire", action="store_true")
    parser.add_argument("--vecteurs", action="store_true")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()
    if args.construire:
        construire()
    if args.vecteurs:
        construire_vecteurs()
    if args.question:
        for texte, score in chercher(args.question, args.k):
            print(f"[{score:.1f}] {texte[:200]}\n")


if __name__ == "__main__":
    sys.exit(main())
