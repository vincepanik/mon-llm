"""
Base de faits de Carl (Wikidata, téléchargée par data/wikidata.py).

Comme pour la calculatrice (outils.py), Carl n'a pas à savoir : il apprend
(data/faits_sft.py) à écrire « [fait: Espagne | capitale = », le programme
cherche ici, insère « Madrid] », et Carl recopie le résultat dans sa phrase.
À l'affichage, l'appel disparaît : il ne reste que « La capitale de
l'Espagne est Madrid. ».

Carl écrit les noms à peu près : « l'espagne », « Hugo », « Van Gogh »,
« Louis IX », « Chine », « nietzche ». Mais une base qui répond à tout est pire
qu'une base qui se tait : Carl recopie sa réponse avec l'autorité de l'outil,
alors que sans réponse il répond de mémoire, comme sans outil. Le principe :
ne répondre que quand l'identification est NETTE. Trois relectures par des
agents ont trouvé une centaine de cas où une règle plus lâche inventait
(« Kennedy » -> un inconnu, « Jésus » -> Gabriel Jesus, « Louis » -> Louis
XIV, « Gerld Ford » -> Henry Ford, « Wakanda » -> Kigali, « Waterloo | date »
-> la chanson d'ABBA, « Léa Martin » -> Dean Martin, « Dordogne » -> un
village) : les plus parlants sont dans tests/test_faits.py.

La recherche procède par paliers, du plus sûr au plus risqué :

1. le nom officiel exact, écrit autrement au besoin (« Napoléon 1er »,
   « Louis 14 », « 1ère Guerre mondiale ») ; pour une personne nommée d'un seul
   mot, seulement si elle domine ses homonymes (« Montaigne » est l'écrivain,
   pas la chanteuse de ce nom) ; pour une date, une bataille bien plus connue
   passe avant l'œuvre homonyme (« Waterloo » : 1815, pas ABBA) ;
2. un autre nom de l'entité (alias Wikidata : « Napoléon Bonaparte »,
   « Chine », « Mona Lisa »), un morceau du nom d'une personne (nom de famille
   « Hugo », « Van Gogh » ; souverain avec son numéro « Louis IX » ; prénom et
   nom sans le milieu « Wolfgang Mozart »), la fin du nom d'un lieu, d'une
   montagne, d'un événement ou d'une entreprise (« Ventoux », « Verdun »), un
   titre sans son sous-titre (« Candide ») : ces candidats sont réunis, et le
   premier n'est gardé que s'il DOMINE (une fois et demie plus connu que le
   suivant ; un alias compte double) ;
3. un mot générique en trop devant (« mont everest », « ville de lyon ») ;
4. une faute de frappe, avec des garde-fous : au plus une faute par mot pour
   une personne (« Léa Martin » n'est pas Dean Martin), même première lettre
   (la Drôme n'est pas Rome), mêmes nombres (Apollo 11 n'est pas Apollo 18 ;
   Louis IX n'est pas Louis XV), un mot seul mal tapé résolu comme le mot juste
   (« Kennedi » -> ce que « Kennedy » désigne, ou rien s'il est ambigu) ;
5. un long mot inconnu corrigé s'il n'y a qu'une correction (« nietzche »).

Et une règle au-dessus : si le nom désigne exactement une entité connue qui
n'a pas la relation demandée (« Paris | naissance », « Jésus | naissance »,
« Hiroshima | date »), un candidat moins connu trouvé par un palier
approximatif n'est pas la réponse : on se tait. Un mot présent dans beaucoup
de noms (« révolution », « guerre ») ne désigne rien de précis par lui-même.
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

CHEMIN = Path("data/big/wikidata/faits.json")

# Ce que Carl peut écrire après « | », ramené aux relations de la base (clés
# normalisées : sans accents ni majuscules). Chaque entrée est une liste de
# GROUPES essayés dans l'ordre ; dans un groupe, les relations sont
# équivalentes (« date » d'un événement = sa date, son début ou sa création).
# Un groupe suivi de « @événement » ne vaut que pour un événement : le début
# ou la fin d'une bataille peut se rabattre sur sa date, mais « guerre | fin »
# ne doit pas donner la date de parution du livre « De la guerre ».
DATE = [["date", "début", "création"]]
RELATIONS = {
    "capitale": [["capitale"]], "capital": [["capitale"]], "monnaie": [["monnaie"]], "devise": [["monnaie"]],
    "langue": [["langue"]], "langues": [["langue"]], "langue officielle": [["langue"]],
    "continent": [["continent"]],
    "population": [["population"]], "habitants": [["population"]], "pays": [["pays"]],
    "naissance": [["naissance"]], "ne": [["naissance"]], "nee": [["naissance"]],
    "date de naissance": [["naissance"]], "annee de naissance": [["naissance"]],
    "deces": [["décès"]], "mort": [["décès"]], "date de deces": [["décès"]], "annee de deces": [["décès"]],
    "lieu de naissance": [["lieu de naissance"]], "ville natale": [["lieu de naissance"]],
    "auteur": [["auteur"]], "ecrivain": [["auteur"]], "peintre": [["auteur"]],
    "compositeur": [["compositeur"]], "realisateur": [["réalisateur"]],
    "date": DATE, "annee": DATE, "sortie": [["date"]], "date de sortie": [["date"]], "publication": [["date"]],
    "debut": [["début"], ["date", "@événement"]], "fin": [["fin"], ["date", "@événement"]],
    "symbole": [["symbole"]], "numero atomique": [["numéro atomique"]],
    "altitude": [["altitude"]], "hauteur": [["altitude"]],
    "fondateur": [["fondateur"]], "creation": [["création"]], "siege": [["siège"]],
    # « Qui était Albert Einstein ? » -> « physicien helvético-américain d'origine allemande... »
    "description": [["description"]], "qui est": [["description"]], "c est quoi": [["description"]],
}
LIEUX = {"lieu de naissance", "siège", "pays"}
DATES = {"naissance", "décès", "date", "début", "fin", "création"}
NOMBRES = {"population", "altitude", "numéro atomique"}
ARTICLES = re.compile(r"^(?:de la |de l |du |des |de |d |la |le |les |l )+")
# Numéros de règne : « Napoléon Ier », « Louis XIV », « Élisabeth II ».
REGNE = re.compile(r"^(?:[ivxl]+|ier|ire|1er|1re|premier|premiere|le grand)$")
ROMAIN = re.compile(r"^(x{0,3})(ix|iv|v?i{0,3})$")
# Types dont on reconnaît un nom par sa fin : « mont ventoux », « bataille de
# verdun », « republique populaire de chine », « the walt disney company ».
PAR_LA_FIN = {"montagne", "événement", "pays", "entreprise", "ville"}
# Devant la fin d'un nom de ville, ces petits mots disent que c'est une AUTRE
# ville : « Beaulieu-sur-Dordogne » n'est pas la Dordogne.
LIAISONS = set("sur sous en de d du des les la le l aux au et lez".split())
# Mots génériques qu'on peut retirer devant un nom (palier 3), et seulement eux :
# retirer un prénom (« gerld ford » -> « ford ») ou la tête d'un groupe
# (« guerre de troie » -> le film « Troie ») changeait d'entité. Pas « île » :
# « Île-de-France » devenait la France.
EN_TETE = re.compile(r"^(?:la |le |l )?(?:mont|massif|pic|volcan|ville|cite|commune|pays|film|livre|roman|"
                     r"tableau|oeuvre|album|entreprise|societe|groupe|marque|firme)(?: de la| de l| du| des| de| d)? ")
DOMINE = 1.5   # le premier candidat approximatif doit être 1,5 fois plus connu que le suivant
NOMBRES_ECRITS = {m: str(i) for i, m in enumerate(
    "zero un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize".split())}
NOMBRES_ECRITS |= {"dix sept": "17", "dix huit": "18", "dix neuf": "19", "vingt": "20", "cent": "100",
                   "premier": "1", "premiere": "1", "1er": "1", "1re": "1", "1ere": "1", "ier": "1", "ire": "1",
                   "second": "2", "seconde": "2", "deuxieme": "2", "2e": "2", "2eme": "2", "troisieme": "3",
                   "3e": "3", "3eme": "3", "quatrieme": "4", "cinquieme": "5", "sixieme": "6"}
POSSESSIFS = set("ma mon mes ta ton tes sa son ses notre nos votre vos leur leurs".split())


def normaliser(texte: str) -> str:
    """« de l'Espagne » -> « espagne » ; « Saint-Exupéry » -> « saint exupery »."""
    # « œ », « æ » ne se décomposent pas : sans ça, « Schœlcher » devenait « sch lcher ».
    texte = texte.lower().replace("œ", "oe").replace("æ", "ae").replace("ß", "ss")
    texte = unicodedata.normalize("NFD", texte)
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    texte = re.sub(r"[^a-z0-9]+", " ", texte).strip()
    return ARTICLES.sub("", texte).strip()


def _en_chiffres(nom: str) -> str:
    """« premiere guerre mondiale » -> « 1 guerre mondiale » ; « louis xiv » -> « louis 14 »."""
    for mots, chiffre in sorted(NOMBRES_ECRITS.items(), key=lambda x: -len(x[0])):
        nom = re.sub(rf"\b{mots}\b", chiffre, nom)
    mots = nom.split()
    for i, m in enumerate(mots):
        if i > 0 and m and ROMAIN.match(m):  # pas le premier mot : « Vi » (prénom) n'est pas 6
            valeur, precedent = 0, 0
            for c in reversed(m):
                v = {"i": 1, "v": 5, "x": 10}[c]
                valeur += -v if v < precedent else v
                precedent = max(precedent, v)
            mots[i] = str(valeur)
    return " ".join(mots)


def _chiffres(nom: str) -> list[str]:
    return re.findall(r"\d+", _en_chiffres(nom))


def _dl(a: str, b: str) -> int:
    """Nombre de fautes entre deux mots (Damerau-Levenshtein : une inversion compte pour une)."""
    d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _une_faute(a: str, b: str) -> bool:
    """Au plus une faute (deux pour un mot de plus de douze lettres) : « merx » = « marx », « bush » != « burns »."""
    return _dl(a, b) <= (2 if min(len(a), len(b)) > 12 else 1)


@lru_cache(maxsize=1)
def _base() -> tuple[dict, dict, dict[str, list[str]]]:
    donnees = json.loads(CHEMIN.read_text(encoding="utf-8"))
    entites, faits = donnees["entites"], donnees["faits"]
    index: dict[str, list[str]] = {}
    for qid, e in entites.items():
        index.setdefault(normaliser(e["nom"]), []).append(qid)
    for qids in index.values():  # le plus connu d'abord : « Paris » la ville, pas un homonyme
        qids.sort(key=lambda q: -entites[q]["liens"])
    return entites, faits, index


@lru_cache(maxsize=1)
def _index_chiffres() -> dict[str, list[str]]:
    """Les noms avec un nombre écrit en lettres ou en chiffres romains, sous leur forme en chiffres."""
    _, _, index = _base()
    autre: dict[str, list[str]] = {}
    for n, qids in index.items():
        if (c := _en_chiffres(n)) != n:
            autre.setdefault(c, []).extend(qids)
    return autre


@lru_cache(maxsize=1)
def _premiers_mots() -> dict[str, int]:
    """Combien de personnes ont chaque premier mot : « henri » des dizaines, « dante » trois."""
    entites, _, _ = _base()
    compte: dict[str, int] = {}
    for e in entites.values():
        if e["type"] == "personne" and (mots := normaliser(e["nom"]).split()):
            compte[mots[0]] = compte.get(mots[0], 0) + 1
    return compte


@lru_cache(maxsize=1)
def _frequences() -> dict[str, int]:
    """Dans combien de noms apparaît chaque mot : « guerre », « révolution », « nord » sont trop courants."""
    _, _, index = _base()
    compte: dict[str, int] = {}
    for n in index:
        for m in set(n.split()):
            compte[m] = compte.get(m, 0) + 1
    return compte


def _alias_generique(alias: str) -> bool:
    """« Ma mère », « le Général », « The King » : des noms communs, pas des noms."""
    mots = re.findall(r"[\w']+", alias.lower())
    if not mots or mots[0] in POSSESSIFS:
        return True
    return len(mots) == 2 and (mots[0] in ("le", "la", "les", "the") or mots[0].startswith("l'"))


@lru_cache(maxsize=1)
def _tous_les_alias() -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    """
    Trois tables d'autres noms (alias Wikidata) :
    - les alias ordinaires ;
    - les prénoms seuls d'une personne (« Dante », « Léonard », « Henri ») :
      ils ne gagnent que s'ils écrasent tous les autres candidats (Dante
      Alighieri contre Joe Dante), pas quand il y a dix Henri ;
    - les pseudonymes d'une personne, sans aucun mot de son nom (« Saint-Just »
      pour Romain Rolland) : seulement s'il n'y a rien d'autre.
    Écartés : un surnom d'un seul mot absent du nom (« Jupiter » pour Macron,
    « Roberts », nom de jeune fille de Thatcher) et, pour une chose, un mot
    courant (« bataille », « paris », « histoire » : un alias de la bataille de
    Watling Street ou de la Symphonie nº 31).
    """
    entites, _, _ = _base()
    frequences = _frequences()
    alias: dict[str, list[str]] = {}
    prenoms: dict[str, list[str]] = {}
    pseudos: dict[str, list[str]] = {}
    for qid, e in entites.items():
        mots = normaliser(e["nom"]).split()
        for a in e.get("alias", []):
            n = normaliser(a)
            if not n or _alias_generique(a):
                continue
            table = alias
            if e["type"] == "personne":
                if " " not in n:
                    if n not in mots:
                        continue
                    if n == mots[0] and len(mots) > 1 and not all(REGNE.match(m) for m in mots[1:]):
                        table = prenoms  # « Napoléon » pour Napoléon Ier reste un alias ordinaire
                elif not set(n.split()) & set(mots):
                    table = pseudos
            elif " " not in n and frequences.get(n, 0) > 15 and (not mots or n != mots[-1]):
                continue
            if qid not in table.setdefault(n, []):
                table[n].append(qid)
    for table in (alias, prenoms, pseudos):
        for qids in table.values():
            qids.sort(key=lambda q: -entites[q]["liens"])
    return alias, prenoms, pseudos


def _alias() -> dict[str, list[str]]:
    return _tous_les_alias()[0]


@lru_cache(maxsize=64)
def _noms_avec(relation: str) -> dict[str, list[str]]:
    """Noms officiels puis alias (normalisés) des entités qui ont cette relation ; le nom officiel d'abord."""
    _, faits, index = _base()
    noms: dict[str, list[str]] = {}
    for source in (index, _alias()):
        for n, qs in source.items():
            for q in qs:
                if relation in faits.get(q, {}) and q not in noms.setdefault(n, []):
                    noms[n].append(q)
    return {n: qs for n, qs in noms.items() if qs}


@lru_cache(maxsize=1)
def _vocabulaire() -> set[str]:
    _, _, index = _base()
    return {m for source in (index, _alias()) for n in source for m in n.split()}


@lru_cache(maxsize=64)
def _titres_courts(relation: str) -> dict[str, list[str]]:
    """« candide » -> « Candide, ou l'Optimisme » : le titre avant la virgule ou les deux-points."""
    entites, faits, _ = _base()
    courts: dict[str, list[str]] = {}
    for qid, e in entites.items():
        if e["type"] != "personne" and relation in faits.get(qid, {}):
            court = re.split(r"\s*[,:(]", e["nom"], maxsplit=1)[0]
            if court != e["nom"] and normaliser(court):
                courts.setdefault(normaliser(court), []).append(qid)
    return courts


def vider_cache() -> None:
    """Pour les tests, qui changent de base."""
    for f in (_base, _index_chiffres, _premiers_mots, _frequences, _tous_les_alias, _noms_avec, _vocabulaire,
              _titres_courts, _mot_le_plus_proche):
        f.cache_clear()


def disponible() -> bool:
    return CHEMIN.exists()


def _annee(date: str) -> int:
    """« 14 juin 1991 » -> 1991 ; « 1389 av. J.-C. » -> -1389."""
    nombres = re.findall(r"\d+", date)
    annee = int(nombres[-1]) if nombres else 0
    return -annee if "av. J.-C." in date else annee


def _liens(qid: str) -> int:
    return _base()[0][qid]["liens"]


def _type(qid: str) -> str:
    return _base()[0][qid]["type"]


def _a(qid: str, groupe: list[str]) -> bool:
    types = [g[1:] for g in groupe if g.startswith("@")]
    if types and _type(qid) not in types:
        return False
    return any(r in _base()[1].get(qid, {}) for r in groupe if not r.startswith("@"))


def _dans_l_ordre(petits: list[str], grands: list[str]) -> bool:
    it = iter(grands)
    return all(p in it for p in petits)


class Ambigu(Exception):
    """Plusieurs entités répondent aussi bien : on se tait, sans chercher plus loin."""


def _domine(poids: dict[str, float], facteur: float = DOMINE) -> list[str]:
    """
    Le premier, s'il est au moins `facteur` fois plus lourd que le deuxième.
    Sinon, c'est ambigu, et les paliers suivants ne doivent pas trancher à sa
    place : « François Ier » désigne trois souverains, et l'orthographe
    « François 1er », que seul le roi de France a en alias, ne le rend pas plus
    sûr.
    """
    classes = sorted(poids, key=lambda x: -poids[x])
    if len(classes) >= 2 and poids[classes[0]] < facteur * poids[classes[1]]:
        raise Ambigu
    return classes[:1]


# --- Palier 1 : le nom exact. ---

def _exacts(nom: str, groupe: list[str]) -> list[str]:
    """Le nom officiel, ou le même écrit autrement (« napoleon 1er » = « napoleon ier »)."""
    _, _, index = _base()
    trouves = [x for x in index.get(nom, []) if _a(x, groupe)]
    if not trouves:
        trouves = [x for x in _index_chiffres().get(_en_chiffres(nom), []) if _a(x, groupe)]
    return sorted(trouves, key=lambda x: -_liens(x))


def _personne_nette(nom: str, groupe: list[str], exacte: str) -> list[str]:
    """
    Une personne nommée d'un seul mot, ou avec un numéro de règne : elle doit
    dominer ceux que ce nom désigne aussi (nom de famille, alias). « Montaigne »
    est l'écrivain (140 liens), pas la chanteuse de ce nom (33) ; « Henri IV »
    a trop d'homonymes pour choisir.
    """
    poids = {exacte: 2 * _liens(exacte)}
    for x in _morceau_de_personne(nom, groupe) + [y for y in _alias().get(nom, []) if _a(y, groupe)]:
        if x != exacte and _type(x) == "personne":
            poids.setdefault(x, _liens(x))
    try:
        return _domine(poids)
    except Ambigu:
        return []


# --- Palier 2 : les candidats approximatifs, réunis puis départagés. ---

def _morceau_de_personne(nom: str, groupe: list[str]) -> list[str]:
    """« hugo », « van gogh », « louis ix », « wolfgang mozart »."""
    _, _, index = _base()
    q = nom.split()
    if not q or (len(q) == 1 and len(q[0]) < 3):
        return []
    trouves = []
    for n, qids in index.items():
        m = n.split()
        if len(m) <= len(q):
            continue
        famille = m[-len(q):] == q and not REGNE.match(q[-1])  # « Paul II » n'est pas « Jean-Paul II »
        # Seulement avec le numéro : « Louis » seul désigne cinquante rois.
        souverain = m[:len(q)] == q and bool(REGNE.match(q[-1]))
        sans_milieu = len(q) >= 2 and m[0] == q[0] and m[-1] == q[-1] and _dans_l_ordre(q, m)
        if famille or souverain or sans_milieu:
            trouves += [x for x in qids if _type(x) == "personne" and _a(x, groupe)]
    return trouves


def _par_la_fin(nom: str, groupe: list[str]) -> list[str]:
    """« ventoux » -> « mont ventoux » ; « ulm » -> « bataille d'ulm » ; « e t » -> « E.T., l'extra-terrestre »."""
    _, _, index = _base()
    courts = [x for r in groupe if not r.startswith("@") for x in _titres_courts(r).get(nom, []) if _a(x, groupe)]
    if len(nom) < 3 or (" " not in nom and _frequences().get(nom, 0) > 15):
        return courts
    trouves = []
    for n, qids in index.items():
        if n.endswith(" " + nom) and n != nom:
            avant = n[: -len(nom) - 1].split()[-1]
            # Trois lettres : seulement après « de » (« bataille d'ulm »), pas n'importe quel nom en « ... ulm ».
            if len(nom) == 3 and avant not in ("de", "d", "du", "des"):
                continue
            trouves += [x for x in qids if _type(x) in PAR_LA_FIN and _a(x, groupe)
                        and not (_type(x) == "ville" and avant in LIAISONS)]
    return trouves + courts


def _approches(nom: str, groupe: list[str]) -> list[str]:
    """Alias, morceaux de nom, fins de nom : le premier seulement s'il domine."""
    _, prenoms, pseudos = _tous_les_alias()
    alias = [x for x in _alias().get(nom, []) if _a(x, groupe)]
    autres = [x for x in _morceau_de_personne(nom, groupe) + _par_la_fin(nom, groupe) if x not in alias]
    q = nom.split()
    if q and REGNE.match(q[-1]):
        # « Louis IX » : le souverain dont c'est le nom, pas un alias du pape
        # François. S'ils sont plusieurs à égalité (François Ier de France,
        # d'Autriche, du Saint-Empire), on ne devine pas.
        souverains = [x for x in alias + autres if normaliser(_base()[0][x]["nom"]).startswith(nom)]
        if souverains:
            autres, alias = [x for x in souverains if x not in alias], [x for x in souverains if x in alias]
    # Un alias compte double (« Chine » désigne d'abord la république populaire),
    # sauf s'il n'est qu'un nom de famille : même indice qu'un morceau de nom
    # (« Clinton » : alias de Hillary, nom de Bill).
    poids = {x: (1 if " " not in nom and _type(x) == "personne" else 2) * _liens(x) for x in alias}
    for x in autres:
        poids.setdefault(x, _liens(x))
    # Un prénom seul (« Dante ») : il ne gagne que s'il est rare comme prénom
    # et qu'il écrase les autres candidats (Dante Alighieri contre Joe Dante).
    # « Henri » : des dizaines de personnes s'appellent ainsi, on se tait.
    candidats_prenoms = {x: _liens(x) for x in prenoms.get(nom, []) if _a(x, groupe) and x not in poids}
    if candidats_prenoms:
        if _premiers_mots().get(nom, 0) > 5:
            raise Ambigu
        gagnant = _domine(poids | candidats_prenoms, facteur=2 * DOMINE)
        if gagnant[0] in candidats_prenoms:
            return gagnant
    if not poids:
        return [x for x in pseudos.get(nom, []) if _a(x, groupe)][:1]  # « Saint-Just » : seulement faute de mieux
    return _domine(poids)


# --- Palier 3 : un mot générique en trop devant. ---

def _mot_en_trop(nom: str, groupe: list[str]) -> list[str]:
    _, _, index = _base()
    reste = EN_TETE.sub("", nom, count=1)
    if reste == nom or len(reste) < 3:
        return []
    return [x for x in index.get(reste, []) if _type(x) != "personne" and _a(x, groupe)]


# --- Paliers 4 et 5 : les fautes de frappe. ---

def _resoudre(nom: str, groupe: list[str]) -> list[str]:
    """Ce qu'un nom bien écrit désigne, par les mêmes paliers (exact, puis approché)."""
    exacts = _exacts(nom, groupe)
    if exacts:
        if _type(exacts[0]) == "personne" and len(nom.split()) == 1:
            return _personne_nette(nom, groupe, exacts[0])
        return exacts[:1]
    return _approches(nom, groupe)


@lru_cache(maxsize=4096)
def _mot_le_plus_proche(mot: str) -> tuple[str, int] | None:
    """
    Le mot du vocabulaire le plus proche, s'il est SEUL à ce nombre de fautes :
    (mot, fautes). « audten » est à une faute d'« austen » et d'« auden » : on
    ne choisit pas.
    """
    proches = difflib.get_close_matches(mot, list(_vocabulaire()), n=8, cutoff=0.75)
    classes = sorted((_dl(mot, p), p) for p in proches)
    if not classes or (len(classes) > 1 and classes[0][0] == classes[1][0]):
        return None
    return classes[0][1], classes[0][0]


def _proches(nom: str, groupe: list[str]) -> list[str]:
    if len(nom) < 4:
        return []
    if " " not in nom:
        if _frequences().get(nom, 0) >= 2:
            return []  # un mot courant n'est pas une faute de frappe (« reine » -> Répine)
        # Un mot seul mal tapé désigne ce que désigne le mot juste le plus proche :
        # « kennedi » -> ce que « kennedy » désigne (JFK), pas le violoniste Nigel
        # Kennedy ; « cinton » -> rien, comme « clinton ». Pour les petites listes
        # (197 pays), la comparaison sur la liste elle-même suit.
        # Moins de 6 lettres, c'est deviner : « twin », « bogr », « nada » sont
        # peut-être d'autres noms. Et si le mot juste est ambigu (« cinton » ->
        # « clinton »), _resoudre lève Ambigu : on se tait comme pour lui.
        mot = _mot_le_plus_proche(nom) if len(nom) >= 6 else None
        if mot and mot[1] <= (2 if len(nom) > 7 else 1) and mot[0][0] == nom[0]:
            trouves = _resoudre(mot[0], groupe)
            if trouves:
                return trouves
    noms: dict[str, list[str]] = {}
    for r in groupe:
        if r.startswith("@"):
            continue
        for n, qs in _noms_avec(r).items():
            noms.setdefault(n, []).extend(q for q in qs if q not in noms.get(n, []) and _a(q, groupe))
    # Peu de candidats (197 pays) : un peu plus de tolérance (« slovaki » ->
    # slovaquie, 0,75). Parmi 30 000 personnes, non (« kevin pacini » -> kevin
    # bacon, 0,78).
    seuil = 0.75 if len(noms) <= 1000 else 0.85
    colle = nom.replace(" ", "")
    signifiants = lambda x: [m for m in _en_chiffres(x).split() if len(m) > 2 or m.isdigit()]  # noqa: E731
    garde: dict[str, tuple[int, float]] = {}
    for n in difflib.get_close_matches(nom, list(noms), n=8, cutoff=seuil):
        if n == nom:
            continue  # le nom exact n'est pas une faute de frappe : réglé plus haut
        ratio = difflib.SequenceMatcher(None, nom, n).ratio()
        if ratio < 0.85 and (n[:3] != nom[:3] or len(nom) < 5):  # « nord » n'est pas « Norv(ège) »
            continue
        if n[0] != nom[0] or _chiffres(n) != _chiffres(nom):
            continue  # « drome » n'est pas « rome » ; « apollo 11 » pas « apollo 18 »
        # « charliechaplin », « cote divoire » : un espace en plus ou en moins
        espace = difflib.SequenceMatcher(None, colle, n.replace(" ", "")).ratio() >= 0.9
        if len(signifiants(n)) != len(signifiants(nom)) and not espace:
            continue
        # Le nom officiel avant l'alias d'une autre entité (« charle ier » : Charles Ier,
        # pas Charlemagne, dont un alias est « Charles Ier »).
        officiels = [x for x in noms[n] if normaliser(_base()[0][x]["nom"]) == n]
        for x in officiels or noms[n]:
            if _type(x) == "personne":
                a, b = nom.split(), n.split()
                if len(a) == 1 and len(nom) < 6:
                    continue  # un nom court d'un mot : trop risqué de deviner
                if len(b) == 1 and x not in _resoudre(n, groupe):
                    continue  # « cinton » -> « clinton » : ce que ce mot désigne (ici, ambigu : Ambigu est levé)
                if len(a) == len(b):
                    if not all(_une_faute(u, v) for u, v in zip(a, b)):
                        continue  # « lea martin » n'est pas « dean martin »
                elif not espace:
                    continue
            elif len(n.split()) > 1 and len(nom.split()) > 1 and not _une_faute(nom.split()[-1], n.split()[-1]):
                continue  # la guerre d'Algérie n'est pas celle d'Alexandrie
            if _type(x) == "ville" and _liens(x) < 100:
                continue  # une faute pour un village : plutôt un autre lieu absent de la base
            garde[x] = min(garde.get(x, (99, 0.0)), (_dl(colle, n.replace(" ", "")), -ratio))
    if not garde:
        return []
    # Le plus proche (en nombre de fautes) ; à égalité, le plus connu seulement s'il domine.
    classes = sorted(garde, key=lambda x: (garde[x], -_liens(x)))
    premier = classes[0]
    ex_aequo = [x for x in classes[1:] if garde[x][0] == garde[premier][0]]
    if ex_aequo and _liens(premier) < DOMINE * max(_liens(x) for x in ex_aequo):
        return []
    return [premier]


def _corrige(nom: str, groupe: list[str]) -> list[str]:
    """« nietzche » -> « nietzsche » : un long mot inconnu, une seule correction possible."""
    vocabulaire = _vocabulaire()
    mots = []
    for m in nom.split():
        if m not in vocabulaire and len(m) >= 7 and not m.isdigit():
            proches = difflib.get_close_matches(m, list(vocabulaire), n=2, cutoff=0.85)
            ratios = [difflib.SequenceMatcher(None, m, p).ratio() for p in proches]
            if proches and (len(proches) == 1 or ratios[0] > ratios[1]):
                m = proches[0]
        mots.append(m)
    corrige = " ".join(mots)
    if corrige == nom:
        return []
    # Le nom corrigé, résolu comme un nom bien écrit ; pas par sa fin ni un titre
    # court (« spotify » -> « sportif » -> « Le Coq sportif »).
    trouves = _resoudre(corrige, groupe)
    return [x for x in trouves if x in _base()[2].get(corrige, []) or _type(x) == "personne"
            or x in _alias().get(corrige, []) or x in _index_chiffres().get(_en_chiffres(corrige), [])]


# --- La recherche. ---

def _valeur(qid: str, r: str) -> str | None:
    entites, faits, index = _base()
    valeurs = faits.get(qid, {}).get(r)
    if not valeurs:
        return None
    if r in LIEUX and len(valeurs) > 1:  # « Varsovie », pas la rue « Ulica Freta »
        connus = [v for v in valeurs if normaliser(v) in index]
        valeurs = sorted(connus, key=lambda v: -entites[index[normaliser(v)][0]]["liens"])[:1] or valeurs[:1]
    if r in DATES:  # plusieurs dates de sortie : la première
        valeurs = [min(valeurs, key=_annee)]
    if r in NOMBRES:  # deux recensements : un seul chiffre
        valeurs = valeurs[:1]
    valeurs = valeurs[:3]
    return valeurs[0] if len(valeurs) == 1 else ", ".join(valeurs[:-1]) + " et " + valeurs[-1]


def _valeur_du_groupe(qid: str, groupe: list[str]) -> str | None:
    for r in (g for g in groupe if not g.startswith("@")):
        if (v := _valeur(qid, r)) is not None:
            return v
    return None


def chercher(entite: str, relation: str) -> str | None:
    """« Espagne », « capitale » -> « Madrid » ; None si la base ne sait pas, ou pas sûrement."""
    trouve = trouver(entite, relation)
    return _valeur_du_groupe(*trouve) if trouve else None


def trouver(entite: str, relation: str) -> tuple[str, list[str]] | None:
    """L'entité trouvée (son identifiant Wikidata) et le groupe de relations qui a répondu, ou None."""
    if not disponible():
        return None
    try:
        return _chercher(entite, relation)
    except Ambigu:
        return None


def entite(qid: str) -> dict:
    """Nom, type, notoriété d'une entité, et ses faits."""
    entites, faits, _ = _base()
    return entites[qid] | {"faits": faits.get(qid, {})}


def _chercher(entite: str, relation: str) -> tuple[str, list[str]] | None:
    nom, cle = normaliser(entite), normaliser(relation)
    groupes = RELATIONS.get(cle, [[relation.strip().lower()]])
    if not nom:
        return None
    entites, faits, index = _base()
    toutes = [r for g in groupes for r in g if not r.startswith("@")]
    # Ce que le nom désigne exactement (nom officiel, ou alias qui reprend un de
    # ses mots) : si c'est connu et sans la relation, un candidat moins connu
    # trouvé approximativement n'est pas la réponse (« Jésus | naissance » n'est
    # pas Gabriel Jesus, « Paris | naissance » pas Geneviève de Paris). Sauf un
    # lieu, une montagne ou un événement par la fin de son nom : « Verdun |
    # début » est la bataille, même si la ville est plus connue.
    exacts = index.get(nom, []) + [x for x in _alias().get(nom, [])
                                   if set(nom.split()) & set(normaliser(entites[x]["nom"]).split())]
    sans = [x for x in exacts if not any(r in faits.get(x, {}) for r in toutes)]
    plafond = max((_liens(x) for x in sans), default=0)
    if any(_type(x) == "pays" for x in sans) and len(sans) == len(exacts):
        return None  # « Portugal | naissance » n'est pas la naissance d'un roi de Portugal
    for groupe in groupes:
        if groupe == ["description"]:
            # « Qui est Napoléon ? » : parmi tout ce que ce nom désigne (le film
            # « Napoléon » de 2023, Napoléon Ier par son alias...), le nettement plus
            # connu, sinon rien. Le nom exact ne suffit pas.
            tous = [x for x in index.get(nom, []) + _alias().get(nom, []) + _morceau_de_personne(nom, groupe)
                    + _par_la_fin(nom, groupe) if _a(x, groupe)]
            if tous:
                return (_domine({x: _liens(x) for x in tous})[0], groupe)
            for palier in (_proches, _corrige):
                if candidats := [x for x in palier(nom, groupe) if _liens(x) >= plafond]:
                    return (max(candidats, key=_liens), groupe)
            return None
        exact = _exacts(nom, groupe)
        if exact:
            e0 = exact[0]
            # « Hiroshima | date » : pas le livre (14 liens), bien moins connu que la ville (159).
            plafond_nom = max((_liens(x) for x in index.get(nom, []) if x in sans), default=0)
            if _liens(e0) * DOMINE < plafond_nom:
                return None
            # « Waterloo | date » : la bataille (1815), bien plus connue que la chanson d'ABBA.
            if any(r in DATES for r in groupe) and _type(e0) != "événement":
                batailles = [x for x in _par_la_fin(nom, groupe) if _type(x) == "événement"]
                if batailles and max(_liens(x) for x in batailles) >= DOMINE * _liens(e0):
                    return (max(batailles, key=_liens), groupe)
            if _type(e0) == "personne" and (len(nom.split()) == 1 or REGNE.match(nom.split()[-1])):
                nette = _personne_nette(nom, groupe, e0)
                return (nette[0], groupe) if nette else None
            return (e0, groupe)
        courts = {x for r in groupe for x in _titres_courts(r).get(nom, [])}
        for palier in (_approches, _mot_en_trop, _proches, _corrige):
            # Pour une faute de frappe ou un titre court, le plafond vaut pour tous :
            # « Paris | auteur » n'est pas une faute pour « Le Pari », bien moins
            # connu que la ville ; « Rome | date » n'est pas « Rome, ville ouverte ».
            faute = palier in (_proches, _corrige)
            candidats = [x for x in palier(nom, groupe)
                         if (_type(x) != "personne" and not faute and x not in courts) or _liens(x) >= plafond]
            if candidats:
                return (max(candidats, key=_liens), groupe)
    return None
