"""
Discuter avec le modèle après le SFT.

    python chat.py --checkpoint checkpoints/sft/best.pt
    python chat.py --checkpoint checkpoints/sft/best.pt --question "Qu'est-ce que la photosynthèse ?"

Sans --question : conversation au clavier. Entrée vide pour quitter. Chaque
conversation est enregistrée au fil de l'eau dans conversations/, un fichier
Markdown par session (--sans-journal pour ne rien enregistrer).

Mémoire : Carl ne voit l'échange précédent que pour une relance (« et de la
France ? », « pourquoi ? »), jamais sinon. Mesuré sur une conversation de 6
questions rejouée 3 fois : sans mémoire, il répond juste aux questions
d'identité de la fin 6 fois sur 6 ; avec l'échange précédent sous les yeux,
2 fois sur 6 seulement. À 125M paramètres, le réflexe de recopier ce qui est
dans le contexte l'emporte sur le fil de la conversation, sauf quand la
question n'a aucun sens sans lui. --memoire N impose les N échanges précédents.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import torch

from chat_format import debut_de_reponse
import faits
from outils import APPEL, APPEL_FAIT, afficher, calculer
from model import GPT
from tokenizer import BPETokenizer
from utils import get_device, load_checkpoint


def avec_document(messages: list[dict]) -> list[dict]:
    """
    Cherche dans Wikipédia (rag.py) le passage le plus proche de la dernière
    question et le place juste avant elle, comme dans les exemples de lecture
    de l'entraînement (data/lecture.py). Carl y puise la réponse s'il la trouve,
    et l'ignore sinon. Sans index, les messages passent tels quels.
    """
    import rag

    if not rag.disponible() or not messages or messages[-1]["role"] != "user":
        return messages
    if not rag.utile(messages[-1]["content"]):
        return messages
    trouves = rag.chercher(messages[-1]["content"], k=1)
    if not trouves:
        return messages
    return messages[:-1] + [{"role": "document", "content": trouves[0][0]}, messages[-1]]


# Débuts de phrase qui font d'une question la suite de la précédente. Pas
# « ou » : sans accent, c'est souvent « où » tapé vite (« ou est nee marie
# curie »), une question complète.
RELANCE = re.compile(r"^\s*(et|mais|alors|donc|pourquoi|comment ça|c'est-à-dire|sinon|ok et|d'accord et)\b", re.I)
POLITESSES = set("""merci beaucoup bien infiniment bonjour bonsoir salut coucou hello hey ok okay d accord au revoir
    bonne nuit journée soirée super génial parfait top cool sympa bravo excellent nickel vu compris oui non ça va""".split())
# Une question courte n'est une relance que si elle ne nomme rien : « pourquoi
# ? », « en quelle année ? », « et où ? ». « nantes pays ? » ou « population
# lyon » se suffisent à elles-mêmes ; avant, toute question de trois mots ou
# moins voyait l'échange précédent, souvent un simple « Hello Carl ! ».
SANS_SUJET = set("""et ou où quand pourquoi comment combien qui quoi que quel quelle quels quelles lequel laquelle
    en de du des la le les l d à a au aux est c ça ca il elle ils elles y t alors donc sinon année an ans date lieu
    depuis né née mort morte âge age habitants population superficie taille altitude capitale monnaie langue
    auteur réalisateur fondateur prix""".split())
# « sa population ? », « son âge ? », « où est-il né ? », « qui l'a écrit ? » :
# un possessif ou un pronom renvoie forcément à ce qui précède, même si la
# question nomme autre chose. Un pronom seulement à sa place de pronom : en tête
# de question, ou en minuscules (« qui a réalisé Elle ? » parle du film).
RENVOI = re.compile(r"\b(sa|son|ses|leur|leurs|celui|celle|ceux|celui-ci|celle-ci)\b|\bl'(a|ont|avait|avaient)\b", re.I)
# « décris-moi la recette 3 », « la 2 », « le premier », « la dernière » : un
# élément de la réponse précédente (souvent une liste), invisible sans elle.
ELEMENT = re.compile(r"\b(la|le|les|l')\s*(\w+\s)?(\d+|n°\s*\d+|numéro \d+|premi[eè]re?|deuxi[eè]me|second[e]?"
                     r"|troisi[eè]me|quatri[eè]me|cinqui[eè]me|derni[eè]re?)\b(?!\s*\w*\s*(de|du|des|d')\b)", re.I)
PRONOM = re.compile(r"(?:^|[\s-])(il|ils|elle|elles|lui|eux)(?=[\s?!.,-]|$)")
IMPERSONNEL = re.compile(r"\b(y a-t-il|heure est-il|fait-il|il y a|il faut|il pleut|il fait|s'il)\b", re.I)


def renvoie(question: str) -> bool:
    texte = question.strip()
    if IMPERSONNEL.search(texte):
        texte = IMPERSONNEL.sub(" ", texte)
    en_tete = re.match(r"(il|ils|elle|elles|lui|eux)\b", texte, re.I)
    return bool(RENVOI.search(texte) or PRONOM.search(texte) or en_tete)
A_CARL = re.compile(r"\b(tu|te|toi|ton|ta|tes|vous|votre|vos|carl)\b|\bt'|\bt’", re.I)


def est_relance(question: str) -> bool:
    """
    « et de la France ? » n'a pas de sens seule : il faut lui montrer l'échange
    précédent. Une question très courte aussi (« pourquoi ? », « en quelle
    année ? »). Jamais une question qui s'adresse à Carl : c'est là que la
    mémoire le faisait recopier le fil au lieu de répondre.
    """
    if A_CARL.search(question):
        return False
    mots = re.findall(r"\w+", question.lower())
    if mots and all(m in POLITESSES for m in mots):  # « merci », « bonjour », « ok »...
        return False
    courte = len(mots) <= 3 and all(m in SANS_SUJET for m in mots)
    element = len(mots) <= 8 and bool(ELEMENT.search(question))
    return bool(RELANCE.match(question)) or courte or element or (len(mots) <= 5 and renvoie(question))


def politesse(message: str) -> bool:
    """« Hello Carl ! », « merci ! », « ok » : rien à quoi une relance puisse se rattacher."""
    mots = re.findall(r"\w+", message.lower())
    return all(m in POLITESSES or m == "carl" for m in mots)


def a_montrer(messages: list[dict], memoire: int | None) -> list[dict]:
    """
    Ce que Carl voit de la conversation : la question, plus, pour une relance,
    le dernier vrai échange. Les politesses sont sautées : dans « Quelle
    langue en Argentine ? » / « merci ! » / « Et au Brésil ? », la relance porte
    sur la langue, pas sur le merci.
    """
    if memoire is not None:
        return messages[-(2 * memoire + 1):]
    if len(messages) < 3 or not est_relance(messages[-1]["content"]):
        return messages[-1:]
    for i in range(len(messages) - 3, -1, -2):  # les questions précédentes, de la plus récente
        # « tu es sûr ? » non plus : une question à Carl ne donne pas de sujet à une relance.
        texte = messages[i]["content"]
        if messages[i]["role"] == "user" and not politesse(texte) and not A_CARL.search(texte):
            return messages[i : i + 2] + messages[-1:]
    return messages[-1:]


def trigrammes_interdits(reponse: list[int], n: int = 3) -> list[int]:
    """
    Les tokens qui compléteraient une suite de n tokens déjà écrite dans la
    réponse. Contre les boucles (« "p" après "p" » répété vingt fois), que la
    pénalité de répétition, trop douce, laissait passer.
    """
    if n <= 0 or len(reponse) < n - 1:
        return []
    fin = tuple(reponse[-(n - 1):])
    return sorted({reponse[i + n - 1] for i in range(len(reponse) - n + 1) if tuple(reponse[i : i + n - 1]) == fin})


def dans_un_calcul(fin_de_reponse: str) -> bool:
    """Carl est-il en train d'écrire « [calc: ... » ou « [fait: ... », pas encore refermé ?"""
    return max(fin_de_reponse.rfind("[calc:"), fin_de_reponse.rfind("[fait:")) > fin_de_reponse.rfind("]")


_CROCHETS: dict[int, list[int]] = {}


def tokens_crochet(tok) -> list[int]:
    """Les tokens qui contiennent « [ » : les interdire empêche d'appeler un outil."""
    if id(tok) not in _CROCHETS:
        _CROCHETS[id(tok)] = [i for i in range(tok.vocab_size) if "[" in tok.decode([i])]
    return _CROCHETS[id(tok)]


# Réponses apprises pour « le passage ne contient pas la réponse » (data/lecture.py).
PAS_DANS_LE_DOCUMENT = ("ne le dit pas", "ne trouve pas cette information", "ne répond pas à cette question")


def repondre(model, tok, messages, device, temperature: float, top_k: int, max_tokens: int,
             repetition_penalty: float, avec_outils: bool = True, wikipedia: bool = False,
             sans_repetition: int = 3, brut: bool = False) -> str:
    """
    Avec wikipedia=True : on lui donne d'abord le passage trouvé ; s'il répond
    que le passage ne contient pas la réponse, on repose la question sans
    document, et il répond de mémoire. La lecture quand elle aide, la mémoire
    quand la recherche a ramené un passage à côté.

    brut=True garde les appels à la calculatrice (« [calc: 12*12 = 144] ») :
    c'est la version à remettre dans l'historique. Sinon, à la question
    suivante, Carl voit « 12 × 12 = 144 », imite une réponse sans calculatrice
    et invente (« 13 × 13 = 156 »).
    """
    reglages = dict(temperature=temperature, top_k=top_k, max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty, avec_outils=avec_outils,
                    sans_repetition=sans_repetition)
    r = None
    if wikipedia:
        # La base de faits d'abord : avec un passage sous les yeux, Carl
        # oubliait son outil et lisait mal (« Égypte. » pour sa capitale).
        r = _generer(model, tok, messages, device, **reglages)
        if "[fait:" in r:
            return r if brut else afficher(r).strip()
        r = None
        documentee = avec_document(messages)
        if documentee is not messages:
            r = _generer(model, tok, documentee, device, **reglages)
            if any(m in r.lower() for m in PAS_DANS_LE_DOCUMENT):
                r = None
    if r is None:
        r = _generer(model, tok, messages, device, **reglages)
    return r if brut else afficher(r).strip()


def repondre_aiguille(model, tok, historique: list[dict], device, memoire: int | None = None,
                      **reglages) -> tuple[str, str]:
    """
    Avec l'aiguilleur (aiguilleur.py) devant Carl : une réponse toute prête quand
    il est sûr de lui (salut, merci, identité, heure, liste de capitales...),
    l'échange précédent pour une relance, sinon Carl comme d'habitude.
    Renvoie (réponse brute, ce qui a décidé : « carl » ou la classe de l'aiguilleur).
    """
    import aiguilleur

    d = aiguilleur.decider(historique[-1]["content"])
    if d.reponse:
        return d.reponse, f"{d.classe} ({d.proba:.0%})"
    vus = a_montrer(historique, 1 if d.relance and len(historique) >= 3 else memoire)
    return repondre(model, tok, vus, device, brut=True, **reglages), "carl" + (" (relance)" if d.relance else "")


def _generer(model, tok, messages, device, temperature: float, top_k: int, max_tokens: int,
             repetition_penalty: float, avec_outils: bool = True, sans_repetition: int = 3) -> str:
    """
    Génère la réponse token par token, pour pouvoir intervenir en cours de route :
    dès que Carl écrit « [calc: <expression> = », la calculatrice (outils.py)
    fait le calcul et on insère le résultat à sa place. Carl continue ensuite
    sa phrase. Le texte rendu remplace « [calc: ... = résultat] » par le résultat.

    Même chose pour « [fait: Espagne | capitale = » avec la base de faits
    (faits.py). Si elle ne sait pas, on efface l'appel et on interdit à Carl
    d'en refaire un : il répond de mémoire, comme avant l'outil.
    """
    ids = debut_de_reponse(tok, messages)
    ids = ids[-(model.cfg.block_size - max_tokens):]  # garder de la place pour la réponse
    fin = tok.special_tokens["<|im_end|>"]
    # Ses propres réponses précédentes : un petit modèle a tendance à les
    # recopier mot pour mot, et une réponse ratée se répète alors en boucle.
    # Sans les appels d'outils : pénaliser le « [ » de « [fait: ... » d'une
    # réponse précédente décourageait Carl d'appeler l'outil pour une relance.
    precedentes = [t for m in messages if m["role"] == "assistant" for t in tok.encode(afficher(m["content"]))]
    reponse: list[int] = []
    # Après un fait, Carl recopie le résultat (« Saint-Exupéry ») : ni la
    # pénalité ni le blocage des trigrammes ne doivent porter sur l'appel.
    depuis = 0
    sans_outil = not avec_outils  # sans outils : il ne peut même pas commencer un appel
    insere = False  # le dernier token vient d'un outil
    while len(reponse) < max_tokens:
        contexte = torch.tensor([(ids + reponse)[-model.cfg.block_size:]], device=device)
        suivant = model.generate(
            contexte, 1,
            temperature=max(temperature, 1e-5), top_k=1 if temperature <= 0 else top_k,
            repetition_penalty=repetition_penalty, penaliser_aussi=precedentes + reponse[depuis:],
            # Pas pendant un appel à la calculatrice : il y recopie volontairement
            # l'opération (« 4827 + 3196 = [calc: 4827+3196 = »), et bloquer ce
            # second « = » l'empêchait d'appeler l'outil (il inventait 51469).
            interdits=([] if dans_un_calcul(tok.decode(reponse[-40:])) else trigrammes_interdits(reponse[depuis:], sans_repetition))
            + (tokens_crochet(tok) if sans_outil else []),
        )[0, -1].item()
        if insere:
            # Le programme a déjà refermé l'appel (« 391] ») ; à l'entraînement,
            # « ]. » ne faisait qu'un token et Carl veut encore l'écrire :
            # on garde le point, pas le second crochet.
            insere = False
            texte = tok.decode([suivant])
            if texte.startswith("]"):
                reponse += tok.encode(texte[1:])
                continue
        if suivant == fin:
            break
        reponse.append(suivant)
        if avec_outils:
            appel = APPEL.search(tok.decode(reponse[-40:]))
            if appel:
                reponse += tok.encode(f" {calculer(appel.group(1))}]")
                insere = True
                continue
            appel = APPEL_FAIT.search(tok.decode(reponse[-60:]))
            if appel:
                trouve = faits.chercher(appel.group(1), appel.group(2))
                if trouve:
                    reponse += tok.encode(f" {trouve}]")
                    depuis = len(reponse)
                    insere = True
                else:
                    debut = len(reponse) - 1
                    while debut > 0 and "[fait:" not in tok.decode(reponse[debut:]):
                        debut -= 1
                    del reponse[debut:]
                    sans_outil = True
    else:  # plus de place : on coupe à la dernière phrase complète, pas au milieu d'un mot
        return couper(tok.decode(reponse).strip())
    return sans_element_vide(tok.decode(reponse).strip())


def sans_element_vide(texte: str) -> str:
    """Une liste qui s'arrête sur un numéro seul (« 2. Ajoutez...\n3. ») : on retire « 3. »."""
    return re.sub(r"\n\s*(?:\d+[.)]|[-*•])\s*$", "", texte).rstrip()


def couper(texte: str) -> str:
    """« Il a dit. Puis il est par » -> « Il a dit. » ; sans phrase complète : « ... par… »."""
    # Un numéro de liste seul à la fin (« 7. Poulet curry.\n8. ») n'est pas une
    # phrase finie, même s'il se termine par un point : on retire l'élément vide.
    texte = re.sub(r"\n\s*(?:\d+[.)]|[-*•])\s*$", "", texte.rstrip())
    if texte.rstrip().endswith((".", "!", "?", "…")) and not dans_un_calcul(texte):
        return texte  # la dernière phrase est finie : rien à couper
    fin = max(texte.rfind(p) for p in (". ", "! ", "? ", ".\n", "!\n", "?\n"))
    if fin >= len(texte) * 0.4 and not dans_un_calcul(texte[: fin + 1]):
        return texte[: fin + 1]
    return texte if texte.endswith((".", "!", "?", "…")) else texte + "…"


MOIS = "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split()


class Journal:
    """
    Une conversation = un fichier Markdown dans conversations/, écrit à chaque
    échange (rien n'est perdu si on quitte avec Ctrl+C). Les appels à la
    calculatrice y sont affichés comme à l'écran, avec leur détail en italique.
    """

    def __init__(self, dossier: Path, checkpoint: str, reglages: dict):
        maintenant = datetime.now()
        dossier.mkdir(parents=True, exist_ok=True)
        self.chemin = dossier / maintenant.strftime("%Y-%m-%d_%Hh%M.md")
        n = 2
        while self.chemin.exists():  # deux sessions la même minute : ne pas écraser la première
            self.chemin = dossier / maintenant.strftime(f"%Y-%m-%d_%Hh%M_{n}.md")
            n += 1
        date = f"{maintenant.day} {MOIS[maintenant.month - 1]} {maintenant.year}, {maintenant:%H} h {maintenant:%M}"
        options = ", ".join(f"{k} {v}" for k, v in reglages.items())
        self.chemin.write_text(
            f"# Conversation avec Carl — {date}\n\n"
            f"*Modèle : `{checkpoint}` — {options}*\n\n---\n\n", encoding="utf-8")
        self.n = 0

    def ajouter(self, question: str, reponse_brute: str, vu: int, aiguillage: str | None = None) -> None:
        self.n += 1
        calculs = [m.group(0) for m in re.finditer(r"\[calc:[^\]]*\]", reponse_brute)]
        consultes = [m.group(0) for m in re.finditer(r"\[fait:[^\]]*\]", reponse_brute)]
        texte = f"**Vous** : {question}\n\n**Carl** : {afficher(reponse_brute).strip()}\n\n"
        notes = []
        if aiguillage:
            notes.append(f"réponse de l'aiguilleur : {aiguillage}")
        if vu > 1:
            notes.append("il voyait l'échange précédent (relance)")
        if calculs:
            notes.append("calculatrice : " + ", ".join(f"`{c}`" for c in calculs))
        if consultes:
            notes.append("base de faits : " + ", ".join(f"`{c}`" for c in consultes))
        if notes:
            texte += f"*({' ; '.join(notes)})*\n\n"
        with self.chemin.open("a", encoding="utf-8") as f:
            f.write(texte)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--question", default=None)
    # 0 = toujours le mot le plus probable. Mesuré sur 40 faits simples (Carl v4) :
    # 27,7/40 contre 23,3 à 0,7 et 25,7 à 0,3. Le hasard fait piocher à un petit
    # modèle des mots moins probables, donc souvent faux. Monter la température
    # pour des textes créatifs (poème, histoire).
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    # Quand il ne sait pas, il s'étale (des listes entières sur « la France » en
    # général) : couper plus tôt limite les dérives. Ses bonnes réponses tiennent
    # presque toujours en quelques phrases.
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--sans-repetition", type=int, default=3,
                        help="interdire de répéter une suite de N tokens dans la réponse (0 : désactivé)")
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    # Désactivée par défaut : mesuré sur 40 faits, Carl v6 en retrouve 25 de
    # mémoire et 21 avec la recherche. Elle ne ramène le bon passage qu'une fois
    # sur deux, et Carl ne sait pas reconnaître un passage voisin qui ne
    # contient pas la réponse (il y pioche une mauvaise réponse).
    parser.add_argument("--wikipedia", action="store_true",
                        help="chercher dans Wikipédia avant de répondre (rag.py), expérimental")
    parser.add_argument("--sans-outils", action="store_true",
                        help="ni calculatrice ni base de faits : Carl répond de mémoire")
    parser.add_argument("--sans-journal", action="store_true",
                        help="ne pas enregistrer la conversation dans conversations/")
    parser.add_argument("--sans-aiguilleur", action="store_true",
                        help="sans l'aiguilleur (aiguilleur.py) devant Carl : politesses, identité, heure, listes")
    parser.add_argument("--memoire", type=int, default=None,
                        help="échanges précédents montrés au modèle (par défaut : 1 pour une relance, 0 sinon)")
    args = parser.parse_args()

    device = get_device()
    ck = load_checkpoint(args.checkpoint, device)
    model = GPT(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = BPETokenizer.load(ck["config"].tokenizer_path)
    reglages = dict(temperature=args.temperature, top_k=args.top_k, max_tokens=args.max_tokens,
                    repetition_penalty=args.repetition_penalty, wikipedia=args.wikipedia,
                    sans_repetition=args.sans_repetition, avec_outils=not args.sans_outils)

    if args.question:
        print(repondre(model, tok, [{"role": "user", "content": args.question}], device, **reglages))
        return

    journal = None if args.sans_journal else Journal(
        Path("conversations"), args.checkpoint,
        {"température": args.temperature, "wikipedia": "oui" if args.wikipedia else "non"})
    if journal:
        print(f"(conversation enregistrée dans {journal.chemin})")
    import aiguilleur

    aiguille = not args.sans_aiguilleur and aiguilleur.disponible()
    if aiguille:
        aiguilleur.charger()
    messages: list[dict] = []
    while True:
        try:
            question = input("\nvous > ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D, Ctrl+C, ou pas de clavier du tout
            print()
            break
        if not question:
            break
        messages.append({"role": "user", "content": question})
        if aiguille:
            reponse, source = repondre_aiguille(model, tok, messages, device, args.memoire, **reglages)
            vu = 2 if "relance" in source else 1
        else:
            vus = a_montrer(messages, args.memoire)
            reponse, source, vu = repondre(model, tok, vus, device, brut=True, **reglages), "carl", len(vus)
        print(f"carl > {afficher(reponse).strip()}")
        messages.append({"role": "assistant", "content": reponse})
        if journal:
            journal.ajouter(question, reponse, vu, None if source.startswith("carl") else source)


if __name__ == "__main__":
    main()
