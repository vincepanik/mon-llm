"""
Conversations et votes de compar:IA (ministère de la Culture, licence Etalab 2.0)
pour le Carl de 125M : SFT, puis DPO.

    hf auth login     # une fois : les deux jeux de données sont à accès contrôlé
    python data/download_comparia.py

Écrit :
  data/sft/comparia.json      conversations pour le SFT (même format que
                              data/sft/conversations.json)
  data/sft/comparia_dpo.json  paires {"prompt", "chosen", "rejected"} pour le DPO

compar:IA fait discuter des Français avec deux modèles à la fois, puis leur
demande lequel a le mieux répondu. On garde :
  - le français seulement ;
  - les réponses de modèles ouverts sous licence permissive (Apache 2.0, MIT) :
    les conditions des modèles fermés (GPT, Claude, Gemini...) limitent
    l'entraînement d'autres modèles sur leurs réponses ;
  - les conversations qui tiennent entières dans le contexte de Carl (512 tokens
    sur le Mac) : les réponses courtes conviennent mieux à un petit modèle.

Réutilisation sous licence Etalab 2.0 : citer « compar:IA, ministère de la Culture ».
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chat_format import encoder_conversation  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

# Préfixes de noms de modèles à poids ouverts sous licence Apache 2.0 ou MIT.
# (ministral-8b est sous licence de recherche Mistral, mistral-large et
# mistral-medium sont fermés, command-r et aya sont non commerciaux : exclus.)
PERMISSIFS = (
    "mistral-small", "mistral-nemo", "mixtral", "phi-4", "deepseek-v3", "deepseek-r1",
    "qwen", "gpt-oss", "olmo", "lucie",
)


def permissif(nom: str) -> bool:
    return any(nom.lower().startswith(p) for p in PERMISSIFS)


def messages_propres(conv) -> list[dict] | None:
    """Garde user/assistant, en alternance, sans message vide ; None si inutilisable."""
    if isinstance(conv, str):
        conv = json.loads(conv)
    msgs = [{"role": m["role"], "content": (m.get("content") or "").strip()}
            for m in conv if m.get("role") in ("user", "assistant")]
    if len(msgs) < 2 or any(not m["content"] for m in msgs):
        return None
    if [m["role"] for m in msgs] != ["user", "assistant"] * (len(msgs) // 2) or len(msgs) % 2:
        return None
    return msgs


AUTRE_MODELE = re.compile(
    r"\b(qwen|alibaba|mistral|mixtral|phi-?\d|microsoft|deepseek|openai|chatgpt|gpt-?\d|anthropic|"
    r"claude|gemini|google|meta ai|llama|olmo|lucie)\b", re.I)
PRESENTATION = re.compile(r"\b(je suis|je m'appelle|mon nom est|en tant qu['e])\b", re.I)


def se_presente_comme_un_autre(msgs: list[dict]) -> bool:
    """
    L'assistant se présente et cite un modèle ou une entreprise (« Je suis Qwen,
    créé par Alibaba Cloud »). 1,2 % des conversations retenues, mais trois fois
    plus que les exemples d'identité de Carl : après un premier SFT, il
    répondait « Je suis Qwen » à « qui es-tu ? ». On les écarte toutes.
    """
    reponses = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
    return bool(AUTRE_MODELE.search(reponses) and PRESENTATION.search(reponses))


def francais(langues) -> bool:
    if isinstance(langues, str):
        langues = json.loads(langues)
    return list(langues or []) == ["fr"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=512, help="longueur maximale, tokens de Carl")
    parser.add_argument("--max-sft", type=int, default=30000)
    parser.add_argument("--max-dpo", type=int, default=10000)
    args = parser.parse_args()

    tok = BPETokenizer.load("tokenizer/vocab.json")
    tient = lambda msgs: len(encoder_conversation(tok, msgs)[0]) <= args.max_tokens  # noqa: E731

    # --- SFT : chaque ligne contient deux conversations, une par modèle.
    chemin = hf_hub_download("ministere-culture/comparia-conversations", "conversations.parquet",
                             repo_type="dataset")
    colonnes = ["languages", "model_a_name", "model_b_name", "conversation_a", "conversation_b"]
    sft, vues = [], set()
    for lot in pq.ParquetFile(chemin).iter_batches(batch_size=5000, columns=colonnes):
        for ligne in lot.to_pylist():
            if not francais(ligne["languages"]):
                continue
            for cote in "ab":
                if not permissif(ligne[f"model_{cote}_name"]):
                    continue
                msgs = messages_propres(ligne[f"conversation_{cote}"])
                if msgs and msgs[0]["content"] not in vues and tient(msgs) and not se_presente_comme_un_autre(msgs):
                    vues.add(msgs[0]["content"])  # une seule réponse par question
                    sft.append(msgs)
    print(f"SFT : {len(sft):,} conversations retenues", flush=True)

    # --- DPO : un tour, deux modèles ouverts, pas d'égalité ; « chosen » est
    # la réponse du modèle choisi par l'utilisateur.
    chemin = hf_hub_download("ministere-culture/comparia-votes", "votes.parquet", repo_type="dataset")
    colonnes = ["chosen_model_name", "both_equal", "conv_turns", "model_a_name", "model_b_name",
                "conversation_a", "conversation_b"]
    dpo = []
    for lot in pq.ParquetFile(chemin).iter_batches(batch_size=5000, columns=colonnes):
        for v in lot.to_pylist():
            if str(v["both_equal"]).lower() == "true" or str(v["conv_turns"]) != "1":
                continue
            if not (permissif(v["model_a_name"]) and permissif(v["model_b_name"])):
                continue
            a, b = messages_propres(v["conversation_a"]), messages_propres(v["conversation_b"])
            if not a or not b or a[0]["content"] != b[0]["content"] or not francais_texte(a[0]["content"]):
                continue
            gagnant, perdant = (a, b) if v["chosen_model_name"] == v["model_a_name"] else (b, a)
            if tient(gagnant) and tient(perdant) and not (se_presente_comme_un_autre(gagnant) or se_presente_comme_un_autre(perdant)):
                dpo.append({"prompt": gagnant[:1], "chosen": gagnant[1]["content"], "rejected": perdant[1]["content"]})
    print(f"DPO : {len(dpo):,} paires retenues", flush=True)

    rng = random.Random(1337)
    rng.shuffle(sft)
    rng.shuffle(dpo)
    sortie = Path("data/sft")
    sortie.mkdir(parents=True, exist_ok=True)
    (sortie / "comparia.json").write_text(json.dumps(sft[: args.max_sft], ensure_ascii=False), encoding="utf-8")
    (sortie / "comparia_dpo.json").write_text(json.dumps(dpo[: args.max_dpo], ensure_ascii=False), encoding="utf-8")
    print(f"écrit : {min(len(sft), args.max_sft):,} conversations, {min(len(dpo), args.max_dpo):,} paires -> {sortie}")
    sys.stdout.flush()
    os._exit(0)


def francais_texte(texte: str) -> bool:
    """Le fichier des votes n'a pas de colonne de langue : un filtre grossier sur les mots courants."""
    mots = texte.lower().split()
    communs = {"le", "la", "les", "de", "des", "un", "une", "est", "et", "en", "que", "qui", "pour", "je", "tu", "comment", "quel", "quelle"}
    return len(mots) > 0 and sum(m in communs for m in mots) / len(mots) > 0.08 or len(mots) <= 4


if __name__ == "__main__":
    main()
