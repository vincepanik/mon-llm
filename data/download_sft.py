"""
Récupère les conversations françaises pour l'étape 5 (SFT) et les écrit dans
data/sft/conversations.json : une liste de conversations, chacune une liste de
{"role": "user" | "assistant", "content": "..."}.

    python data/download_sft.py

Sources, toutes écrites par des humains et sous licence Apache 2.0 :
  OpenAssistant/oasst2    conversations à plusieurs tours, notées par des
                          bénévoles ; on suit à chaque tour la réponse classée
                          meilleure. ~360 conversations en français.
  CohereLabs/aya_dataset  paires question / réponse rédigées par des
                          locuteurs natifs. ~1 400 en français.

C'est peu (les grands modèles en voient des millions), mais assez pour
apprendre au modèle le format d'une conversation : répondre à la question au
lieu de continuer le texte, puis s'arrêter.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

SORTIE = Path("data/sft/conversations.json")


def oasst2() -> list[list[dict]]:
    chemin = hf_hub_download(
        "OpenAssistant/oasst2", "2023-11-05_oasst2_ready.trees.jsonl.gz", repo_type="dataset"
    )
    convs = []
    with gzip.open(chemin, "rt", encoding="utf-8") as f:
        for ligne in f:
            arbre = json.loads(ligne)
            if arbre["prompt"]["lang"] != "fr":
                continue
            # À chaque tour, la réponse la mieux classée par les annotateurs (rank 0).
            noeud, chemin_conv = arbre["prompt"], []
            while noeud is not None:
                role = "user" if noeud["role"] == "prompter" else "assistant"
                chemin_conv.append({"role": role, "content": noeud["text"].strip()})
                reponses = [r for r in noeud.get("replies", []) if not r.get("deleted")]
                reponses.sort(key=lambda r: (r.get("rank") is None, r.get("rank") or 0))
                noeud = reponses[0] if reponses else None
            if chemin_conv[-1]["role"] != "assistant":
                chemin_conv = chemin_conv[:-1]
            if len(chemin_conv) >= 2:
                convs.append(chemin_conv)
    return convs


def aya() -> list[list[dict]]:
    ds = load_dataset("CohereLabs/aya_dataset", split="train", streaming=True)
    convs = []
    for ex in ds:
        if ex["language"] != "French":
            continue
        q, r = ex["inputs"].strip(), ex["targets"].strip()
        if q and r:
            convs.append([{"role": "user", "content": q}, {"role": "assistant", "content": r}])
    return convs


def main() -> None:
    a, b = oasst2(), aya()
    print(f"oasst2 : {len(a):,} conversations | aya : {len(b):,}", flush=True)

    # Doublons : même première question.
    vues, convs = set(), []
    for c in a + b:
        cle = c[0]["content"].lower()
        if cle not in vues:
            vues.add(cle)
            convs.append(c)
    random.Random(1337).shuffle(convs)

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(json.dumps(convs, ensure_ascii=False, indent=0), encoding="utf-8")
    car = sum(len(m["content"]) for c in convs for m in c)
    print(f"{len(convs):,} conversations, {car / 1e6:.2f} M caractères -> {SORTIE}", flush=True)

    # Même souci que data/download.py : datasets laisse des threads qui
    # empêchent Python de rendre la main.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
