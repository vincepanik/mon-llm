"""
Affinage LoRA de Gemma 4 E4B pour en faire Carl (niveau 4), sur le Mac avec MLX.

    python carl_gemma/identite.py
    python carl_gemma/distiller.py --n 100
    python carl_gemma/entrainer.py

LoRA : on gèle les ~8 milliards de paramètres de Gemma et on n'entraîne qu'une
petite couche d'adaptation greffée sur ses couches (moins de 1 % des poids).
Ça tient dans 16 Go, et le résultat (carl_gemma/adapters/) pèse quelques
dizaines de Mo, posé par-dessus Gemma au chargement.

Données : les conversations d'identité (répétées, elles sont peu nombreuses)
mélangées aux réponses générales écrites par Gemma lui-même, pour qu'il ne
change que d'identité (voir distiller.py).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import mlx_lm.lora
import mlx_lm.tuner.datasets as datasets

MODELE = "mlx-community/gemma-4-E4B-it-qat-4bit"
DOSSIER = Path("carl_gemma")


def sans_reflexion() -> None:
    """
    mlx-lm met les conversations en forme sans préciser le mode, et le modèle
    de conversation de Gemma 4 active alors la réflexion (<|think|>). On
    utilise Carl sans elle : on entraîne donc sans elle aussi.
    """
    def process(self, d):
        messages = d[self.chat_key]
        tokens = self.tokenizer.apply_chat_template(messages, return_dict=False, enable_thinking=False)
        if not self.mask_prompt:
            return (tokens, 0)
        invite = self.tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, return_dict=False, enable_thinking=False
        )
        return (tokens, len(invite))

    datasets.ChatDataset.process = process


def preparer_donnees(repetitions: int, seed: int) -> None:
    identite = json.loads((DOSSIER / "identite.json").read_text(encoding="utf-8"))
    general = json.loads((DOSSIER / "general.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)
    rng.shuffle(general)
    n_val = max(5, len(general) // 10)
    # Validation : des réponses générales jamais vues, pour vérifier que Gemma
    # n'oublie pas son comportement, et quelques questions d'identité.
    valid = general[:n_val] + rng.sample(identite, 5)
    train = general[n_val:] + identite * repetitions
    rng.shuffle(train)
    sortie = DOSSIER / "data"
    sortie.mkdir(exist_ok=True)
    for nom, lignes in (("train", train), ("valid", valid)):
        with open(sortie / f"{nom}.jsonl", "w", encoding="utf-8") as f:
            for ex in lignes:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"entraînement : {len(train)} conversations ({len(identite)} d'identité x{repetitions}, "
          f"{len(general) - n_val} générales) | validation : {len(valid)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=250)
    parser.add_argument("--repetitions", type=int, default=1)
    # Réglages légers : avec lr 1e-4 sur 16 couches, 400 pas et l'identité
    # répétée 2 fois, Gemma avait « oublié » (17 x 23 = 1789, « Non » à
    # « quelle heure est-il ? »). carl_gemma/examen.py mesure les deux côtés.
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--adapters", default=str(DOSSIER / "adapters"))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    preparer_donnees(args.repetitions, args.seed)
    sans_reflexion()
    sys.argv = [
        "mlx_lm.lora", "--model", MODELE, "--train", "--data", str(DOSSIER / "data"),
        "--adapter-path", args.adapters, "--mask-prompt",
        "--iters", str(args.iters), "--batch-size", "1", "--num-layers", str(args.num_layers),
        "--learning-rate", str(args.lr), "--max-seq-length", "1024", "--grad-checkpoint",
        "--steps-per-eval", "50", "--val-batches", "25", "--save-every", "50", "--seed", str(args.seed),
    ]
    mlx_lm.lora.main()


if __name__ == "__main__":
    main()
