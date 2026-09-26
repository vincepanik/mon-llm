"""
Mélange de modèles (« model soup », Wortsman et al., 2022) : la moyenne des
poids de plusieurs Carl affinés depuis le même point de départ, avec les mêmes
données mais des graines différentes. Chaque graine fait ses propres erreurs ;
la moyenne les lisse souvent, sans rien réentraîner.

    python melanger.py checkpoints/carl_v15/best.pt checkpoints/carl_v15_graine2/best.pt --out checkpoints/carl_v15_melange
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from utils import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cks = [load_checkpoint(c, torch.device("cpu")) for c in args.checkpoints]
    assert all(ck["config"] == cks[0]["config"] for ck in cks), "les modèles n'ont pas la même architecture"
    poids = {k: sum(ck["model"][k].float() for ck in cks) / len(cks) for k in cks[0]["model"]}
    poids = {k: v.to(cks[0]["model"][k].dtype) for k, v in poids.items()}
    sortie = Path(args.out) / "best.pt"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": poids, "config": cks[0]["config"], "melange": args.checkpoints}, sortie)
    print(f"moyenne de {len(cks)} modèles -> {sortie}")


if __name__ == "__main__":
    main()
