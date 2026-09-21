#!/usr/bin/env bash
# Mise en place d'un pod RunPod fraîchement démarré (template PyTorch).
# À lancer une fois par pod, depuis le dossier /workspace (le volume persistant).
#
#   bash scripts/runpod_setup.sh
#
# Ensuite, TOUJOURS dans tmux :
#   tmux new -s train
#   python train.py --config configs/run_150m.py
#   (Ctrl-b puis d pour détacher, `tmux attach -t train` pour revenir)

set -euo pipefail

cd /workspace
if [ ! -d mon-llm ]; then
  git clone https://github.com/vincepanik/mon-llm.git
fi
cd mon-llm
git pull

# Pas de paquet flash-attn : use_flash_attn passe par scaled_dot_product_attention
# de PyTorch, qui embarque déjà FlashAttention sur CUDA. Compiler flash-attn
# prendrait jusqu'à une heure de GPU payé pour rien.
pip install -q -r requirements.txt

# Les données et checkpoints vivent sur le volume, pas dans le repo
# (les .bin du vrai run, fabriqués sur le Mac par data/prepare.py, s'envoient une
# seule fois dans /workspace/data ; configs/run_150m.py les lit dans data/big/)
mkdir -p /workspace/data /workspace/checkpoints data/big
ln -sfn /workspace/data/train.bin data/big/train.bin
ln -sfn /workspace/data/val.bin data/big/val.bin
ln -sfn /workspace/checkpoints checkpoints

nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), 'bf16', torch.cuda.is_bf16_supported())"
echo "prêt. Lance dans tmux."
