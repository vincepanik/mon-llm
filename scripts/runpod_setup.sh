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
  git clone https://github.com/TON_COMPTE/mon-llm.git
fi
cd mon-llm
git pull

pip install -q -r requirements.txt
pip install -q flash-attn --no-build-isolation || echo "flash-attn non installé, use_flash_attn restera sans effet"

# Les données et checkpoints vivent sur le volume, pas dans le repo
mkdir -p /workspace/data /workspace/checkpoints
ln -sfn /workspace/data/train.bin data/train.bin
ln -sfn /workspace/data/val.bin data/val.bin
ln -sfn /workspace/checkpoints checkpoints

nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), 'bf16', torch.cuda.is_bf16_supported())"
echo "prêt. Lance dans tmux."
