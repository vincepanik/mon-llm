#!/usr/bin/env bash
# Mise en place d'un pod RunPod fraîchement démarré (template PyTorch).
# À lancer à chaque démarrage du pod (il ne refait que ce qui manque) :
#
#   cd /workspace && git clone https://github.com/vincepanik/mon-llm.git
#   bash /workspace/mon-llm/scripts/runpod_setup.sh
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

# Checkpoints : le dépôt est cloné dans /workspace, qui EST le volume persistant,
# donc checkpoints/ y survit déjà à l'arrêt du pod. Pas de lien à faire.
#
# Données : les .bin, fabriqués sur le Mac par data/prepare.py, s'envoient une
# seule fois dans /workspace/data (le volume). Mais l'entraînement y lit sans
# arrêt des petits morceaux pris au hasard, ce qu'un volume réseau fait mal :
# on les recopie sur le disque local du conteneur (un SSD, effacé à l'arrêt du
# pod, d'où la recopie à chaque démarrage). configs/run_150m.py les lit dans data/big/.
mkdir -p /workspace/data data/big
if [ -f /workspace/data/train.bin ] && [ -f /workspace/data/val.bin ]; then
  mkdir -p /root/data
  echo "copie des .bin sur le disque local..."
  cp -u /workspace/data/train.bin /workspace/data/val.bin /root/data/
  ln -sfn /root/data/train.bin data/big/train.bin
  ln -sfn /root/data/val.bin data/big/val.bin
  ls -laL data/big/
else
  echo "ATTENTION : pas encore de train.bin / val.bin dans /workspace/data."
  echo "Envoie-les depuis le Mac, puis relance ce script."
fi

nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), 'bf16', torch.cuda.is_bf16_supported())"
echo "prêt. Lance dans tmux."
