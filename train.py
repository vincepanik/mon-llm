"""
Boucle d'entraînement. Identique sur le Mac et sur une machine louée.

    python train.py --config configs/debug_mac.py
    python train.py --config configs/run_150m.py --resume

Tout ce qui est spécifique au matériel passe par la config et utils.py.
"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from model import GPT
from utils import (
    Throttle,
    count_params,
    get_device,
    get_dtype,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seed,
)


def get_batch(data: np.ndarray, cfg, device: torch.device):
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def get_lr(step: int, cfg) -> float:
    """Warmup linéaire puis décroissance cosinus jusqu'à min_lr."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def evaluate(model, data: np.ndarray, cfg, device, ctx) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_steps):
        x, y = get_batch(data, cfg, device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    dtype = get_dtype(device, cfg.use_bf16)
    ctx = torch.autocast(device_type=device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()
    print(f"device={device} dtype={dtype} run={cfg.run_name}")

    int_dtype = np.uint8 if cfg.vocab_size <= 256 else np.uint16
    train_data = np.memmap(cfg.data_dir / cfg.train_file, dtype=int_dtype, mode="r")
    val_data = np.memmap(cfg.data_dir / cfg.val_file, dtype=int_dtype, mode="r")

    model = GPT(cfg).to(device)
    optimizer = model.configure_optimizer(cfg, device)
    print(f"{count_params(model) / 1e6:.2f}M paramètres")

    step, best_val = 0, float("inf")
    latest = cfg.run_dir / "latest.pt"
    if args.resume and latest.exists():
        ck = load_checkpoint(latest, device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        step, best_val = ck["step"], ck["best_val"]
        print(f"reprise à l'étape {step}")

    if cfg.use_compile and device.type == "cuda":
        model = torch.compile(model)

    throttle = Throttle(cfg.throttle)
    t_last = time.time()

    while step < cfg.max_steps:
        throttle.start()

        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        for _ in range(cfg.grad_accum_steps):
            x, y = get_batch(train_data, cfg, device)
            with ctx:
                _, loss = model(x, y)
            (loss / cfg.grad_accum_steps).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.log_interval == 0:
            dt = time.time() - t_last
            t_last = time.time()
            print(f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt / cfg.log_interval * 1000:.0f} ms/step")

        if step % cfg.eval_interval == 0:
            val_loss = evaluate(model, val_data, cfg, device, ctx)
            print(f"step {step:6d} | val {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(cfg.run_dir / "best.pt", model, optimizer, step, cfg, best_val)

        if step % cfg.checkpoint_interval == 0:
            save_checkpoint(latest, model, optimizer, step, cfg, best_val)

        throttle.wait()

    save_checkpoint(latest, model, optimizer, step, cfg, best_val)
    print("terminé")


if __name__ == "__main__":
    main()
