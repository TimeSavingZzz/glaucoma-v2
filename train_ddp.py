"""DDP 训练：timm 骨干 + AMP + EMA + warmup/cosine + 双 checkpoint。

启动（4 卡）：
  HF_ENDPOINT=https://hf-mirror.com torchrun --nproc_per_node=4 train_ddp.py \
      --exp-id V2-CNXB --arch convnext_base.fb_in22k_ft_in1k --size 512 \
      --epochs 10 --batch-size 4 --accum 2 --lr 2e-4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from config import IMAGE_SIZE, RUNS_DIR, SEED, VALID_CSV, VALID_DIR
from dataset import DistributedWeightedSampler, GlaucomaDataset, _worker_init_fn
from models import build_model as build_timm_model, data_norm as timm_data_norm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = deepcopy(unwrap(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.step = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        msd = unwrap(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


class SWADStats:
    """SWAD 权重累加器：对（最后一段 epoch 的）参数做均匀平均。

    与 EMA 互补：EMA 是指数平均、偏向近期；SWAD 是等权平均、落在低损失平坦区域。
    只在 rank0 维护，不引入任何跨进程通信；评估用一份独立副本。
    """

    def __init__(self) -> None:
        self.sum_sd: dict[str, torch.Tensor] = {}
        self.n = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = unwrap(model).state_dict()
        if not self.sum_sd:
            for k, v in msd.items():
                if v.dtype.is_floating_point:
                    self.sum_sd[k] = v.detach().float().clone()
                else:
                    self.sum_sd[k] = v.detach().clone()
        else:
            for k, v in msd.items():
                if self.sum_sd[k].dtype.is_floating_point:
                    self.sum_sd[k].add_(v.detach().float())
        self.n += 1

    def averaged(self):
        if self.n == 0:
            return None
        out = {}
        for k, v in self.sum_sd.items():
            if v.dtype.is_floating_point:
                out[k] = (v / self.n).to(torch.float32)
            else:
                out[k] = v.clone()
        return out


def make_optimizer(model: nn.Module, lr: float, wd: float):
    decay, no_decay = [], []
    for n, p in unwrap(model).named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def make_scheduler(optimizer, epochs: int, warmup: int, min_lr: float, base_lr: float):
    floor = min_lr / max(base_lr, 1e-12)

    def fn(e: int) -> float:
        if e < warmup:
            return (e + 1) / max(warmup, 1)
        p = (e - warmup) / max(1, epochs - warmup)
        return max(floor, 0.5 * (1.0 + math.cos(math.pi * min(1.0, p))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.inference_mode()
def evaluate(model: nn.Module, loader, device) -> dict:
    model.eval()
    probs, labels = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        labels.append(y)
    y_prob = torch.cat(probs).numpy()
    y_true = torch.cat(labels).numpy()
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "acc": float(((y_prob >= 0.5) == y_true).mean()),
    }


def save_ckpt(obj: dict, path: Path) -> None:
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, str(path))


IMAGENET_MEAN_STD = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def build_train_model(arch: str, size: int, drop_path: float):
    """timm 骨干走 models.build_model，torchvision 骨干（resnet50 等）走 torchvision_models。"""
    if "." in arch:
        return build_timm_model(arch, num_classes=2, pretrained=True, image_size=size,
                                drop_path=drop_path)
    import torchvision_models
    return torchvision_models.build_model(num_classes=2, pretrained=True, arch=arch,
                                          image_size=size)


def train_data_norm(arch: str):
    if "." in arch:
        return timm_data_norm(arch)
    return IMAGENET_MEAN_STD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", default="V2")
    parser.add_argument("--arch", default="convnext_base.fb_in22k_ft_in1k")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4, help="单卡 batch")
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--aug", default="geo2", choices=["basic", "geo", "geo2", "zoom", "zoomrobust", "strong"])
    parser.add_argument("--input-mode", default="rgb", choices=["rgb", "green", "gray", "clahe"])
    parser.add_argument("--fov-crop", action="store_true")
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--swad", action="store_true",
                        help="低学习率阶段密集权重平均（SWAD），rank0 保存 best_swad.pt")
    parser.add_argument("--swad-start", type=float, default=0.65,
                        help="从第几比例 epoch 开始平均（默认 0.65，即最后 35%%）")
    parser.add_argument("--swad-every", type=int, default=20, help="每多少优化步平均一次")
    parser.add_argument("--syncbn", action="store_true",
                        help="DDP 下转 SyncBatchNorm（默认关闭，单卡 BN 更稳）")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--label-smooth", type=float, default=0.0)
    parser.add_argument("--rank-weight", type=float, default=0.0,
                        help="pairwise logistic 排序损失权重（0 关闭，旧配方用 0.5）")
    parser.add_argument("--pos-weight", type=float, default=1.0,
                        help="正类过采样权重（1 关闭；4 约把正类比例提升到 12%%）")
    parser.add_argument("--train-csv", default="")
    parser.add_argument("--train-dir", default="")
    parser.add_argument("--valid-csv", default="")
    parser.add_argument("--valid-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--init-ckpt", default="",
                        help="从已有 checkpoint 初始化权重（高分辨率微调/继续训练）")
    parser.add_argument("--extra-csv", default="", help="额外训练数据（如 valid 并入训练）")
    parser.add_argument("--extra-dir", default="")
    parser.add_argument("--skip-valid", action="store_true",
                        help="跳过 valid 评估/选模（valid 已并入训练时使用，只存 last.pt）")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world > 1
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main = rank == 0
    mean, std = train_data_norm(args.arch)

    from config import TRAIN_CSV, TRAIN_DIR
    train_csv = Path(args.train_csv) if args.train_csv else TRAIN_CSV
    train_dir = Path(args.train_dir) if args.train_dir else TRAIN_DIR
    valid_csv = Path(args.valid_csv) if args.valid_csv else VALID_CSV
    valid_dir = Path(args.valid_dir) if args.valid_dir else VALID_DIR

    set_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / f"{args.exp_id}_{stamp}"
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    log_file = open(run_dir / "train.log", "a", encoding="utf-8") if is_main else None

    def log(msg: str) -> None:
        if is_main:
            line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()

    train_ds = GlaucomaDataset(train_csv, train_dir, image_size=args.size, train=True,
                               aug=args.aug, limit=args.limit, fov_crop=args.fov_crop,
                               input_mode=args.input_mode, mean=mean, std=std)
    if args.extra_csv:
        if not args.extra_dir:
            raise SystemExit("--extra-csv 需要 --extra-dir")
        from torch.utils.data import ConcatDataset
        extra_ds = GlaucomaDataset(args.extra_csv, args.extra_dir, image_size=args.size,
                                   train=True, aug=args.aug, limit=args.limit,
                                   fov_crop=args.fov_crop, input_mode=args.input_mode,
                                   mean=mean, std=std)
        train_ds = ConcatDataset([train_ds, extra_ds])
    val_ds = GlaucomaDataset(valid_csv, valid_dir, image_size=args.size, train=False,
                             limit=args.limit, fov_crop=args.fov_crop,
                             input_mode=args.input_mode, mean=mean, std=std) if (is_main and not args.skip_valid) else None
    if distributed and args.pos_weight > 1.0:
        labels = train_ds.labels if hasattr(train_ds, "labels") else np.concatenate(
            [d.labels for d in train_ds.datasets])
        sampler = DistributedWeightedSampler(labels, world, rank,
                                             pos_weight=args.pos_weight, seed=args.seed)
    else:
        sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank,
                                     shuffle=True, drop_last=True) if distributed else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
        drop_last=True, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0, worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch, shuffle=False, num_workers=max(4, args.workers // 2),
        pin_memory=True,
    ) if val_ds is not None else None

    log("=" * 72)
    log(f"run_dir={run_dir} world={world} arch={args.arch} size={args.size} "
        f"bs/gpu={args.batch_size} accum={args.accum} epochs={args.epochs} lr={args.lr} "
        f"wd={args.weight_decay} aug={args.aug} amp={args.amp} ema={not args.no_ema} "
        f"pos_weight={args.pos_weight}")
    if hasattr(train_ds, "label_counts"):
        counts = train_ds.label_counts()
    else:
        parts = train_ds.datasets
        vals, cnts = np.unique(np.concatenate([np.asarray(p.labels) for p in parts]), return_counts=True)
        counts = {int(v): int(c) for v, c in zip(vals, cnts)}
    log(f"train={len(train_ds)} {counts} batches/gpu={len(train_loader)}")

    model = build_train_model(args.arch, args.size, args.drop_path).to(
        device, memory_format=torch.channels_last)
    if args.init_ckpt:
        init = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        init_sd = init.get("model", init)
        missing, unexpected = model.load_state_dict(init_sd, strict=False)
        real_missing = [k for k in missing if "head" not in k]
        if real_missing or unexpected:
            raise SystemExit(f"init-ckpt 不匹配: missing={real_missing[:5]} unexpected={unexpected[:5]}")
        log(f"init from {args.init_ckpt} (epoch={init.get('epoch')}, "
            f"skipped={len(missing)} head keys)")
    if distributed:
        if args.syncbn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    ema = None if args.no_ema else ModelEMA(model, args.ema_decay)
    swad = SWADStats() if (is_main and args.swad) else None
    swad_eval_model = deepcopy(unwrap(model)).eval() if swad is not None else None
    if swad_eval_model is not None:
        for p_ in swad_eval_model.parameters():
            p_.requires_grad_(False)
    swad_start_epoch = max(1, int(round(args.epochs * args.swad_start)))

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr, args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if is_main:
        cfg = {**vars(args), "world_size": world, "distributed": distributed,
               "mean": list(mean), "std": list(std),
               "torch": torch.__version__, "cuda": torch.version.cuda}
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        metrics_file = open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8")
        writer = csv.writer(metrics_file)
        writer.writerow(["epoch", "train_loss", "valid_auroc", "valid_ap", "valid_logloss",
                         "valid_acc", "ema_ap", "swad_ap", "lr", "sec"])
        metrics_file.flush()

    best_ap, best_ap_epoch = -1.0, 0
    best_swad_epoch = 0
    best_au, best_au_epoch = -1.0, 0
    bad = 0
    global_step = 0
    # global_step 按优化步计数（每 accum 个 batch 一次），warmup 必须换算成优化步
    warmup_steps = max(0, args.warmup_epochs) * (len(train_loader) // max(1, args.accum))
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        if distributed and sampler is not None:
            sampler.set_epoch(epoch)
        if ema is not None:
            ema.ema.train()
        ep_start = time.perf_counter()
        running, count = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y, _) in enumerate(train_loader, start=1):
            if warmup_steps and global_step < warmup_steps:
                scale = (global_step + 1) / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = args.lr * scale
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x)
                loss = criterion(logits, y)
                if args.rank_weight > 0:
                    z = logits[:, 1] - logits[:, 0]
                    zp, zn = z[y == 1], z[y == 0]
                    if zp.numel() and zn.numel():
                        rank = F.softplus(-(zp[:, None] - zn[None, :])).mean()
                        loss = loss + args.rank_weight * rank
                loss = loss / args.accum
            scaler.scale(loss).backward()
            if step % args.accum == 0:
                if args.clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if ema is not None:
                    ema.update(model)
                if (swad is not None and epoch >= swad_start_epoch
                        and global_step % args.swad_every == 0):
                    swad.update(model)
            running += loss.item() * args.accum
            count += 1
            if is_main and step % 200 == 0:
                log(f"e{epoch} step {step}/{len(train_loader)} loss={running / count:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step()
        train_loss = running / max(count, 1)

        if is_main and args.skip_valid:
            base = {"arch": args.arch, "image_size": args.size, "input_mode": args.input_mode,
                    "fov_crop": args.fov_crop,
                    "mean": list(mean), "std": list(std)}
            save_ckpt({**base, "model": unwrap(model).state_dict(), "epoch": epoch,
                       "metrics": None, "ema": False, "exp_id": args.exp_id},
                      run_dir / "last.pt")
            if ema is not None:
                save_ckpt({**base, "model": ema.ema.state_dict(), "epoch": epoch,
                           "metrics": None, "ema": True, "exp_id": args.exp_id},
                          run_dir / "last_ema.pt")
            sec = time.perf_counter() - ep_start
            log(f"epoch {epoch}/{args.epochs} loss={train_loss:.4f} ({sec:.0f}s, no valid)")
            writer.writerow([epoch, f"{train_loss:.5f}", "", "", "", "", "", "",
                             f"{optimizer.param_groups[0]['lr']:.2e}", f"{sec:.1f}"])
            metrics_file.flush()
        elif is_main:
            metrics = evaluate(unwrap(model), val_loader, device)
            ema_metrics = evaluate(ema.ema, val_loader, device) if ema is not None else None
            swad_metrics = None
            swad_avg = swad.averaged() if swad is not None else None
            if swad_avg is not None:
                swad_eval_model.load_state_dict(swad_avg)
                swad_metrics = evaluate(swad_eval_model, val_loader, device)
            sec = time.perf_counter() - ep_start
            log(f"epoch {epoch}/{args.epochs} loss={train_loss:.4f} "
                f"AP={metrics['ap']:.4f} AUROC={metrics['auroc']:.4f} "
                + (f"EMA_AP={ema_metrics['ap']:.4f} EMA_AUROC={ema_metrics['auroc']:.4f} "
                   if ema_metrics else "")
                + (f"SWAD_AP={swad_metrics['ap']:.4f} " if swad_metrics else "")
                + f"({sec:.0f}s)")
            writer.writerow([epoch, f"{train_loss:.5f}", f"{metrics['auroc']:.6f}",
                             f"{metrics['ap']:.6f}", f"{metrics['logloss']:.6f}",
                             f"{metrics['acc']:.6f}",
                             f"{ema_metrics['ap']:.6f}" if ema_metrics else "",
                             f"{swad_metrics['ap']:.6f}" if swad_metrics else "",
                             f"{optimizer.param_groups[0]['lr']:.2e}", f"{sec:.1f}"])
            metrics_file.flush()

            base = {"arch": args.arch, "image_size": args.size, "input_mode": args.input_mode,
                    "fov_crop": args.fov_crop,
                    "mean": list(mean), "std": list(std)}
            raw_sd = unwrap(model).state_dict()
            prev_best = best_ap
            if metrics["ap"] > best_ap:
                best_ap, best_ap_epoch = metrics["ap"], epoch
                save_ckpt({**base, "model": raw_sd, "epoch": epoch, "metrics": metrics,
                           "ema": False, "exp_id": args.exp_id}, run_dir / "best_ap.pt")
            if ema_metrics and ema_metrics["ap"] > best_ap:
                best_ap, best_ap_epoch = ema_metrics["ap"], epoch
                save_ckpt({**base, "model": ema.ema.state_dict(), "epoch": epoch,
                           "metrics": ema_metrics, "ema": True, "exp_id": args.exp_id},
                          run_dir / "best_ap.pt")
            if swad_metrics and swad_metrics["ap"] > best_ap:
                best_ap, best_swad_epoch = swad_metrics["ap"], epoch
                save_ckpt({**base, "model": swad_eval_model.state_dict(), "epoch": epoch,
                           "metrics": swad_metrics, "ema": False, "swad": True,
                           "exp_id": args.exp_id}, run_dir / "best_ap.pt")
            if metrics["auroc"] > best_au:
                best_au, best_au_epoch = metrics["auroc"], epoch
                save_ckpt({**base, "model": raw_sd, "epoch": epoch, "metrics": metrics,
                           "ema": False, "exp_id": args.exp_id}, run_dir / "best_auroc.pt")
            save_ckpt({**base, "model": raw_sd, "epoch": epoch, "metrics": metrics,
                       "ema": False, "exp_id": args.exp_id}, run_dir / "last.pt")

            cur_best = max(metrics["ap"], ema_metrics["ap"] if ema_metrics else -1.0,
                           swad_metrics["ap"] if swad_metrics else -1.0)
            if cur_best > prev_best + 1e-9:
                bad = 0
            else:
                bad += 1
            if bad >= args.patience:
                log(f"early stop at epoch {epoch} (best AP {best_ap:.4f} @e{best_ap_epoch})")
        if distributed:
            stop = torch.tensor([1 if (is_main and bad >= args.patience) else 0], device=device)
            dist.broadcast(stop, src=0)
            if stop.item() == 1:
                break
        elif is_main and bad >= args.patience:
            break

    if is_main:
        log(f"DONE best_AP={best_ap:.6f}@e{best_ap_epoch} best_SWAD_e{best_swad_epoch} "
            f"best_AUROC={best_au:.6f}@e{best_au_epoch} "
            f"total={(time.perf_counter() - start) / 60:.1f}min")
        metrics_file.close()
        log_file.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
