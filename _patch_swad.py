"""给 train_ddp 增加 SWAD（低学习率阶段密集权重平均）并放行 strong/clahe 选项。"""

from pathlib import Path

p = Path("/mnt/glaucoma/proj/v2/train_ddp.py")
s = p.read_text()

# 1) argparse 放行 strong 增强与 clahe 输入
s = s.replace('choices=["basic", "geo", "geo2", "zoom"]',
              'choices=["basic", "geo", "geo2", "zoom", "strong"]')
s = s.replace('choices=["rgb", "green", "gray"]',
              'choices=["rgb", "green", "gray", "clahe"]')

# 2) SWAD 参数
old = '    parser.add_argument("--no-ema", action="store_true")\n'
new = old + '''    parser.add_argument("--swad", action="store_true",
                        help="低学习率阶段密集权重平均（SWAD），rank0 保存 best_swad.pt")
    parser.add_argument("--swad-start", type=float, default=0.65,
                        help="从第几比例 epoch 开始平均（默认 0.65，即最后 35%%）")
    parser.add_argument("--swad-every", type=int, default=20, help="每多少优化步平均一次")
'''
assert old in s
s = s.replace(old, new, 1)

# 3) SWADStats 类
old_cls = '''def make_optimizer(model: nn.Module, lr: float, wd: float):
'''
new_cls = '''class SWADStats:
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
'''
assert old_cls in s
s = s.replace(old_cls, new_cls, 1)

# 4) 实例化（rank0）
old = '    ema = None if args.no_ema else ModelEMA(model, args.ema_decay)\n'
new = old + '''    swad = SWADStats() if (is_main and args.swad) else None
    swad_eval_model = deepcopy(unwrap(model)).eval() if swad is not None else None
    if swad_eval_model is not None:
        for p_ in swad_eval_model.parameters():
            p_.requires_grad_(False)
    swad_start_epoch = max(1, int(round(args.epochs * args.swad_start)))
'''
assert old in s
s = s.replace(old, new, 1)

# 5) 每个优化步累加
old = '''                if ema is not None:
                    ema.update(model)
'''
new = '''                if ema is not None:
                    ema.update(model)
                if (swad is not None and epoch >= swad_start_epoch
                        and global_step % args.swad_every == 0):
                    swad.update(model)
'''
assert old in s
s = s.replace(old, new, 1)

# 6) metrics 表头加 swad_ap
old = '''        writer.writerow(["epoch", "train_loss", "valid_auroc", "valid_ap", "valid_logloss",
                         "valid_acc", "ema_ap", "lr", "sec"])
'''
new = '''        writer.writerow(["epoch", "train_loss", "valid_auroc", "valid_ap", "valid_logloss",
                         "valid_acc", "ema_ap", "swad_ap", "lr", "sec"])
'''
assert old in s
s = s.replace(old, new, 1)

# 7) skip-valid 行补一列
old = '''            writer.writerow([epoch, f"{train_loss:.5f}", "", "", "", "", "",
                             f"{optimizer.param_groups[0]['lr']:.2e}", f"{sec:.1f}"])
'''
new = '''            writer.writerow([epoch, f"{train_loss:.5f}", "", "", "", "", "", "",
                             f"{optimizer.param_groups[0]['lr']:.2e}", f"{sec:.1f}"])
'''
assert old in s
s = s.replace(old, new, 1)

# 8) 评估时算 SWAD
old = '''            metrics = evaluate(unwrap(model), val_loader, device)
            ema_metrics = evaluate(ema.ema, val_loader, device) if ema is not None else None
            sec = time.perf_counter() - ep_start
'''
new = '''            metrics = evaluate(unwrap(model), val_loader, device)
            ema_metrics = evaluate(ema.ema, val_loader, device) if ema is not None else None
            swad_metrics = None
            swad_avg = swad.averaged() if swad is not None else None
            if swad_avg is not None:
                swad_eval_model.load_state_dict(swad_avg)
                swad_metrics = evaluate(swad_eval_model, val_loader, device)
            sec = time.perf_counter() - ep_start
'''
assert old in s
s = s.replace(old, new, 1)

# 9) 日志与 csv 行加入 SWAD
old = '''                + (f"EMA_AP={ema_metrics['ap']:.4f} EMA_AUROC={ema_metrics['auroc']:.4f} "
                   if ema_metrics else "")
                + f"({sec:.0f}s)")
'''
new = '''                + (f"EMA_AP={ema_metrics['ap']:.4f} EMA_AUROC={ema_metrics['auroc']:.4f} "
                   if ema_metrics else "")
                + (f"SWAD_AP={swad_metrics['ap']:.4f} " if swad_metrics else "")
                + f"({sec:.0f}s)")
'''
assert old in s
s = s.replace(old, new, 1)

old = '''                             f"{ema_metrics['ap']:.6f}" if ema_metrics else "",
                             f"{optimizer.param_groups[0]['lr']:.2e}", f"{sec:.1f}"])
'''
new = '''                             f"{ema_metrics['ap']:.6f}" if ema_metrics else "",
                             f"{swad_metrics['ap']:.6f}" if swad_metrics else "",
                             f"{optimizer.param_groups[0]['lr']:.2e}", f"{sec:.1f}"])
'''
assert old in s
s = s.replace(old, new, 1)

# 10) best_ap 也考虑 SWAD
old = '''            if ema_metrics and ema_metrics["ap"] > best_ap:
                best_ap, best_ap_epoch = ema_metrics["ap"], epoch
                save_ckpt({**base, "model": ema.ema.state_dict(), "epoch": epoch,
                           "metrics": ema_metrics, "ema": True, "exp_id": args.exp_id},
                          run_dir / "best_ap.pt")
'''
new = old + '''            if swad_metrics and swad_metrics["ap"] > best_ap:
                best_ap, best_swad_epoch = swad_metrics["ap"], epoch
                save_ckpt({**base, "model": swad_eval_model.state_dict(), "epoch": epoch,
                           "metrics": swad_metrics, "ema": False, "swad": True,
                           "exp_id": args.exp_id}, run_dir / "best_ap.pt")
'''
assert old in s
s = s.replace(old, new, 1)

old = '''            cur_best = max(metrics["ap"], ema_metrics["ap"] if ema_metrics else -1.0)
'''
new = '''            cur_best = max(metrics["ap"], ema_metrics["ap"] if ema_metrics else -1.0,
                           swad_metrics["ap"] if swad_metrics else -1.0)
'''
assert old in s
s = s.replace(old, new, 1)

# 11) 初始化 swad 最佳记录
old = '''    best_ap, best_ap_epoch = -1.0, 0
'''
new = '''    best_ap, best_ap_epoch = -1.0, 0
    best_swad_epoch = 0
'''
assert old in s
s = s.replace(old, new, 1)

# 12) DONE 日志带 SWAD 最佳
old = '''        log(f"DONE best_AP={best_ap:.6f}@e{best_ap_epoch} best_AUROC={best_au:.6f}@e{best_au_epoch} "
            f"total={(time.perf_counter() - start) / 60:.1f}min")
'''
new = '''        log(f"DONE best_AP={best_ap:.6f}@e{best_ap_epoch} best_SWAD_e{best_swad_epoch} "
            f"best_AUROC={best_au:.6f}@e{best_au_epoch} "
            f"total={(time.perf_counter() - start) / 60:.1f}min")
'''
assert old in s
s = s.replace(old, new, 1)

p.write_text(s)
print("train_ddp.py patched: SWAD + strong/clahe options")
