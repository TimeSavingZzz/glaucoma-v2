"""微基准：4 卡 DDP 下纯计算 step 耗时（排除数据加载）。"""
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from models import build_model

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
dist.init_process_group("nccl")
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
torch.backends.cudnn.benchmark = True

arch = os.environ.get("BENCH_ARCH", "tf_efficientnetv2_m.in21k_ft_in1k")
bs = int(os.environ.get("BENCH_BS", "4"))
model = build_model(arch, num_classes=2, pretrained=False, image_size=512).to(
    device, memory_format=torch.channels_last)
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[local_rank])
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = nn.CrossEntropyLoss()
x = torch.randn(bs, 3, 512, 512, device=device).contiguous(memory_format=torch.channels_last)
y = torch.randint(0, 2, (bs,), device=device)
scaler = torch.cuda.amp.GradScaler()

for i in range(5):  # warmup
    with torch.cuda.amp.autocast():
        loss = crit(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
n = 20
for i in range(n):
    with torch.cuda.amp.autocast():
        loss = crit(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / n
if rank == 0:
    print(f"arch={arch} bs={bs} world={world} step={dt*1000:.0f}ms imgs/s={bs*world/dt:.1f} "
          f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB")
dist.destroy_process_group()
