# https://github.com/pytorch/pytorch/issues/156680
# https://github.com/pytorch/pytorch/issues/161302
#
# Bug: nn.SyncBatchNorm causes graph breaks under torch.compile.
# With fullgraph=True, compilation fails because SyncBatchNorm's forward
# uses autograd.Function and calls torch.cuda.is_current_stream_capturing()
# which returns a non-Tensor and can't be traced into the FX graph.
#
# Observed error:
#   Unsupported: torch.* op returned non-Tensor
#   Higher Order Operator: autograd.Function
#   target: <function is_current_stream_capturing>
#
# Run with: torchrun --nproc_per_node=2 repros/issue_156680.py

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class Demo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.SyncBatchNorm(3)

    def forward(self, x):
        return torch.nn.functional.relu(self.bn(self.conv1(x)))

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    model = Demo().to(rank)
    model = torch.compile(model, fullgraph=True)
    model = DDP(model, device_ids=[rank])

    try:
        example_inputs = torch.randn(1, 3, 224, 224, device=f"cuda:{rank}")
        out = model(example_inputs)
        print(f"[Rank {rank}] Success: output shape {out.shape}")
    except Exception as e:
        print(f"[Rank {rank}] BUG REPRODUCED: {type(e).__name__}: {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
