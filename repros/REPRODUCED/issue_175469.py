# https://github.com/pytorch/pytorch/issues/175469
#
# Bug: run_decompositions() fails because aten.linear.default lacks a DTensor
# sharding strategy.
#
# Observed error:
#   AssertionError: While executing %linear : call_function[target=
#   torch.ops.aten.linear.default](args = (%flat_apply, %in_proj_weight,
#   %in_proj_bias), kwargs = {})
#
# The export mechanism preserves aten.linear operations, but DTensor dispatch
# requires sharding strategies for all ops. When aten.linear is preserved but
# has no registered strategy, run_decompositions() fails.
#
# Run with: torchrun --nproc_per_node=2 repros/issue_175469.py

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel
)
from torch.distributed._tensor import init_device_mesh
from torch._decomp import get_decompositions
import torch.utils._pytree as pytree
import torch.distributed.tensor._dtensor_spec as dtensor_spec

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    device_mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    # Register DTensorSpec as pytree constant
    pytree.register_constant(dtensor_spec.DTensorSpec)

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(10, 320)
            self.out_proj = nn.Linear(320, 160)

        def forward(self, x):
            return self.out_proj(torch.relu(self.in_proj(x)))

    model = ToyModel().to("cuda")
    parallelize_module(model.in_proj, device_mesh, ColwiseParallel())
    parallelize_module(model.out_proj, device_mesh, RowwiseParallel())

    inp = torch.rand(2, 10, device="cuda")

    # Export the model
    try:
        exported_program = torch.export.export(model, (inp,), strict=False)
        print(f"[Rank {rank}] Export succeeded")
    except Exception as e:
        print(f"[Rank {rank}] Export failed: {type(e).__name__}: {e}")
        dist.destroy_process_group()
        return

    # Attempt decompositions -- this should fail
    try:
        decomp_table = get_decompositions([
            torch.ops.aten.embedding_dense_backward,
            torch.ops.aten.native_layer_norm_backward,
            torch.ops.aten.slice_backward,
            torch.ops.aten.select_backward,
            torch.ops.aten.norm.ScalarOpt_dim,
            torch.ops.aten.native_group_norm_backward,
            torch.ops.aten.upsample_bilinear2d.vec,
            torch.ops.aten.split.Tensor,
            torch.ops.aten.split_with_sizes,
        ])
        decomposed = exported_program.run_decompositions(decomp_table)
        print(f"[Rank {rank}] Decomposition succeeded (bug may be fixed)")
    except AssertionError as e:
        print(f"[Rank {rank}] BUG REPRODUCED: AssertionError: {e}")
    except Exception as e:
        print(f"[Rank {rank}] BUG REPRODUCED: {type(e).__name__}: {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
