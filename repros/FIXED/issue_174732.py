# https://github.com/pytorch/pytorch/issues/174732
#
# Bug: DTensor aot_eager compiling sharded embedding with dynamic seqlen fails
# Error: IndexError: index out of range in range function
#
# When combining torch.compile with aot_eager backend, mark_dynamic for dynamic
# sequence length, and DTensor with RowwiseParallel embedding sharding, we get
# an IndexError during compilation.
#
# Run with: torchrun --nproc_per_node=2 repros/issue_174732.py

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._tensor import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # Create embedding model
    model = nn.Embedding(151936, 128).cuda()

    # Setup tensor parallel
    tp_mesh = init_device_mesh("cuda", (world_size,))
    parallelize_module(model, tp_mesh, RowwiseParallel())

    # Create input with dynamic sequence length marked
    tokens = torch.zeros(1, 16384, dtype=torch.int32, device="cuda")
    torch._dynamo.decorators.mark_dynamic(tokens, 1)

    # Eager mode should work
    try:
        eager_out = model(tokens)
        print(f"[Rank {rank}] Eager mode: OK, shape={eager_out.shape}")
    except Exception as e:
        print(f"[Rank {rank}] Eager failed: {type(e).__name__}: {e}")

    # Compiled with aot_eager should fail
    try:
        fn = torch.compile(model, backend="aot_eager", fullgraph=True)
        compiled_out = fn(tokens)
        print(f"[Rank {rank}] Compiled mode: OK, shape={compiled_out.shape}")
    except Exception as e:
        print(f"[Rank {rank}] BUG REPRODUCED: {type(e).__name__}: {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()