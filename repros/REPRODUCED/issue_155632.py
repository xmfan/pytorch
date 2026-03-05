# https://github.com/pytorch/pytorch/issues/155632
# https://github.com/pytorch/pytorch/issues/138795
#
# Bug: all_gather_into_tensor functional collective only implements concat,
# not stack-style. When output tensor shape implies stacking (e.g., (world_size, N)
# from input (N,)), it fails under compile with shape mismatch.
#
# Observed error:
#   RuntimeError('expand: attempting to expand a dimension of length 4 -> 2!')
#   Dynamo failed to run FX node with fake tensors: call_method copy_(
#     FakeTensor(size=(2, 2)), FakeTensor(size=(4,)))
#
# The functional version produces a concatenated (4,) tensor but the output
# expects a stacked (2, 2) tensor.
#
# Run with: torchrun --nproc_per_node=2 repros/issue_155632.py

import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # Input: shape (2,) per rank
    tensor_in = torch.arange(2, dtype=torch.int64, device="cuda") + rank * 10

    # Output: shape (world_size, 2) -- implies stacking, not concatenation
    tensor_out = torch.zeros(world_size, 2, dtype=torch.int64, device="cuda")

    print(f"[Rank {rank}] tensor_in: {tensor_in}, tensor_out shape: {tensor_out.shape}")

    # Eager mode should work
    try:
        dist.all_gather_into_tensor(tensor_out, tensor_in)
        print(f"[Rank {rank}] Eager succeeded: {tensor_out}")
    except Exception as e:
        print(f"[Rank {rank}] Eager failed: {e}")

    # Compiled mode -- this should fail
    tensor_out2 = torch.zeros(world_size, 2, dtype=torch.int64, device="cuda")
    try:
        @torch.compile(fullgraph=True)
        def compiled_allgather(out, inp):
            dist.all_gather_into_tensor(out, inp)
            return out

        result = compiled_allgather(tensor_out2, tensor_in)
        print(f"[Rank {rank}] Compiled succeeded: {result}")
    except Exception as e:
        print(f"[Rank {rank}] Compiled failed: {type(e).__name__}: {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
