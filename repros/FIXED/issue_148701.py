# https://github.com/pytorch/pytorch/issues/148701
#
# Bug: aot_eager produces wrong output with all_gather_tensor_autograd
# High priority correctness issue - gradients differ between eager and compiled
#
# The gradient computation produces incorrect results with torch.compile(...,
# backend="aot_eager") compared to uncompiled version on distributed collective
# operations.
#
# Run with: torchrun --nproc_per_node=2 repros/issue_148701.py

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c

def model(x):
    x = ft_c.all_gather_tensor_autograd(x, gather_dim=0, group=dist.group.WORLD)
    x = ft_c.wait_tensor(x)
    return x

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # Setup input and gradient
    x = torch.randn([16], device="cuda", requires_grad=True)
    gy = torch.ones([16 * world_size], device="cuda")

    print(f"[Rank {rank}] Running reference (eager) version...")

    # Reference (eager) version
    ref_out = model(x)
    ref_out.backward(gy)
    ref_grad = x.grad.clone()
    print(f"[Rank {rank}] Reference grad: {ref_grad}")

    # Reset gradient
    x.grad = None

    print(f"[Rank {rank}] Running compiled (aot_eager) version...")

    # Compiled version - should produce different (wrong) gradient
    try:
        compiled_model = torch.compile(model, backend="aot_eager")
        compiled_out = compiled_model(x)
        compiled_out.backward(gy)
        compiled_grad = x.grad.clone()
        print(f"[Rank {rank}] Compiled grad: {compiled_grad}")

        # Check if gradients match
        if torch.allclose(ref_grad, compiled_grad, atol=1e-6):
            print(f"[Rank {rank}] Gradients match - bug may be fixed")
        else:
            max_diff = torch.max(torch.abs(ref_grad - compiled_grad))
            print(f"[Rank {rank}] BUG REPRODUCED: Gradient mismatch, max diff: {max_diff}")

    except Exception as e:
        print(f"[Rank {rank}] Compiled version failed: {type(e).__name__}: {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()