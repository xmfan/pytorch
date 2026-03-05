# https://github.com/pytorch/pytorch/issues/175258
#
# Bug: Block mask caching for flex attention and SAC don't play nicely together
# Error: RuntimeError: Only Tensors of floating point and complex dtype can require gradients
#
# The issue occurs when combining compiled create_block_mask with cached BlockMask,
# DTensors, selective activation checkpointing (SAC), and aot_autograd.
#
# This is a simplified repro attempt - the full repro is quite complex and requires
# specific flex attention setup that may not be available in this environment.
#
# Run with: torchrun --nproc_per_node=2 repros/issue_175258.py

import os
import torch
import torch.distributed as dist
from torch.distributed._tensor import init_device_mesh, DTensor, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl
)

# Enable the problematic flag mentioned in the issue
os.environ["TORCHINDUCTOR_WRAP_INDUCTOR_COMPILED_REGIONS"] = "1"

def main():
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)

        mesh = init_device_mesh("cuda", (world_size,))

        # Try to import flex_attention - may not be available in all builds
        try:
            from torch.nn.attention.flex_attention import create_block_mask, BlockMask, flex_attention
            print(f"[Rank {rank}] flex_attention imported successfully")
        except ImportError as e:
            print(f"[Rank {rank}] flex_attention not available: {e}")
            print(f"[Rank {rank}] This issue requires flex_attention to reproduce")
            dist.destroy_process_group()
            return

        # Create a simple block mask function
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        # Try to create compiled block mask
        try:
            block_mask = create_block_mask(causal_mask, B=1, H=8, Q_LEN=128, KV_LEN=128)
            compiled_create_mask = torch.compile(create_block_mask)

            print(f"[Rank {rank}] Block mask creation: OK")
        except Exception as e:
            print(f"[Rank {rank}] Block mask creation failed: {type(e).__name__}: {e}")
            dist.destroy_process_group()
            return

        # Create input tensors
        B, H, S, D = 1, 8, 128, 64
        q = torch.randn(B, H, S, D, device="cuda", requires_grad=True)
        k = torch.randn(B, H, S, D, device="cuda", requires_grad=True)
        v = torch.randn(B, H, S, D, device="cuda", requires_grad=True)

        # Convert to DTensor
        try:
            q_dt = DTensor.from_local(q, mesh, [Shard(2)])
            k_dt = DTensor.from_local(k, mesh, [Shard(2)])
            v_dt = DTensor.from_local(v, mesh, [Shard(2)])
            print(f"[Rank {rank}] DTensor conversion: OK")
        except Exception as e:
            print(f"[Rank {rank}] DTensor conversion failed: {type(e).__name__}: {e}")
            dist.destroy_process_group()
            return

        # Create a function that combines flex_attention with DTensor operations
        def flex_attention_fn(q_dt, k_dt, v_dt):
            # Convert to local for flex_attention
            q_local = q_dt.to_local()
            k_local = k_dt.to_local()
            v_local = v_dt.to_local()

            # Apply flex_attention with block mask
            out_local = flex_attention(q_local, k_local, v_local, block_mask=block_mask)

            # Convert back to DTensor
            return DTensor.from_local(out_local, mesh, [Shard(2)])

        # Wrap with selective activation checkpointing
        def checkpoint_policy(submodule, inputs, outputs):
            return True  # Save all outputs

        try:
            checkpointed_fn = checkpoint_wrapper(
                flex_attention_fn,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )

            # Compile the checkpointed function
            compiled_fn = torch.compile(checkpointed_fn)

            print(f"[Rank {rank}] Running compiled + checkpointed flex_attention...")
            output = compiled_fn(q_dt, k_dt, v_dt)
            print(f"[Rank {rank}] Output shape: {output.shape}")

            # Try backward pass where the error typically occurs
            loss = output.sum()
            loss.backward()
            print(f"[Rank {rank}] Backward pass: OK")
            print(f"[Rank {rank}] Bug may be fixed or repro incomplete")

        except RuntimeError as e:
            if "Only Tensors of floating point and complex dtype can require gradients" in str(e):
                print(f"[Rank {rank}] BUG REPRODUCED: {e}")
            else:
                print(f"[Rank {rank}] Other RuntimeError: {e}")
        except Exception as e:
            print(f"[Rank {rank}] Other error: {type(e).__name__}: {e}")

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()