# https://github.com/pytorch/ao/issues/3982
#
# Bug: float8 + FSDP2: crash on uneven sharding (tensor_to_amax + storage mismatch)
# Created: 2026-03-03 (VERY RECENT)
#
# Two critical failures:
# 1. Empty shard crash: "max(): Expected reduction dim for input.numel() == 0"
# 2. Storage mismatch: "setStorage: sizes [4096, 1024] requiring 4194304 out of bounds for storage of size 4190208"
#
# Trigger condition: Fails with world_size=3 (uneven sharding: 4194304 % 3 ≠ 0), passes with world_size=4
# Root causes: Outdated fsdp_pre_all_gather hook signature + tensor_to_amax cannot handle zero-element tensors
#
# Requirements: torchao 0.17.0.dev20260302+cu128, torch 2.12.0.dev20260221+cu128
# Run with: torchrun --nproc_per_node=3 repros/REPRODUCED/issue_ao_3982.py

import torch
import torch.nn as nn
import torch.distributed as dist

def main():
    try:
        # Check if torchao is available
        try:
            from torchao.float8 import convert_to_float8_training
            from torchao.float8.config import Float8LinearConfig
            from torch.distributed._composable.fsdp import fully_shard
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("This test requires torchao. Install with:")
            print("pip install torchao --pre --index-url https://download.pytorch.org/whl/nightly/cu121")
            return

        # Initialize distributed
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)

        print(f"[Rank {rank}] Testing Float8 + FSDP2 uneven sharding bug...")
        print(f"[Rank {rank}] World size: {world_size}")

        # Create model that will trigger uneven sharding
        # Use odd dimension so params don't divide evenly by world_size
        hidden = 4097 if world_size == 2 else 4096
        model = nn.Sequential(
            nn.Linear(1024, hidden, bias=False),
            nn.Linear(hidden, 1024, bias=False),
        ).cuda().bfloat16()

        print(f"[Rank {rank}] Model parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"[Rank {rank}] First layer params: {model[0].weight.numel()}")
        print(f"[Rank {rank}] Divisible by world_size? {model[0].weight.numel() % world_size == 0}")

        # Convert to Float8 training with FSDP all-gather enabled
        print(f"[Rank {rank}] Converting to Float8 training...")
        try:
            convert_to_float8_training(
                model,
                config=Float8LinearConfig(enable_fsdp_float8_all_gather=True)
            )
            print(f"[Rank {rank}] Float8 conversion: OK")
        except Exception as e:
            print(f"[Rank {rank}] Float8 conversion failed: {type(e).__name__}: {e}")
            dist.destroy_process_group()
            return

        # Apply FSDP2
        print(f"[Rank {rank}] Applying FSDP2...")
        try:
            fully_shard(model)
            print(f"[Rank {rank}] FSDP2 sharding: OK")
        except Exception as e:
            print(f"[Rank {rank}] FSDP2 sharding failed: {type(e).__name__}: {e}")
            dist.destroy_process_group()
            return

        # Create input
        x = torch.randn(2, 128, 1024, device="cuda", dtype=torch.bfloat16)

        print(f"[Rank {rank}] Running forward + backward pass...")

        try:
            # Forward pass
            output = model(x)
            print(f"[Rank {rank}] Forward pass: OK, output shape: {output.shape}")

            # Backward pass - this is where the bug typically occurs
            loss = output.sum()
            loss.backward()
            print(f"[Rank {rank}] ✅ Backward pass: OK")
            print(f"[Rank {rank}] Test completed successfully - bug may be fixed!")

        except RuntimeError as e:
            error_msg = str(e)
            if "max(): Expected reduction dim for input.numel() == 0" in error_msg:
                print(f"[Rank {rank}] 🚨 BUG REPRODUCED: Empty shard crash")
                print(f"[Rank {rank}] Error: {e}")
            elif "setStorage: sizes" in error_msg and "out of bounds for storage" in error_msg:
                print(f"[Rank {rank}] 🚨 BUG REPRODUCED: Storage mismatch")
                print(f"[Rank {rank}] Error: {e}")
            else:
                print(f"[Rank {rank}] Other RuntimeError: {e}")
        except Exception as e:
            print(f"[Rank {rank}] Unexpected error: {type(e).__name__}: {e}")

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()