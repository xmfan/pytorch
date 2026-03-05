# https://github.com/pytorch/pytorch/issues/176446
#
# Bug: DTensor tp_conv rejects batch-dim-only sharding for valid convolution configs
# Error: RuntimeError: Stride must be 1 when there is padding for tensor parallel convolution.
#
# Root cause: Spatial constraints (stride/kernel/padding relationships) are enforced
# unconditionally through _is_supported() before checking if data exchange is
# necessary with _requires_data_exchange().
#
# This affects valid configurations like ResNet's first layer (kernel_size=7, stride=2, padding=3)
# when only batch dimension is sharded (no actual tensor parallel convolution).
#
# Run with: python repros/issue_176446.py

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributed.tensor import distribute_module, distribute_tensor, Replicate, Shard

def run_worker(rank, world_size):
    """Worker function for multiprocessing spawn"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device_mesh = dist.device_mesh.init_device_mesh("cuda", (world_size,))

        print(f"[Rank {rank}] Setting up ResNet-style conv layer...")

        # Create ResNet-style first conv layer (the problematic configuration)
        model = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3).cuda()

        # Parameter distribution function
        def param_fn(name, module, device_mesh):
            for pname, param in module.named_parameters():
                # Replicate parameters (no tensor parallel on weights)
                module.register_parameter(
                    pname.replace(".", "_"),
                    nn.Parameter(distribute_tensor(param, device_mesh, [Replicate()])),
                )

        print(f"[Rank {rank}] Distributing module...")
        model = distribute_module(model, device_mesh, param_fn)

        # Create input with batch-dimension-only sharding
        x = torch.randn(4, 3, 224, 224, device="cuda")
        x_dt = distribute_tensor(x, device_mesh, [Shard(0)])  # Only batch dim sharded

        print(f"[Rank {rank}] Input shape: {x.shape}")
        print(f"[Rank {rank}] DTensor input local shape: {x_dt.to_local().shape}")
        print(f"[Rank {rank}] Conv config: kernel_size=7, stride=2, padding=3")
        print(f"[Rank {rank}] Attempting forward pass...")

        # This should work since only batch dim is sharded (no spatial tensor parallel)
        # but currently fails due to overly strict validation
        try:
            out = model(x_dt)
            print(f"[Rank {rank}] ✅ Forward pass succeeded: {out.shape}")
            print(f"[Rank {rank}] Local output shape: {out.to_local().shape}")
        except RuntimeError as e:
            if "Stride must be 1 when there is padding" in str(e):
                print(f"[Rank {rank}] 🚨 BUG REPRODUCED: {e}")
            else:
                print(f"[Rank {rank}] Other RuntimeError: {e}")
        except Exception as e:
            print(f"[Rank {rank}] Unexpected error: {type(e).__name__}: {e}")

    except Exception as e:
        print(f"[Rank {rank}] Setup error: {type(e).__name__}: {e}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    if not torch.cuda.is_available():
        print("CUDA not available - this test requires CUDA")
        return

    if torch.cuda.device_count() < 2:
        print("At least 2 GPUs required for this test")
        return

    print("Testing DTensor tp_conv batch-dim sharding validation bug...")
    print("Expected: Forward pass should succeed (batch-dim-only sharding)")
    print("Actual: RuntimeError about stride/padding constraints")
    print()

    try:
        mp.spawn(run_worker, args=(2,), nprocs=2, join=True)
    except Exception as e:
        print(f"Multiprocessing error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()