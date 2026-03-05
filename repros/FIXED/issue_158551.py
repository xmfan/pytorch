# https://github.com/pytorch/pytorch/issues/158551
#
# Bug: CUDAGraphs RuntimeError: Accessing Overwritten Tensor Output Despite Clone + cudagraph_mark_step_begin
# Environment: PyTorch 2.3.0+cu121, CUDA 11.6, NVIDIA L20, Ubuntu 18.04
#
# Error: RuntimeError: accessing tensor output of CUDAGraphs that has been
# overwritten by a subsequent run... To prevent overwriting, clone the tensor
# outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin()
#
# Issue: Error occurs despite implementing suggested workarounds (cloning and mark_step_begin)
#
# Run with: torchrun --nproc_per_node=2 repros/issue_158551.py

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class ConvBNAct(nn.Module):
    """Basic ConvBNAct module that triggers the CUDAGraphs issue"""
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

def main():
    if not torch.cuda.is_available():
        print("CUDA not available - this test requires CUDA")
        return

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    print(f"[Rank {rank}] Testing CUDAGraphs + DDP issue...")

    # Create model and wrap with DDP
    model = ConvBNAct().cuda()
    model = DDP(model)

    # Compile the model - this enables CUDA graphs
    print(f"[Rank {rank}] Compiling model with torch.compile...")
    compiled_model = torch.compile(model)

    # Create sample input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224, device="cuda")

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    print(f"[Rank {rank}] Starting training loop...")

    for step in range(5):
        print(f"[Rank {rank}] Step {step}")

        # Workaround 1: Call cudagraph_mark_step_begin (suggested fix)
        try:
            torch.compiler.cudagraph_mark_step_begin()
            print(f"[Rank {rank}]   Called cudagraph_mark_step_begin()")
        except AttributeError:
            print(f"[Rank {rank}]   cudagraph_mark_step_begin not available")

        optimizer.zero_grad()

        try:
            # Forward pass with compiled model
            output = compiled_model(input_tensor)

            # Workaround 2: Clone tensor output (suggested fix)
            output = output.clone()
            print(f"[Rank {rank}]   Forward pass completed, output shape: {output.shape}")

            # Compute loss
            loss = output.mean()
            print(f"[Rank {rank}]   Loss: {loss.item():.6f}")

            # Backward pass
            loss.backward()
            print(f"[Rank {rank}]   Backward pass completed")

            # Optimizer step
            optimizer.step()
            print(f"[Rank {rank}]   Optimizer step completed")

        except RuntimeError as e:
            error_msg = str(e)
            if "accessing tensor output of CUDAGraphs" in error_msg and "overwritten" in error_msg:
                print(f"[Rank {rank}] 🚨 BUG REPRODUCED: {e}")
                print(f"[Rank {rank}]   Despite using suggested workarounds!")
                break
            else:
                print(f"[Rank {rank}] Other RuntimeError: {e}")
                break
        except Exception as e:
            print(f"[Rank {rank}] Unexpected error: {type(e).__name__}: {e}")
            break

    print(f"[Rank {rank}] Training loop completed")

    # Try alternative workaround - disable CUDA graphs entirely
    print(f"[Rank {rank}] Testing with CUDA graphs disabled...")

    # Set environment variable to disable CUDA graphs
    os.environ["TORCHINDUCTOR_USE_CUDAGRAPHS"] = "0"
    torch.backends.cuda.enable_cudagraph_trees = False

    # Re-compile with CUDA graphs disabled
    model2 = ConvBNAct().cuda()
    model2 = DDP(model2)
    compiled_model2 = torch.compile(model2)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.001)

    try:
        for step in range(2):
            optimizer2.zero_grad()
            output = compiled_model2(input_tensor)
            loss = output.mean()
            loss.backward()
            optimizer2.step()
            print(f"[Rank {rank}] Step {step} with disabled CUDA graphs: OK")
    except Exception as e:
        print(f"[Rank {rank}] Error even with CUDA graphs disabled: {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()