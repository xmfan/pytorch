# https://github.com/pytorch/pytorch/issues/161937
#
# Bug: AOTAutograd runtime alias reconstruction applies view on non-contiguous base under torch.compile (inductor + DDP)
# Error: RuntimeError: view size is not compatible with input tensor's size and stride... Use .reshape(...) instead.
#
# Environment: PyTorch 2.4.1+cu121, CUDA 12.1, NVIDIA L20, Ubuntu 18.04.6
#
# Root cause: Custom FX passes rewrite view operations to reshape operations successfully,
# but AOT's runtime alias reconstruction still attempts to use the original view chain
# on non-contiguous tensor bases.
#
# Configuration: torch._dynamo.config.optimize_ddp = False, mode="reduce-overhead", fullgraph=True
#
# Run with: torchrun --nproc_per_node=2 repros/issue_161937.py

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._functorch import config as functorch_config

class TestModel(nn.Module):
    """Model that creates non-contiguous tensors and uses view operations"""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.linear1 = nn.Linear(784, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 10)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)  # This creates a view operation

        # Forward through layers
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)

        # Create non-contiguous tensor through operations
        x = x.transpose(0, 1).contiguous().transpose(0, 1)  # Force non-contiguous

        x = torch.relu(self.linear2(x))
        x = self.dropout(x)

        # Another view operation on potentially non-contiguous tensor
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # This may trigger the bug

        x = self.linear3(x)
        return x

def main():
    if not torch.cuda.is_available():
        print("CUDA not available - this test requires CUDA")
        return

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    print(f"[Rank {rank}] Testing AOTAutograd alias reconstruction bug...")

    # Configuration from the issue report
    torch._dynamo.config.optimize_ddp = False
    print(f"[Rank {rank}] Set optimize_ddp = False")

    # Create model
    model = TestModel().cuda()

    # Convert to half precision (fp16) as mentioned in issue
    model = model.half()
    print(f"[Rank {rank}] Model converted to fp16")

    # Compile model BEFORE wrapping with DDP (as mentioned in issue)
    print(f"[Rank {rank}] Compiling model with reduce-overhead mode...")
    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    # Wrap with DDP after compilation
    ddp_model = DDP(compiled_model, device_ids=[rank])
    print(f"[Rank {rank}] Wrapped compiled model with DDP")

    # Create sample input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 28, 28, device="cuda", dtype=torch.half)

    # Setup optimizer
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=0.001)

    print(f"[Rank {rank}] Starting training that should trigger alias reconstruction bug...")

    for step in range(3):
        print(f"[Rank {rank}] Training step {step}")

        optimizer.zero_grad()

        try:
            # Forward pass - this is where the bug typically occurs
            output = ddp_model(input_tensor)
            print(f"[Rank {rank}]   Forward pass completed: {output.shape}")

            # Compute loss
            target = torch.randint(0, 10, (batch_size,), device="cuda")
            loss = nn.functional.cross_entropy(output, target)
            print(f"[Rank {rank}]   Loss computed: {loss.item():.4f}")

            # Backward pass - bug often occurs during AOT runtime reconstruction here
            loss.backward()
            print(f"[Rank {rank}]   Backward pass completed")

            # Optimizer step
            optimizer.step()
            print(f"[Rank {rank}]   Step {step} completed successfully")

        except RuntimeError as e:
            error_msg = str(e)
            if "view size is not compatible" in error_msg and "Use .reshape" in error_msg:
                print(f"[Rank {rank}] 🚨 BUG REPRODUCED: AOTAutograd alias reconstruction bug")
                print(f"[Rank {rank}]   Error: {e}")
                print(f"[Rank {rank}]   This occurs in runtime_wrappers.py → gen_alias_from_base")
                break
            else:
                print(f"[Rank {rank}] Other RuntimeError: {e}")
                break
        except Exception as e:
            print(f"[Rank {rank}] Unexpected error: {type(e).__name__}: {e}")
            break

    # Test the suggested workaround
    print(f"\n[Rank {rank}] Testing suggested workaround...")
    print(f"[Rank {rank}] Setting view_replay_for_aliased_outputs = False")

    try:
        functorch_config.view_replay_for_aliased_outputs = False

        # Create fresh model with workaround
        model2 = TestModel().cuda().half()
        compiled_model2 = torch.compile(model2, mode="reduce-overhead", fullgraph=True)
        ddp_model2 = DDP(compiled_model2, device_ids=[rank])
        optimizer2 = torch.optim.AdamW(ddp_model2.parameters(), lr=0.001)

        # Test one step
        optimizer2.zero_grad()
        output = ddp_model2(input_tensor)
        target = torch.randint(0, 10, (batch_size,), device="cuda")
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer2.step()
        print(f"[Rank {rank}] ✅ Workaround successful - training step completed")

    except Exception as e:
        print(f"[Rank {rank}] Workaround failed: {type(e).__name__}: {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()