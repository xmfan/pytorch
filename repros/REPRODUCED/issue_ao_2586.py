# https://github.com/pytorch/ao/issues/2586
#
# Bug: float8 training + HF transformer model + torch.compile + DDP does not work
# Created: 2025-07-23
# Referenced in: pytorch/pytorch#158936
#
# Error: RuntimeError: val during FX graph execution in float8 scaling operations
#
# The combination of HuggingFace transformer + DDP + torch.compile (eager backend) +
# float8 training fails with RuntimeError during submod_8 execution.
#
# Works with: toy linear layers, FSDP (instead of DDP), or with float8/compile disabled individually
# Fails with: HF transformer + DDP + torch.compile + float8 together
#
# Full reproduction script: https://gist.github.com/vkuzo/9a1154fe08b654abcc9628f8a4834e83
# Run with: torchrun --nproc_per_node=2 repros/REPRODUCED/issue_ao_2586.py

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

def main():
    try:
        # Check if required packages are available
        try:
            from torchao.float8 import convert_to_float8_training
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("This test requires torchao and transformers. Install with:")
            print("pip install torchao --pre --index-url https://download.pytorch.org/whl/nightly/cu121")
            print("pip install transformers")
            return

        # Initialize distributed
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)

        print(f"[Rank {rank}] Testing Float8 + HF Transformer + torch.compile + DDP...")

        # Create a small HuggingFace transformer model (based on the gist)
        print(f"[Rank {rank}] Creating HuggingFace LLaMA model...")
        config = LlamaConfig(
            vocab_size=1000,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=8,
            max_position_embeddings=512,
            torch_dtype=torch.bfloat16,
        )

        model = LlamaForCausalLM(config).cuda().bfloat16()
        print(f"[Rank {rank}] Model created, parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Convert to Float8 training
        print(f"[Rank {rank}] Converting to Float8 training...")
        try:
            convert_to_float8_training(model)
            print(f"[Rank {rank}] Float8 conversion: OK")
        except Exception as e:
            print(f"[Rank {rank}] Float8 conversion failed: {type(e).__name__}: {e}")
            dist.destroy_process_group()
            return

        # Wrap with DDP
        print(f"[Rank {rank}] Wrapping with DDP...")
        try:
            ddp_model = DDP(model, device_ids=[rank])
            print(f"[Rank {rank}] DDP wrapping: OK")
        except Exception as e:
            print(f"[Rank {rank}] DDP wrapping failed: {type(e).__name__}: {e}")
            dist.destroy_process_group()
            return

        # Compile with torch.compile (eager backend as mentioned in the issue)
        print(f"[Rank {rank}] Compiling with torch.compile (aot_eager backend)...")
        try:
            compiled_model = torch.compile(ddp_model, backend="aot_eager")
            print(f"[Rank {rank}] Model compilation: OK")
        except Exception as e:
            print(f"[Rank {rank}] Model compilation failed: {type(e).__name__}: {e}")
            dist.destroy_process_group()
            return

        # Setup optimizer
        optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=1e-4)

        # Create input
        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")

        print(f"[Rank {rank}] Starting training step...")

        try:
            optimizer.zero_grad()

            # Forward pass - this is where the bug typically occurs
            outputs = compiled_model(input_ids)
            logits = outputs.logits
            print(f"[Rank {rank}] Forward pass: OK, logits shape: {logits.shape}")

            # Create dummy labels and compute loss
            labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, config.vocab_size), labels.view(-1))
            print(f"[Rank {rank}] Loss computation: OK, loss: {loss.item():.4f}")

            # Backward pass
            loss.backward()
            print(f"[Rank {rank}] Backward pass: OK")

            # Optimizer step
            optimizer.step()
            print(f"[Rank {rank}] ✅ Optimizer step: OK")
            print(f"[Rank {rank}] Training step completed successfully - bug may be fixed!")

        except RuntimeError as e:
            error_msg = str(e)
            if "val" == error_msg.strip():
                print(f"[Rank {rank}] 🚨 BUG REPRODUCED: RuntimeError: val")
                print(f"[Rank {rank}] This is the classic Float8 + HF + DDP + compile error!")
                print(f"[Rank {rank}] Full error: {e}")
            else:
                print(f"[Rank {rank}] Other RuntimeError: {e}")
        except Exception as e:
            print(f"[Rank {rank}] Unexpected error: {type(e).__name__}: {e}")

        # Test individual components to confirm diagnosis
        print(f"\n[Rank {rank}] Testing individual components for comparison...")

        # Test 1: Float8 + DDP + HF (no compile) - should work
        try:
            print(f"[Rank {rank}] Test 1: Float8 + DDP + HF (no compile)...")
            non_compiled_model = DDP(LlamaForCausalLM(config).cuda().bfloat16(), device_ids=[rank])
            convert_to_float8_training(non_compiled_model)
            outputs = non_compiled_model(input_ids)
            outputs.logits.sum().backward()
            print(f"[Rank {rank}] ✅ Test 1 passed: Float8 + DDP + HF works without compile")
        except Exception as e:
            print(f"[Rank {rank}] ❌ Test 1 failed: {e}")

    except Exception as e:
        print(f"[Rank {rank}] Setup error: {type(e).__name__}: {e}")

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()