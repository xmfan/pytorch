# https://github.com/pytorch/pytorch/issues/173123
#
# Bug: DTensor gradient of unused output becomes plain torch.Tensor in backward,
# causing type mismatch with torch.compile
#
# Error: Expected type: <class 'torch.distributed.tensor.DTensor'>
#        Runtime metadata: None, runtime type: <class 'torch.Tensor'>
#
# When an operation with multiple DTensor outputs only uses some downstream,
# unused output gradients are materialized as plain torch.Tensor instead of
# preserving DTensor type, causing type mismatches during backward passes.
#
# Based on gist: https://gist.github.com/fegin/ac5ded1a7bdb8686a94357043ef74b50
# Run with: torchrun --nproc_per_node=4 repros/issue_173123.py

import torch
import torch.distributed as dist
from torch.distributed._tensor import init_device_mesh, DTensor, Shard

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    mesh = init_device_mesh("cuda", (world_size,))

    # Create DTensor input
    input_tensor = torch.randn(8, 16, device="cuda", requires_grad=True)
    dt_input = DTensor.from_local(input_tensor, mesh, [Shard(0)])

    print(f"[Rank {rank}] Created DTensor input: {dt_input.shape}")

    def model_with_multiple_outputs(x):
        """
        Function that returns multiple DTensor outputs but only some are used downstream.
        This triggers the bug where unused output gradients become plain torch.Tensor.
        """
        # Some operation that produces multiple outputs
        y1 = x * 2
        y2 = x + 1  # This output won't be used downstream
        y3 = x.sin()

        return y1, y2, y3

    def downstream_computation(y1, y2, y3):
        """
        Only use y1 and y3, leaving y2 unused.
        This should cause y2's gradient to be a plain torch.Tensor instead of DTensor.
        """
        # Only use y1 and y3, y2 is unused
        result = y1.sum() + y3.sum()
        return result

    try:
        print(f"[Rank {rank}] Running eager version...")

        # Eager version - should work
        y1, y2, y3 = model_with_multiple_outputs(dt_input)
        loss = downstream_computation(y1, None, y3)  # y2 unused
        loss.backward()
        print(f"[Rank {rank}] Eager backward: OK")

        # Reset gradients
        dt_input.grad = None

        print(f"[Rank {rank}] Running compiled version...")

        # Compiled version - should fail due to type mismatch
        @torch.compile(backend="aot_eager")
        def compiled_model_and_loss(x):
            y1, y2, y3 = model_with_multiple_outputs(x)
            return downstream_computation(y1, None, y3)

        compiled_loss = compiled_model_and_loss(dt_input)
        compiled_loss.backward()
        print(f"[Rank {rank}] Compiled backward: OK - bug may be fixed")

    except Exception as e:
        error_msg = str(e)
        if "Expected type:" in error_msg and "DTensor" in error_msg and "torch.Tensor" in error_msg:
            print(f"[Rank {rank}] BUG REPRODUCED: DTensor type mismatch")
            print(f"[Rank {rank}] Error: {e}")
        else:
            print(f"[Rank {rank}] Other error: {type(e).__name__}: {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()