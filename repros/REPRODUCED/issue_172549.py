# https://github.com/pytorch/pytorch/issues/172549
#
# Bug: AOTAutograd error when compiled graph has two DTensor outputs that alias
# each other, one differentiable one not
#
# Error: AssertionError: Node mul was invalid, but is output
#
# Root cause: Metadata mismatch in AOTAutograd when handling tensor subclass
# desugaring with aliased outputs. Outer metadata calculates
# num_intermediate_bases=0, but inner metadata calculates
# num_intermediate_bases=1.
#
# Run with: python repros/issue_172549.py

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, DeviceMesh, Shard, distribute_tensor
from torch.testing._internal.distributed.fake_pg import FakeStore

def main():
    # Setup fake process group for single-GPU testing
    store = FakeStore()
    dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

    device_mesh = DeviceMesh("cuda", [0, 1])

    # Create DTensor input
    t1 = torch.randn(4, 4, device="cuda")
    dt1 = distribute_tensor(t1, device_mesh, [Shard(0)])
    dt1.detach_().requires_grad_()

    @torch.compile(backend="aot_eager")
    def f(x):
        # Convert to local, apply operation, convert back to DTensor
        out = x.to_local().sin()
        out_dt = DTensor.from_local(out, device_mesh=device_mesh, placements=[Shard(0)])
        # Return both detached and original - they alias but have different grad requirements
        return out_dt.detach(), out_dt

    try:
        print("Running compiled function with aliased DTensor outputs...")
        out1, out2 = f(dt1)
        print(f"Success: out1.shape={out1.shape}, out2.shape={out2.shape}")
        print("Bug may be fixed - no AssertionError occurred")
    except AssertionError as e:
        print(f"BUG REPRODUCED: AssertionError: {e}")
    except Exception as e:
        print(f"Other error: {type(e).__name__}: {e}")
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()