# https://github.com/pytorch/ao/issues/3926
#
# Bug: torch.compile + DeviceMesh bug: "'DeviceMesh' object has no attribute '_mesh_dim_names'"
# Created: 2026-02-20
#
# This is a PyTorch distributed + compile compatibility issue that affects torchao's
# MoE training tests. The DeviceMesh object lacks the required _mesh_dim_names attribute
# when accessed during torch.compile, causing InternalTorchDynamoError.
#
# Full test command: torchrun --nproc_per_node=4 -m pytest test/prototype/moe_training/test_distributed.py -s -v
# Error: torch._dynamo.exc.InternalTorchDynamoError: AttributeError: 'DeviceMesh' object has no attribute '_mesh_dim_names'
#
# Run with: torchrun --nproc_per_node=4 repros/REPRODUCED/issue_ao_3926.py

import torch
import torch.nn as nn
import torch.distributed as dist

def main():
    try:
        # Initialize distributed
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)

        print(f"[Rank {rank}] Testing DeviceMesh + torch.compile compatibility...")

        # Create a DeviceMesh (this is the core of the bug)
        try:
            from torch.distributed.device_mesh import init_device_mesh
            mesh_2d = init_device_mesh("cuda", (1, world_size), mesh_dim_names=["dp", "tp"])
            print(f"[Rank {rank}] DeviceMesh created: {mesh_2d}")
        except Exception as e:
            print(f"[Rank {rank}] DeviceMesh creation failed: {e}")
            # Create fallback mesh
            mesh_2d = init_device_mesh("cuda", (world_size,))
            print(f"[Rank {rank}] Created fallback 1D mesh: {mesh_2d}")

        # Create a simple model that uses the DeviceMesh in a way that torch.compile will inspect
        class ModelWithMesh(nn.Module):
            def __init__(self, mesh):
                super().__init__()
                self.mesh = mesh
                self.linear = nn.Linear(512, 512)

            def forward(self, x):
                # Access mesh properties that trigger the bug during compilation
                # This simulates what torchao's MoE training does
                try:
                    # This will likely trigger the _mesh_dim_names access
                    mesh_str = str(self.mesh)  # This calls __repr__ which accesses _mesh_dim_names
                    mesh_shape = self.mesh.shape

                    # Print mesh info (this forces evaluation during compile)
                    if hasattr(self.mesh, '_mesh_dim_names'):
                        dim_names = self.mesh._mesh_dim_names
                    else:
                        print(f"[Rank {rank}] 🚨 BUG REPRODUCED: DeviceMesh missing _mesh_dim_names attribute")
                        dim_names = None

                except AttributeError as e:
                    if "_mesh_dim_names" in str(e):
                        print(f"[Rank {rank}] 🚨 BUG REPRODUCED: {e}")

                return self.linear(x)

        model = ModelWithMesh(mesh_2d).cuda()

        # Test without compilation (should work)
        print(f"[Rank {rank}] Testing without compilation...")
        x = torch.randn(4, 512, device="cuda")
        try:
            output = model(x)
            print(f"[Rank {rank}] ✅ Non-compiled execution: OK")
        except Exception as e:
            print(f"[Rank {rank}] ❌ Non-compiled execution failed: {e}")

        # Test with torch.compile (should trigger the bug)
        print(f"[Rank {rank}] Testing with torch.compile...")
        try:
            compiled_model = torch.compile(model)
            output = compiled_model(x)
            print(f"[Rank {rank}] ✅ Compiled execution: OK - bug may be fixed")
        except Exception as e:
            error_msg = str(e)
            if "_mesh_dim_names" in error_msg:
                print(f"[Rank {rank}] 🚨 BUG REPRODUCED: DeviceMesh _mesh_dim_names attribute error")
                print(f"[Rank {rank}] Error: {e}")
            elif "InternalTorchDynamoError" in error_msg:
                print(f"[Rank {rank}] 🚨 BUG REPRODUCED: TorchDynamo internal error (likely mesh-related)")
                print(f"[Rank {rank}] Error: {e}")
            else:
                print(f"[Rank {rank}] Other compilation error: {e}")

        # Additional test: Direct mesh attribute access under compilation
        print(f"[Rank {rank}] Testing direct mesh attribute access...")

        @torch.compile
        def mesh_inspector(mesh):
            try:
                # This should trigger the bug
                return str(mesh)  # This calls __repr__ which needs _mesh_dim_names
            except AttributeError as e:
                print(f"[Rank {rank}] 🚨 Direct mesh access bug: {e}")
                return "error"

        try:
            mesh_str = mesh_inspector(mesh_2d)
            print(f"[Rank {rank}] Compiled mesh access: OK")
        except Exception as e:
            print(f"[Rank {rank}] Compiled mesh access failed: {e}")

    except Exception as e:
        print(f"[Rank {rank}] Setup error: {type(e).__name__}: {e}")

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()