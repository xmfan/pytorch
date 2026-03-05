# https://github.com/pytorch/pytorch/issues/176443
#
# Bug: Wrong argument validation in DeviceMesh._unflatten
# Location: torch/distributed/device_mesh.py (Line 1378)
#
# Current buggy code: elif isinstance(dim, str) and dim in not_none(self.mesh_dim_names):
# Correct logic:     elif isinstance(dim, str) and dim not in not_none(self.mesh_dim_names):
#
# The conditional statement checks if dim IS IN mesh_dim_names, but the error message
# indicates it should trigger when dim is NOT IN that collection - a logic inversion bug.
#
# ANALYSIS: Looking at lines 1378-1381 in device_mesh.py:
# ```python
# elif isinstance(dim, str) and dim in not_none(self.mesh_dim_names):
#     raise ValueError(
#         f"dim {dim} specified in `_unflatten` is not in {self.mesh_dim_names}"
#     )
# ```
# This raises an error when dim IS in mesh_dim_names, but says it's NOT in - clear logic bug!
#
# Run with: python repros/issue_176443.py

import torch

def demonstrate_logic_bug():
    """
    Demonstrate the logic bug without needing distributed setup.
    We'll simulate the buggy condition by extracting the core logic.
    """
    print("=== Demonstrating DeviceMesh._unflatten validation logic bug ===")
    print()

    # Simulate the mesh_dim_names from a typical DeviceMesh
    mesh_dim_names = ["dp", "tp"]

    def not_none(x):
        """Helper function that mimics the behavior in the actual code"""
        return x if x is not None else []

    # Extract the buggy validation logic from line 1378
    def buggy_validation(dim, mesh_dim_names):
        """This is the BUGGY logic from the actual code"""
        if isinstance(dim, str) and dim in not_none(mesh_dim_names):
            raise ValueError(f"dim {dim} specified in `_unflatten` is not in {mesh_dim_names}")

    def correct_validation(dim, mesh_dim_names):
        """This is the CORRECT logic (fix the bug)"""
        if isinstance(dim, str) and dim not in not_none(mesh_dim_names):
            raise ValueError(f"dim {dim} specified in `_unflatten` is not in {mesh_dim_names}")

    print(f"Simulated mesh_dim_names: {mesh_dim_names}")
    print()

    print("--- Testing VALID dimension name 'dp' ---")
    try:
        buggy_validation("dp", mesh_dim_names)
        print("✅ Buggy validation: VALID 'dp' was accepted")
    except ValueError as e:
        print(f"🚨 BUG REPRODUCED: VALID 'dp' was rejected by buggy logic!")
        print(f"   Error: {e}")
        print(f"   This is wrong - valid dimensions should be accepted")

    try:
        correct_validation("dp", mesh_dim_names)
        print("✅ Correct validation: VALID 'dp' was accepted")
    except ValueError as e:
        print(f"❌ Correct validation unexpectedly rejected 'dp': {e}")

    print()
    print("--- Testing INVALID dimension name 'invalid' ---")
    try:
        buggy_validation("invalid", mesh_dim_names)
        print("🚨 BUG REPRODUCED: INVALID 'invalid' was incorrectly accepted by buggy logic!")
        print("   This is wrong - invalid dimensions should be rejected")
    except ValueError as e:
        print(f"✅ Buggy validation: INVALID 'invalid' was rejected: {e}")

    try:
        correct_validation("invalid", mesh_dim_names)
        print("✅ Correct validation: INVALID 'invalid' was rejected")
    except ValueError as e:
        print(f"✅ Correct validation: INVALID 'invalid' was rejected: {e}")

    print()
    print("=== Analysis ===")
    print("The bug is in line 1378 of device_mesh.py:")
    print("  BUGGY:   elif isinstance(dim, str) and dim in not_none(self.mesh_dim_names):")
    print("  CORRECT: elif isinstance(dim, str) and dim not in not_none(self.mesh_dim_names):")
    print()
    print("Effect of the bug:")
    print("  - VALID dimension names get rejected with 'is not in' error message")
    print("  - INVALID dimension names get accepted (no validation)")
    print("  - This completely inverts the intended validation behavior")

def test_actual_devicemesh_if_possible():
    """Try to test actual DeviceMesh if environment allows"""
    print()
    print("=== Attempting to test actual DeviceMesh (may fail without proper setup) ===")

    try:
        # This will likely fail without distributed setup, but let's try
        from torch.distributed.device_mesh import DeviceMesh

        # Try to create a minimal device mesh without distributed init
        device_mesh = DeviceMesh("cpu", [0], mesh_dim_names=["dim0"])
        print("Successfully created DeviceMesh")

        # If we get here, try the actual bug
        try:
            # This should work but might fail due to the bug
            device_mesh._unflatten("dim0", [1], ["new_dim"])
            print("✅ Valid dimension name worked")
        except ValueError as e:
            if "not in" in str(e):
                print(f"🚨 BUG REPRODUCED in actual DeviceMesh: {e}")

        try:
            # This should fail but might work due to the bug
            device_mesh._unflatten("invalid", [1], ["new_dim"])
            print("🚨 BUG REPRODUCED: Invalid dimension was accepted")
        except ValueError as e:
            print(f"✅ Invalid dimension correctly rejected: {e}")

    except Exception as e:
        print(f"Cannot test actual DeviceMesh: {e}")
        print("This is expected if distributed environment is not set up")

def main():
    print("Testing DeviceMesh._unflatten validation logic bug...")
    print("Bug: Line 1378 has inverted logic - rejects valid dims, accepts invalid dims")
    print()

    demonstrate_logic_bug()
    test_actual_devicemesh_if_possible()

if __name__ == "__main__":
    main()