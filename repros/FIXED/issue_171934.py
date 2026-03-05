# https://github.com/pytorch/pytorch/issues/171934
#
# Bug: test_dtensor_input_mutations consistently fails on main.
# DTensor input mutations under compiled autograd fail with a tensor size
# mismatch error during fake tensor dispatch.
#
# Repro: from the pytorch repo root, run:
#   python test/distributed/tensor/test_dtensor_compile.py \
#       TestDTensorCompileWithCompiledAutograd.test_dtensor_input_mutations
#
# Error:
#   RuntimeError: The size of tensor a (((s35 + 1)//2)) must match the size
#   of tensor b (s35) at non-singleton dimension 0
#
# The error occurs in fake_impls.py during infer_size when compiled autograd
# tries to handle DTensor input mutations. The symbolic shapes don't match up
# correctly during the backward pass compilation.

import subprocess
import sys
import os

result = subprocess.run(
    [sys.executable, "test/distributed/tensor/test_dtensor_compile.py",
     "TestDTensorCompile.test_dtensor_input_mutations"],
    cwd="/home/dev/oncall/pytorch",
    capture_output=True, text=True, timeout=300,
    env={**os.environ}
)
print("STDOUT:", result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
print("STDERR:", result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
print("Return code:", result.returncode)
if result.returncode != 0:
    print("\nBUG REPRODUCED: Test failed")
else:
    print("\nTest passed -- bug may be fixed")
