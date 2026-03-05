# https://github.com/pytorch/pytorch/issues/152442
#
# Bug: test_checkpoint_wrapper_args_kwargs fails with PYTORCH_TEST_WITH_INDUCTOR=1.
# checkpoint_wrapper doesn't pass compile-compatible context_fn to
# torch.utils.checkpoint, which requires a tuple of two TorchDispatchModes
# in torch.compile mode.
#
# Repro: from the pytorch repo root, run:
#   PYTORCH_TEST_WITH_INDUCTOR=1 python test/distributed/fsdp/test_checkpoint_wrapper.py \
#       CheckpointWrapperTest.test_checkpoint_wrapper_args_kwargs
#
# Error:
#   AssertionError: In torch.compile mode, `context_fn` arg passed to
#   `torch.utils.checkpoint` must generate a tuple of two `TorchDispatchMode`s.
#
# The error occurs during AOT compilation when the checkpoint_wrapper's
# _checkpoint_without_reentrant_generator processes
# torch.ops.higher_order.tag_activation_checkpoint.

import subprocess
import sys
import os

env = {**os.environ, "PYTORCH_TEST_WITH_INDUCTOR": "1"}
result = subprocess.run(
    [sys.executable, "test/distributed/fsdp/test_checkpoint_wrapper.py",
     "CheckpointWrapperTest.test_checkpoint_wrapper_args_kwargs"],
    cwd="/home/dev/oncall/pytorch",
    capture_output=True, text=True, timeout=300,
    env=env
)
print("STDOUT:", result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
print("STDERR:", result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
print("Return code:", result.returncode)
if result.returncode != 0:
    print("\nBUG REPRODUCED: Test failed with PYTORCH_TEST_WITH_INDUCTOR=1")
else:
    print("\nTest passed -- bug may be fixed")
