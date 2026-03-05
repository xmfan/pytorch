# https://github.com/pytorch/pytorch/issues/158519
#
# Bug: get_model_state_dict returns keys with `_orig_mod.` prefix before
# set_model_state_dict, but without it after, when using torch.compile(model).
# This violates idempotency of state dict round-tripping.
#
# Observed:
#   Before set_model_state_dict: {'_orig_mod.weight', '_orig_mod.bias'}
#   After set_model_state_dict:  {'weight', 'bias'}
#   The `_orig_mod.` prefix is stripped after round-tripping through set/get.
#
# Run with: python repros/issue_158519.py

import torch
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)

def main():
    torch.manual_seed(42)
    model = torch.nn.Linear(8, 8)
    model = torch.compile(model)

    before_set = get_model_state_dict(model)
    print(f"Keys before set_model_state_dict: {sorted(before_set.keys())}")

    set_model_state_dict(model, before_set)

    after_set = get_model_state_dict(model)
    print(f"Keys after set_model_state_dict:  {sorted(after_set.keys())}")

    before_keys = set(before_set)
    after_keys = set(after_set)
    if before_keys != after_keys:
        print(f"\nBUG REPRODUCED: Keys changed!")
        print(f"  Before: {before_keys}")
        print(f"  After:  {after_keys}")
    else:
        print("\nNo bug: keys match.")

if __name__ == "__main__":
    main()
