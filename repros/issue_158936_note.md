# Issue #158936 - TorchAO FP8 + torch.compile + DDP failure

**GitHub Issue:** https://github.com/pytorch/pytorch/issues/158936
**Status:** Incomplete reproduction information

## Problem Description

**Title:** "TorchAO FP8 training + HF transformer model + torch.compile + DDP does not work"

**Labels:**
- `oncall: distributed`
- `triaged`
- `module: ddp`
- `module: floatx (formerly float8)`
- `pt2d-triage-nov2024`

## Issue Details

The issue is described as "a DDPOptimizer + TorchAO issue with tensor.meta['val'] propagation" but lacks:

- ❌ No reproduction script provided
- ❌ No error messages or stack traces
- ❌ No environment specifications
- ❌ No step-by-step reproduction steps
- ❌ References external issue: pytorch/ao#2586

## Cross-Repository Coordination

This issue appears to serve primarily as a cross-repository coordination mechanism rather than a standalone bug report. The full reproduction details would need to be obtained from:

- **External Reference:** pytorch/ao#2586
- **TorchAO Repository:** https://github.com/pytorch/ao

## Recommended Next Steps

1. **Check pytorch/ao#2586** for detailed reproduction steps
2. **Review TorchAO documentation** for FP8 + DDP integration examples
3. **Create minimal repro** based on TorchAO FP8 training examples
4. **Test combination** of TorchAO FP8 quantization + torch.compile + DDP

## Potential Reproduction Framework

```python
# Hypothetical reproduction based on issue title
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchao.quantization import quantize_, float8_weight_only

# Setup distributed training
dist.init_process_group("nccl")
model = SomeHuggingFaceTransformer()

# Apply TorchAO FP8 quantization
quantize_(model, float8_weight_only())

# Compile model
compiled_model = torch.compile(model)

# Wrap with DDP - this combination reportedly fails
ddp_model = DDP(compiled_model)

# Training loop would reveal the tensor.meta['val'] propagation issue
```

## Status

**Cannot reproduce without external dependency resolution.** This issue requires:
1. TorchAO installation
2. Access to pytorch/ao repository issue details
3. HuggingFace transformers setup
4. Specific FP8 configuration that triggers the bug