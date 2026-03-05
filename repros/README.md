# PyTorch Distributed + Compile Bug Reproductions

**Environment:** PyTorch 2.12.0a0+git46b4b1e, CUDA 12.1, 8x A100 GPUs
**Date:** 2026-03-04

## Directory Structure

```
repros/
├── REPRODUCED/     # 13 bugs that successfully reproduce (6 original + 4 pytorch + 3 pytorch/ao)
├── FIXED/          # 6 bugs that appear to be fixed (4 original + 2 new)
├── README.md       # This file
├── run_all_tests.sh # Automated test runner
└── issue_158936_note.md  # Note about incomplete TorchAO issue
```

## 🔴 REPRODUCED (13 files) - Active bugs

### 🆕 **NEW: PyTorch AO (Quantization) Issues** - Critical recent findings

| File | Issue | Error | Priority | Run Command |
|------|-------|-------|----------|-------------|
| **`issue_176276.py`** | [pytorch#176276](https://github.com/pytorch/pytorch/issues/176276) | **SECURITY: FileStore path traversal** | **🚨 CRITICAL** | `python REPRODUCED/issue_176276.py` |
| **`issue_ao_3982.py`** | [pytorch/ao#3982](https://github.com/pytorch/ao/issues/3982) | **🆕 Float8 + FSDP2 uneven sharding crash** | **🚨 CRITICAL** | `torchrun --nproc_per_node=3 REPRODUCED/issue_ao_3982.py` |
| **`issue_ao_2586.py`** | [pytorch/ao#2586](https://github.com/pytorch/ao/issues/2586) | **🆕 Float8 + HF + torch.compile + DDP fails** | **🔥 HIGH** | `torchrun --nproc_per_node=2 REPRODUCED/issue_ao_2586.py` |
| **`issue_ao_3926.py`** | [pytorch/ao#3926](https://github.com/pytorch/ao/issues/3926) | **🆕 DeviceMesh _mesh_dim_names missing** | **🔥 HIGH** | `torchrun --nproc_per_node=4 REPRODUCED/issue_ao_3926.py` |
| **`issue_176446.py`** | [pytorch#176446](https://github.com/pytorch/pytorch/issues/176446) | **DTensor tp_conv rejects ResNet configs** | **🔥 HIGH** | `python REPRODUCED/issue_176446.py` |
| **`issue_176443.py`** | [pytorch#176443](https://github.com/pytorch/pytorch/issues/176443) | **DeviceMesh validation logic inversion** | **⚠️ MEDIUM** | `python REPRODUCED/issue_176443.py` |
| `issue_155632.py` | [pytorch#155632](https://github.com/pytorch/pytorch/issues/155632) | all_gather shape mismatch | HIGH | `torchrun --nproc_per_node=2 REPRODUCED/issue_155632.py` |
| `issue_158519.py` | [pytorch#158519](https://github.com/pytorch/pytorch/issues/158519) | state_dict key prefix loss | HIGH | `torchrun --nproc_per_node=2 REPRODUCED/issue_158519.py` |
| `issue_152442.py` | [pytorch#152442](https://github.com/pytorch/pytorch/issues/152442) | checkpoint wrapper context_fn | HIGH | `PYTORCH_TEST_WITH_INDUCTOR=1 python REPRODUCED/issue_152442.py` |
| `issue_156680.py` | [pytorch#156680](https://github.com/pytorch/pytorch/issues/156680) | SyncBatchNorm fullgraph error | HIGH | `torchrun --nproc_per_node=2 REPRODUCED/issue_156680.py` |
| `issue_171934.py` | [pytorch#171934](https://github.com/pytorch/pytorch/issues/171934) | DTensor input mutation size mismatch | MEDIUM | `python REPRODUCED/issue_171934.py` |
| `issue_175469.py` | [pytorch#175469](https://github.com/pytorch/pytorch/issues/175469) | aten.linear DTensor strategy missing | MEDIUM | `torchrun --nproc_per_node=2 REPRODUCED/issue_175469.py` |
| `issue_172549.py` | [pytorch#172549](https://github.com/pytorch/pytorch/issues/172549) | DTensor alias AOTAutograd error | MEDIUM | `python REPRODUCED/issue_172549.py` |

## 🟢 FIXED (6 files) - No longer reproducing

| File | Issue | Original Problem | Status |
|------|-------|------------------|--------|
| `issue_148701.py` | [#148701](https://github.com/pytorch/pytorch/issues/148701) | aot_eager wrong all_gather gradient | ✅ Gradients now match |
| `issue_158551.py` | [#158551](https://github.com/pytorch/pytorch/issues/158551) | CUDAGraphs + DDP overwritten tensor | ✅ Training completes successfully |
| `issue_161937.py` | [#161937](https://github.com/pytorch/pytorch/issues/161937) | AOTAutograd alias reconstruction | ✅ No view compatibility errors |
| `issue_173123.py` | [#173123](https://github.com/pytorch/pytorch/issues/173123) | DTensor grad becomes torch.Tensor | ✅ Type preserved correctly |
| `issue_174732.py` | [#174732](https://github.com/pytorch/pytorch/issues/174732) | DTensor embedding IndexError | ✅ Compiles without error |
| `issue_175258.py` | [#175258](https://github.com/pytorch/pytorch/issues/175258) | Flex attention + SAC caching | ✅ Runs without RuntimeError |

## 🎯 Priority Action Items (All 13 Reproduced Issues)

**🚨 CRITICAL:**
1. **Fix FileStore path traversal** (pytorch#176276) - Security vulnerability
2. **🆕 Fix Float8 + FSDP2 uneven sharding** (pytorch/ao#3982) - Storage mismatch crashes

**🔥 HIGH:**
3. **🆕 Fix Float8 + HF + torch.compile + DDP** (pytorch/ao#2586) - Quantization training blocker
4. **🆕 Fix DeviceMesh _mesh_dim_names** (pytorch/ao#3926) - torch.compile compatibility
5. **Fix DTensor tp_conv validation** (pytorch#176446) - Blocks ResNet architectures
6. **Fix DeviceMesh logic inversion** (pytorch#176443) - Simple one-line fix
7. Fix all_gather stacking (pytorch#155632) - Shape mismatch in functional collectives
8. Fix state_dict key prefix (pytorch#158519) - Key normalization in compiled models
9. Fix checkpoint wrapper context_fn (pytorch#152442) - TorchDispatchMode requirements
10. Fix SyncBatchNorm fullgraph (pytorch#156680) - Compilation compatibility

**⚠️ MEDIUM:**
11. DTensor input mutations (pytorch#171934) - Compiled autograd tensor size issues
12. DTensor export decompositions (pytorch#175469) - Missing sharding strategies
13. AOTAutograd DTensor aliasing (pytorch#172549) - Metadata mismatch in tensor subclasses

### 🆕 **New Focus Area: Quantization + Distributed + Compile**
The pytorch/ao issues reveal critical gaps in the intersection of modern quantization (Float8, MXFP8) with distributed training (FSDP2, DDP) and compilation (torch.compile). These affect major use cases like large model training with quantization.

## Quick Test Commands

```bash
# Test all reproduced bugs (expect failures showing bugs)
for f in REPRODUCED/*.py; do
    echo "=== Testing $f ==="
    if [[ "$f" == *"152442"* ]]; then
        PYTORCH_TEST_WITH_INDUCTOR=1 python "$f"
    elif [[ "$f" == *"171934"* ]] || [[ "$f" == *"172549"* ]] || [[ "$f" == *"176"* ]]; then
        python "$f"
    else
        torchrun --nproc_per_node=2 --master_port=$((29500 + RANDOM % 100)) "$f"
    fi
    echo
done

# Test all fixed bugs (expect success)
for f in FIXED/*.py; do
    echo "=== Testing $f ==="
    if [[ "$f" == *"148701"* ]] || [[ "$f" == *"158551"* ]] || [[ "$f" == *"161937"* ]]; then
        torchrun --nproc_per_node=2 --master_port=$((29600 + RANDOM % 100)) "$f"
    elif [[ "$f" == *"173123"* ]]; then
        torchrun --nproc_per_node=4 --master_port=$((29600 + RANDOM % 100)) "$f"
    else
        torchrun --nproc_per_node=2 --master_port=$((29600 + RANDOM % 100)) "$f"
    fi
    echo
done

# Or run the comprehensive test suite
bash run_all_tests.sh
```

## Notes

- All scripts include the GitHub issue URL and detailed error descriptions
- Scripts are designed to be self-contained and show clear pass/fail status
- REPRODUCED bugs should show errors/failures (demonstrating the bugs)
- FIXED bugs should run successfully (showing the fixes work)
- Critical security bug (#176276) needs immediate attention
- Keep fixed bug scripts for regression testing