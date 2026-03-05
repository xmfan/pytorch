# Distributed + Compiler Bug Bash

**Date:** 2026-03-03 (created) | 2026-03-05 (last verified)
**Environment:** PyTorch 2.12.0a0+git46b4b1e, CUDA 12.1, 4x H100 80GB
**Team size:** ~7 people
**Sources:** pytorch/pytorch, pytorch/torchtitan, pytorch/ao

---

## Executive Summary

| Metric | Count |
|--------|-------|
| Total issues tracked | 60+ |
| **Reproduced (active bugs)** | **10** |
| Confirmed fixed | 12 |
| Needs torchao upgrade | 2 |
| Skipped (hardware/env) | 8 |
| Feature requests / RFCs | 6 |
| Needs investigation | 5 |

**Repro scripts:** `repros/REPRODUCED/` (12 files) and `repros/FIXED/` (7 files)

### Verification Run (2026-03-05, 2x H100 80GB)

| Script | Result |
|--------|--------|
| issue_176276.py | **REPRODUCED** -- FileStore path traversal |
| issue_176443.py | **REPRODUCED** -- DeviceMesh logic inversion |
| issue_158519.py | **REPRODUCED** -- state_dict key prefix mismatch |
| issue_172549.py | **REPRODUCED** -- AOTAutograd DTensor aliasing |
| issue_152442.py | **REPRODUCED** -- checkpoint_wrapper compile error |
| issue_155632.py | **REPRODUCED** -- allgather stack vs concat shape mismatch |
| issue_156680.py | **REPRODUCED** -- SyncBatchNorm `is_current_stream_capturing` non-Tensor |
| issue_175469.py | **REPRODUCED** -- `aten.linear.default` no DTensor sharding strategy |
| issue_176446.py | **REPRODUCED** -- tp_conv rejects batch-dim-only sharding |
| issue_ao_3982.py | **REPRODUCED** -- Float8+FSDP2 storage mismatch (adapted to 2 GPUs) |
| issue_171934.py | **FIXED** -- DTensor input mutations now passes |
| issue_ao_3926.py | **FIXED** -- DeviceMesh `_mesh_dim_names` works (torchao 0.7.0) |
| issue_ao_2586.py | **INCONCLUSIVE** -- torchao 0.7.0 too old, hits fp8 mat dim error instead |

---

## 1. Reproduced Bugs -- Ready to Fix

### 1a. Critical / Security

| # | Issue | Problem | Difficulty | Repro |
|---|-------|---------|------------|-------|
| 1 | [pytorch#176276](https://github.com/pytorch/pytorch/issues/176276) | **SECURITY:** FileStore path traversal allows writing files outside intended directories | Easy | `issue_176276.py` |

### 1b. High Priority

| # | Issue | Problem | Difficulty | Repro |
|---|-------|---------|------------|-------|
| 2 | [pytorch#176446](https://github.com/pytorch/pytorch/issues/176446) | DTensor `tp_conv` rejects valid ResNet-style convolution configs with batch-dim-only sharding. Error: "Stride must be 1 when there is padding" | Medium | `issue_176446.py` |
| 3 | [pytorch#176443](https://github.com/pytorch/pytorch/issues/176443) | DeviceMesh `_unflatten` validation logic inverted -- rejects valid dims, accepts invalid. One-line fix: `in` -> `not in` at line 1378 | Easy | `issue_176443.py` |
| 4 | [pytorch#155632](https://github.com/pytorch/pytorch/issues/155632) + [#138795](https://github.com/pytorch/pytorch/issues/138795) | `all_gather_into_tensor` functional collective only implements concat, not stack-style. Fails under compile with explicit `group` arg | Easy | `issue_155632.py` |
| 5 | [pytorch#158519](https://github.com/pytorch/pytorch/issues/158519) | `get_model_state_dict` returns keys with `_orig_mod.` prefix before `set_model_state_dict` but without it after, on `torch.compile(model)` | Easy | `issue_158519.py` |
| 6 | [pytorch#152442](https://github.com/pytorch/pytorch/issues/152442) | `test_checkpoint_wrapper_args_kwargs` fails with `PYTORCH_TEST_WITH_INDUCTOR=1` -- checkpoint_wrapper doesn't pass compile-compatible `context_fn` | Medium | `issue_152442.py` |
| 7 | [pytorch#156680](https://github.com/pytorch/pytorch/issues/156680) | `nn.SyncBatchNorm` + `fullgraph=True` fails: `is_current_stream_capturing returned non-Tensor`. Related: [#161302](https://github.com/pytorch/pytorch/issues/161302) | Medium-Hard | `issue_156680.py` |

### 1c. Medium Priority

| # | Issue | Problem | Difficulty | Repro |
|---|-------|---------|------------|-------|
| 8 | [pytorch#172549](https://github.com/pytorch/pytorch/issues/172549) | AOTAutograd error when compiled graph has two DTensor outputs that alias each other (one differentiable, one not). Metadata mismatch in subclass desugaring | Medium | `issue_172549.py` |
| 9 | [pytorch#175469](https://github.com/pytorch/pytorch/issues/175469) | `run_decompositions()` fails because `aten.linear.default` lacks a DTensor sharding strategy. POC fix in PR #175591 | Medium | `issue_175469.py` |

### 1d. PyTorch AO (Quantization + Distributed)

| # | Issue | Problem | Difficulty | Repro |
|---|-------|---------|------------|-------|
| 10 | [ao#3982](https://github.com/pytorch/ao/issues/3982) | Float8 + FSDP2 crashes with uneven sharding. Storage mismatch + `tensor_to_amax` empty tensor failures. Reproduced on 2 GPUs with odd hidden dim | High | `issue_ao_3982.py` |

### 1e. Needs Newer torchao (currently have 0.7.0, issues need 0.17.0+)

| # | Issue | Problem | Difficulty | Repro |
|---|-------|---------|------------|-------|
| 11 | [ao#2586](https://github.com/pytorch/ao/issues/2586) | Float8 + HuggingFace + `torch.compile` + DDP fails with "RuntimeError: val". See also [pytorch#158936](https://github.com/pytorch/pytorch/issues/158936) | High | `issue_ao_2586.py` |
| 12 | [ao#3926](https://github.com/pytorch/ao/issues/3926) | DeviceMesh objects missing `_mesh_dim_names` during compilation. Blocks MoE training tests. **May be fixed** -- passes with torchao 0.7.0 on 2 GPUs | High | `issue_ao_3926.py` |

---

## 2. Confirmed Fixed (No Longer Reproducing)

| Issue | Original Problem | Verified |
|-------|------------------|----------|
| [pytorch#156649](https://github.com/pytorch/pytorch/issues/156649) | `randint_like` on DTensor fails during dynamo tracing | **CLOSED 2026-03-04** via PR #174446 |
| [pytorch#119023](https://github.com/pytorch/pytorch/issues/119023) | `DTensor.to_local()` graph breaks with `grad_placements` kwarg | Compiles with `fullgraph=True` |
| [pytorch#169635](https://github.com/pytorch/pytorch/issues/169635) | DTensor RNG compile fails without user-passed generator | Compiles and runs |
| [pytorch#162087](https://github.com/pytorch/pytorch/issues/162087) | `all_gather_into_tensor_coalesced` reversed args after torch function support | Test passes (still open, on mlazos todo) |
| [pytorch#174732](https://github.com/pytorch/pytorch/issues/174732) | DTensor embedding dynamic seqlen IndexError | `repros/FIXED/issue_174732.py` |
| [pytorch#148701](https://github.com/pytorch/pytorch/issues/148701) | `aot_eager` wrong gradient with `all_gather` | `repros/FIXED/issue_148701.py` |
| [pytorch#173123](https://github.com/pytorch/pytorch/issues/173123) | DTensor gradient becomes `torch.Tensor` in backward | `repros/FIXED/issue_173123.py` |
| [pytorch#158551](https://github.com/pytorch/pytorch/issues/158551) | CUDAGraphs + DDP overwritten tensor output | `repros/FIXED/issue_158551.py` |
| [pytorch#161937](https://github.com/pytorch/pytorch/issues/161937) | AOTAutograd alias reconstruction view error on non-contiguous base | `repros/FIXED/issue_161937.py` |
| [pytorch#175258](https://github.com/pytorch/pytorch/issues/175258) | Flex attention + SAC caching | `repros/FIXED/issue_175258.py` (partial) |
| [pytorch#171934](https://github.com/pytorch/pytorch/issues/171934) | `test_dtensor_input_mutations` disabled -- now passes as of 2026-03-05 | `repros/FIXED/issue_171934.py` |

---

## 3. Hardware-Specific Issues

Require specific GPU hardware to reproduce.

### B200

| Issue | Problem | Difficulty |
|-------|---------|------------|
| [torchtitan#2055](https://github.com/pytorch/torchtitan/issues/2055) | `autobucketing_reordering` compiler pass crashes with deterministic mode. GPU benchmarking banned under inductor deterministic mode | Easy |
| [torchtitan#1940](https://github.com/pytorch/torchtitan/issues/1940) | Llama4 compile PP+EP fails: `all_to_all_single` shape mismatch with dp2ep=4, pp=2. Workaround merged (PR #1895) | Hard |
| [torchtitan#2401](https://github.com/pytorch/torchtitan/issues/2401) | MXFP8 + TP + `torch.compile` error on DSV3. Spans torchao and DTensor internals | Hard |

### GB200 NVL72

| Issue | Problem | Difficulty |
|-------|---------|------------|
| [torchtitan#2272](https://github.com/pytorch/torchtitan/issues/2272) | Fused RMSNorm Triton kernel exceeds 232,448-byte shared memory limit. Workaround: `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0`. Fix in pytorch#176228 | Medium |

### SM90+ (H100 and above)

| Issue | Problem | Difficulty |
|-------|---------|------------|
| [torchtitan#1932](https://github.com/pytorch/torchtitan/issues/1932) | `torch._grouped_mm` illegal memory access with large tensors. Suspected int64 indexing issue in Triton kernels | Hard |
| [torchtitan#2312](https://github.com/pytorch/torchtitan/issues/2312) | DSV3 + SimpleFSDP compile fail on GPUs below SM90 -- MoE for-loop fallback uses `tolist()` device-to-host syncs. PR #2255 in progress | Hard |

### GH200

| Issue | Problem | Difficulty |
|-------|---------|------------|
| [pytorch#154266](https://github.com/pytorch/pytorch/issues/154266) | DTensor + dynamo failure specific to GH200 with non-power-of-2 world size | Medium |

---

## 4. Torchtitan Issues (Require torchtitan installation)

| Issue | Problem | Difficulty | Status |
|-------|---------|------------|--------|
| [torchtitan#2133](https://github.com/pytorch/torchtitan/issues/2133) | Shared parameters cause duplicate All-Gather/Reduce-Scatter in Simple FSDP. 13 comments, active discussion | Medium | Open |
| [torchtitan#2196](https://github.com/pytorch/torchtitan/issues/2196) | Qwen3-0.6B different outputs compiled vs eager (max diff 1.5625 in logits). Disappears with `aot_eager` | Hard | Open |
| [torchtitan#1769](https://github.com/pytorch/torchtitan/issues/1769) | 495+ unwaited collective warnings at process termination. DSV3+FSDP2+TP+compile specific | Medium | Open |
| [torchtitan#610](https://github.com/pytorch/torchtitan/issues/610) | FSDP2 + transformer block compile saves extra `(bs, seq_len, dim)` tensor per block vs SimpleFSDP | Medium | Open |
| [torchtitan#534](https://github.com/pytorch/torchtitan/issues/534) | 2D whole model compile fails at embedding layer. Long-standing deep DTensor issue | Hard | Open |
| [torchtitan#2089](https://github.com/pytorch/torchtitan/issues/2089) | SimpleFSDP + DSV3 + FlexAttention issue | Unknown | Open |

---

## 5. Investigation / Stretch Goals

### Open pytorch/pytorch issues (not yet reproduced)

| Issue | Problem | Category | Priority |
|-------|---------|----------|----------|
| [#151204](https://github.com/pytorch/pytorch/issues/151204) | Dynamo fails tracing `reduce_scatter` with non-uniform inputs -- no `reduce_scatter_v` functional collective | Functional collectives | Medium |
| [#146045](https://github.com/pytorch/pytorch/issues/146045) | Deduplicated collectives remain as dead code, still executed. Needs config-gated SPMD DCE | Inductor | Medium |
| [#167349](https://github.com/pytorch/pytorch/issues/167349) | AOTAutograd cache bypasses on `DTensor.to_local()` | Performance | Medium |
| [#167656](https://github.com/pytorch/pytorch/issues/167656) | Lack of tensor identity tracking in DebugMode | Debug/Tools | Medium |
| [#158936](https://github.com/pytorch/pytorch/issues/158936) | TorchAO FP8 + `torch.compile` + DDP failure (see also ao#2586) | FP8/DDP | High |
| [#159942](https://github.com/pytorch/pytorch/issues/159942) | Pipeline parallelism dynamic shapes | Pipeline | Medium |
| [#159635](https://github.com/pytorch/pytorch/issues/159635) | DTensor dynamic shapes with `torch.compile`. 11 comments, recent activity | DTensor | Medium |
| [#113737](https://github.com/pytorch/pytorch/issues/113737) | `torch.compile` + SAC: mutations in backward not preserved. Recent verification: eager SAC now detects mutation & errors. Needs design decision | SAC | High |
| [#152954](https://github.com/pytorch/pytorch/issues/152954) | DTensor placement propagation for `as_strided` missing. PR #167188 linked, 19 comments | DTensor | High |

### Open pytorch/ao issues (need torchao setup)

| Issue | Problem | Priority |
|-------|---------|----------|
| [ao#3810](https://github.com/pytorch/ao/issues/3810) | `fsdp_ep_tp` + compile test DeviceGuard error. Needs 8 GPUs | High |
| [ao#2393](https://github.com/pytorch/ao/issues/2393) | TP + FSDP + MXFP8 fails during compile. Needs torchtitan | High |
| [ao#2453](https://github.com/pytorch/ao/issues/2453) | MoE training crash with FSDP + float8 in torchtitan | High |
| [ao#1938](https://github.com/pytorch/ao/issues/1938) | FSDP2 FP8 compatibility when GPU count > out_features | Medium |
| [ao#2445](https://github.com/pytorch/ao/issues/2445) | Gradient Checkpoint makes FP8 Training Slow | Medium |
| [ao#1463](https://github.com/pytorch/ao/issues/1463) | Internal torchao tensor subclasses cause `torch.compile` errors | Medium |
| [ao#803](https://github.com/pytorch/ao/issues/803) | Support `quantize_()` for DTensor | Medium |

---

## 6. RFCs / Feature Requests / Design Discussions

| Issue | Topic |
|-------|-------|
| [pytorch#127772](https://github.com/pytorch/pytorch/issues/127772) | Run existing eager DTensor tests under `torch.compile` (test coverage tracker) |
| [pytorch#162859](https://github.com/pytorch/pytorch/issues/162859) | RFC: Support symmetric memory in `torch.compile` (26 comments) |
| [pytorch#176439](https://github.com/pytorch/pytorch/issues/176439) | RFC: Symmetric Memory explicit lending semantics |
| [pytorch#176419](https://github.com/pytorch/pytorch/issues/176419) | RFC: Verify thread safety of NCCL SymmMem backend |
| [pytorch#176418](https://github.com/pytorch/pytorch/issues/176418) | RFC: Enable Copy Engine all-gather in FSDP |
| [ao#3293](https://github.com/pytorch/ao/issues/3293) | NVFP4 Training Tracker |
| [ao#3290](https://github.com/pytorch/ao/issues/3290) | FP8 Blockwise Training Tracker |
| [ao#3379](https://github.com/pytorch/ao/issues/3379) | Next steps for MXFP8 MoE training |

---

## 7. Excluded Issues

| Issue | Reason |
|-------|--------|
| [torchtitan#534](https://github.com/pytorch/torchtitan/issues/534) | Long-standing deep DTensor issue |
| [pytorch#155679](https://github.com/pytorch/pytorch/issues/155679) | NVIDIA CUDA driver issue, awaiting CUDA 13.2 |
| [pytorch#154847](https://github.com/pytorch/pytorch/issues/154847) | AsyncCollectiveTensor recompile -- design discussion ongoing |
| [pytorch#152829](https://github.com/pytorch/pytorch/issues/152829) | Large architectural refactor (hardcoded DTensor in dynamo) |
| [pytorch#146693](https://github.com/pytorch/pytorch/issues/146693) | Partitioner moves collectives -- active design discussion |
| [pytorch#143134](https://github.com/pytorch/pytorch/issues/143134) | DTensor metadata mismatch -- no minimal repro |
| [pytorch#138813](https://github.com/pytorch/pytorch/issues/138813) | FSDP1+UNet recompiles -- large model, FSDP1 |

---

## 8. Suggested Team Assignments (7 people)

| Person | Primary Task | Stretch |
|--------|-------------|---------|
| P1 | #1 - FileStore path traversal (SECURITY) | #8 - AOTAutograd DTensor aliasing (#172549) |
| P2 | #4 - allgather stacking (#155632) | #9 - export decomp (#175469) |
| P3 | #3 - DeviceMesh logic fix (#176443) | #10 - Float8+FSDP2 (ao#3982) |
| P4 | #5 - state_dict keys (#158519) | #6 - checkpoint wrapper (#152442) |
| P5 | #2 - DTensor tp_conv (#176446) | #7 - SyncBatchNorm (#156680) |
| P6 | #11 - Float8+DDP+compile (ao#2586) | DTensor test suite (#127772) |
| P7 | Investigation: #151204, #146045, #152954 | DTensor dynamic shapes (#159635) |

---

## 9. Repro Files Reference

```
repros/
├── REPRODUCED/           # 12 active bugs (verified 2026-03-05)
│   ├── issue_152442.py   # checkpoint_wrapper compile
│   ├── issue_155632.py   # allgather stacking (torchrun 2 GPU)
│   ├── issue_156680.py   # SyncBatchNorm fullgraph (torchrun 2 GPU)
│   ├── issue_158519.py   # state_dict key prefix (no GPU)
│   ├── issue_172549.py   # AOTAutograd DTensor aliasing (1 GPU)
│   ├── issue_175469.py   # export decompositions (torchrun 2 GPU)
│   ├── issue_176276.py   # FileStore path traversal (no GPU)
│   ├── issue_176443.py   # DeviceMesh validation (no GPU)
│   ├── issue_176446.py   # DTensor tp_conv (mp.spawn 2 GPU)
│   ├── issue_ao_2586.py  # Float8+DDP+compile (needs torchao 0.17+)
│   ├── issue_ao_3926.py  # DeviceMesh _mesh_dim_names (needs torchao 0.17+)
│   └── issue_ao_3982.py  # Float8+FSDP2 uneven sharding (torchrun 2 GPU)
├── FIXED/                # 7 verified fixed
│   ├── issue_148701.py
│   ├── issue_158551.py
│   ├── issue_161937.py
│   ├── issue_171934.py   # newly fixed as of 2026-03-05
│   ├── issue_173123.py
│   ├── issue_174732.py
│   └── issue_175258.py
├── run_all_tests.sh
└── issue_158936_note.md
```

**Run commands:**
- Distributed tests: `torchrun --nproc_per_node=N repros/REPRODUCED/issue_*.py`
- Single-process tests: `python repros/REPRODUCED/issue_*.py`
