"""
cd /fsx/users/willfeng2/pytorch_yf225/benchmarks/dynamo
PYTHONPATH=/fsx/users/willfeng2/benchmark:$PYTHONPATH HUGGING_FACE_HUB_TOKEN=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 python torchbench_mp_debug.py --performance --training --only llama_v2_7b_16h

PYTHONPATH=/fsx/users/willfeng2/benchmark:$PYTHONPATH HUGGING_FACE_HUB_TOKEN=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 python torchbench_mp_debug.py --performance --training --only llama_v2_7b_16h --ddp-bucket-cap-mb-for-compiled 100

PYTHONPATH=/fsx/users/willfeng2/benchmark:$PYTHONPATH HUGGING_FACE_HUB_TOKEN=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 python torchbench_mp_debug.py --performance --training --only stable_diffusion

"""

import argparse
import contextlib

import os
from typing import List, Optional, Tuple

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import _functional_collectives as collectives
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
import torchbenchmark
import subprocess
import functools
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from torchbench import TorchBenchmarkRunner
from common import parse_args, patch_torch_manual_seed, cast_to_fp16

def cleanup():
    torch.cuda.synchronize()

    # kill any running processes using gpu
    output = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    if output.stdout == '':
        return

    process_id_strs = [pid for pid in output.stdout.strip().split("\n")]

    for pid in process_id_strs:
        subprocess.run(
            ["kill", pid],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    print(f"Timed out, killed stuck processes={process_id_strs}")

def bench(
    iter_func,
    iters=8,
    warmup=5,
    profile=False,
    device=None,
    model_name=None,
):
    assert device is not None

    dynamo_config.suppress_errors = False

    f_ = iter_func

    repeat = 5
    f = lambda: [(f_(), torch.cuda.synchronize()) for _ in range(repeat)]
    import time

    # measure memory on cold run
    torch.cuda.reset_peak_memory_stats(device)
    f()
    torch.cuda.synchronize()
    f_gb = torch.cuda.max_memory_allocated(device) / 1e9

    for _ in range(warmup):
        f()

    if profile:
        if dist.get_rank() == 0:
            prof = torch.profiler.profile()
        else:
            prof = contextlib.nullcontext()
        with prof:
            f()
        if dist.get_rank() == 0:
            prof.export_chrome_trace(f"{model_name}.json")
    f_times = []

    for _ in range(iters):
        # Calculate the elapsed time
        torch.cuda.synchronize(device)
        begin = time.time()
        f()
        torch.cuda.synchronize(device)
        f_times.append(time.time() - begin)

    # avg_time = sum(f_times)*1000/repeat/len(f_times)
    # print(f"{model_name}: avg_time    : {avg_time} ms \t{f_gb}GB")

    return f_times, f_gb


def run_one_rank(
    my_rank,
    args,
    runner,
    compile
):
    global print
    if my_rank != 0:
        print = lambda *args, **kwargs: None

    torch.cuda.set_device(my_rank)
    device = torch.device(f"cuda:{my_rank}")

    os.environ["RANK"] = f"{my_rank}"
    os.environ["WORLD_SIZE"] = f"{args.world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    nccl_options = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", pg_options=nccl_options
    )

    (
        _,
        name,
        model,
        example_inputs,
        batch_size,
    ) = runner.load_model("cuda", args.only, batch_size=args.batch_size)

    model, example_inputs = cast_to_fp16(model, example_inputs)

    if args.accuracy:
        torch._inductor.config.fallback_random = True
        if args.only not in {
            "alexnet",
            "Background_Matting",
            "pytorch_CycleGAN_and_pix2pix",
            "pytorch_unet",
            "Super_SloMo",
            "vgg16",
            "Wav2Vec2ForCTC",
            "Wav2Vec2ForPreTraining",
            "sam",
        }:
            # some of the models do not support use_deterministic_algorithms
            torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False

        # Remove randomeness when torch manual seed is called
        patch_torch_manual_seed()

    runner.init_optimizer(name, device, model.parameters())
    runner.model_iter_fn = runner.forward_and_backward_pass

    def run_eager():
        if args.ddp:
            model_eager = DDP(
                model,
                device_ids=[my_rank],
                output_device=my_rank,
                # bucket_cap_mb=25,  # DDP default value
            )
        elif args.fsdp:
            if args.float16:
                dtype = torch.float16
            elif args.bfloat16:
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            print(f"FSDP with dtype={dtype}")

            mp_policy = MixedPrecision(
                param_dtype=dtype,
                # Gradient communication precision.
                reduce_dtype=dtype,
                # Buffer precision.
                buffer_dtype=dtype,
            )

            my_auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, recurse=True,
                min_num_params=int(1)
            )

            model_eager = FSDP(
                model,
                use_orig_params=True,
                device_id=torch.cuda.current_device(),
                mixed_precision=mp_policy,
                limit_all_gathers=True,
                auto_wrap_policy=my_auto_wrap_policy,
            )
            if torch._inductor.config.triton.cudagraphs:
                torch._inductor.config.triton.cudagraphs = False

        eager_times, f_gb = bench(
            lambda: runner.model_iter_fn(model_eager, example_inputs, collect_outputs=False),
            profile=args.export_profiler_trace,
            device=device,
            model_name=f"{args.only}_eager"
        )

        assert len(eager_times) == 8
        print(f"eager    : {(sum(eager_times) / len(eager_times) * 1000):.3f} ms \t{f_gb} GB")

    def run_compile():
        # NOTE: throws `daemonic processes are not allowed to have children` error at `AsyncCompile.warm_pool() -> pool._adjust_process_count()` if we don't set this to 1.
        inductor_config.compile_threads = 1
        torch._inductor.config.triton.cudagraphs = not args.disable_cudagraphs
        if not args.disable_cudagraphs:
            torch.profiler._utils._init_for_cuda_graphs()

        if args.ddp:
            model_compiled = DDP(
                torch.compile(model, mode="reduce-overhead"),
                device_ids=[my_rank],
                output_device=my_rank,
                bucket_cap_mb=args.ddp_bucket_cap_mb_for_compiled
            )
        elif args.fsdp:
            if args.float16:
                dtype = torch.float16
            elif args.bfloat16:
                dtype = torch.bfloat16
            else:
                dtype = torch.float32

            mp_policy = MixedPrecision(
                param_dtype=dtype,
                # Gradient communication precision.
                reduce_dtype=dtype,
                # Buffer precision.
                buffer_dtype=dtype,
            )

            my_auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, recurse=True,
                min_num_params=int(1)
            )

            model_compiled = FSDP(
                torch.compile(model),
                use_orig_params=True,
                device_id=torch.cuda.current_device(),
                mixed_precision=mp_policy,
                limit_all_gathers=True,
                auto_wrap_policy=my_auto_wrap_policy,
            )
            if torch._inductor.config.triton.cudagraphs:
                torch._inductor.config.triton.cudagraphs = False

        compiled_times, g_gb = bench(
            lambda: runner.model_iter_fn(model_compiled, example_inputs),
            profile=args.export_profiler_trace,
            device=device,
            model_name=f"{args.only}_compiled"
        )

        assert len(compiled_times) == 8
        print(f"compiled : {(sum(compiled_times) / len(compiled_times) * 1000):.3f} ms \t{g_gb} GB")

    run_compile() if compile else run_eager()

    # for i in range(len(eager_times)):
    #     speedup = eager_times[i] / compiled_times[i]
    #     if my_rank == 0:
    #         print(f"{speedup:.3f}x")

    # torch.cuda.synchronize()
    # torch._dynamo.reset()
    # print("done!")


def main(compile: bool):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--world-size", type=int, default=8)
    args = parse_args()
    args.world_size = 8

    runner = TorchBenchmarkRunner()
    runner.args = args

    processes = []
    for rank in range(args.world_size):
        p = torch.multiprocessing.get_context("spawn").Process(
            target=run_one_rank,
            args=(
                rank,
                args,
                runner,
                compile
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)

    for rank, p in enumerate(processes):
        p.join()

    cleanup()

if __name__ == "__main__":
    main(compile=False)
    main(compile=True)
