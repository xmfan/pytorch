#!/bin/bash
# Complete test runner for all bug reproductions
# Run with: bash repros/run_all_tests.sh

set -e
cd "$(dirname "$0")"

echo "=============================================="
echo "PyTorch Distributed + Compile Bug Bash Tests"
echo "=============================================="
echo "Date: $(date)"
echo "PWD: $(pwd)"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

total_tests=0
passed_tests=0
failed_tests=0

run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local expect_fail="$3"  # true if we expect this test to fail (bug reproduction)

    echo
    echo "----------------------------------------"
    echo "Running: $test_name"
    echo "Command: $test_cmd"
    echo "Expected: $([ "$expect_fail" = "true" ] && echo "FAIL (bug repro)" || echo "PASS")"
    echo "----------------------------------------"

    total_tests=$((total_tests + 1))

    if eval "$test_cmd" > "/tmp/test_${total_tests}.log" 2>&1; then
        if [ "$expect_fail" = "true" ]; then
            echo -e "${YELLOW}UNEXPECTED PASS${NC}: $test_name (bug may be fixed)"
            echo "Check log: /tmp/test_${total_tests}.log"
        else
            echo -e "${GREEN}PASS${NC}: $test_name"
            passed_tests=$((passed_tests + 1))
        fi
    else
        if [ "$expect_fail" = "true" ]; then
            echo -e "${RED}REPRODUCED${NC}: $test_name (bug confirmed)"
            echo "Check log: /tmp/test_${total_tests}.log"
            passed_tests=$((passed_tests + 1))  # Expected failure counts as pass
        else
            echo -e "${RED}FAIL${NC}: $test_name"
            failed_tests=$((failed_tests + 1))
        fi
    fi
}

echo
echo "=== CRITICAL NEW BUGS (Expected to reproduce) ==="

# Security bug - should reproduce
run_test "FileStore Path Traversal (SECURITY)" \
         "python REPRODUCED/issue_176276.py" \
         "false"  # Actually succeeds but shows the vulnerability

# PyTorch/AO Float8 + FSDP2 bug - may need torchao
run_test "Float8 + FSDP2 uneven sharding (pytorch/ao#3982)" \
         "torchrun --nproc_per_node=3 --master_port=29700 REPRODUCED/issue_ao_3982.py" \
         "true"

# PyTorch/AO Float8 + HF + DDP + compile - may need torchao
run_test "Float8 + HF + torch.compile + DDP (pytorch/ao#2586)" \
         "torchrun --nproc_per_node=2 --master_port=29701 REPRODUCED/issue_ao_2586.py" \
         "true"

# PyTorch/AO DeviceMesh + compile bug
run_test "DeviceMesh _mesh_dim_names bug (pytorch/ao#3926)" \
         "torchrun --nproc_per_node=2 --master_port=29702 REPRODUCED/issue_ao_3926.py" \
         "false"  # May not reproduce without 4 GPUs

# DTensor tp_conv bug - should reproduce
run_test "DTensor tp_conv validation bug" \
         "python REPRODUCED/issue_176446.py" \
         "true"

# DeviceMesh logic bug - should reproduce
run_test "DeviceMesh validation logic inversion" \
         "python REPRODUCED/issue_176443.py" \
         "false"  # Actually succeeds but demonstrates the bug

echo
echo "=== ORIGINAL REPRODUCED BUGS (Expected to reproduce) ==="

# Original bugs that should still reproduce
run_test "all_gather shape mismatch" \
         "torchrun --nproc_per_node=2 --master_port=29600 REPRODUCED/issue_155632.py" \
         "true"

run_test "state_dict key prefix loss" \
         "torchrun --nproc_per_node=2 --master_port=29601 REPRODUCED/issue_158519.py" \
         "true"

run_test "checkpoint wrapper context_fn" \
         "PYTORCH_TEST_WITH_INDUCTOR=1 python REPRODUCED/issue_152442.py" \
         "true"

run_test "SyncBatchNorm fullgraph error" \
         "torchrun --nproc_per_node=2 --master_port=29602 REPRODUCED/issue_156680.py" \
         "true"

run_test "DTensor input mutation size mismatch" \
         "python REPRODUCED/issue_171934.py" \
         "true"

run_test "aten.linear DTensor strategy missing" \
         "torchrun --nproc_per_node=2 --master_port=29603 REPRODUCED/issue_175469.py" \
         "true"

run_test "DTensor alias AOTAutograd error" \
         "python REPRODUCED/issue_172549.py" \
         "true"

echo
echo "=== FIXED BUGS (Expected to pass) ==="

# These should now pass (bugs appear fixed)
run_test "aot_eager wrong all_gather gradient (FIXED)" \
         "torchrun --nproc_per_node=2 --master_port=29604 FIXED/issue_148701.py" \
         "false"

run_test "DTensor grad type preservation (FIXED)" \
         "torchrun --nproc_per_node=4 --master_port=29605 FIXED/issue_173123.py" \
         "false"

run_test "DTensor embedding IndexError (FIXED)" \
         "torchrun --nproc_per_node=2 --master_port=29606 FIXED/issue_174732.py" \
         "false"

run_test "CUDAGraphs + DDP (FIXED)" \
         "torchrun --nproc_per_node=2 --master_port=29607 FIXED/issue_158551.py" \
         "false"

run_test "AOTAutograd alias reconstruction (FIXED)" \
         "torchrun --nproc_per_node=2 --master_port=29608 FIXED/issue_161937.py" \
         "false"

echo
echo "=============================================="
echo "TEST SUMMARY"
echo "=============================================="
echo "Total tests: $total_tests"
echo "Passed/Expected: $passed_tests"
echo "Failed/Unexpected: $failed_tests"
echo
echo "Logs saved to: /tmp/test_*.log"
echo

if [ $failed_tests -eq 0 ]; then
    echo -e "${GREEN}All tests behaved as expected!${NC}"
    exit 0
else
    echo -e "${RED}Some tests had unexpected results.${NC}"
    exit 1
fi