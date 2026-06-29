#!/usr/bin/env bash
set -euo pipefail
python tests/test_reference.py
pytest -q tests/test_kernel_parity.py || echo "kernel parity needs a CUDA GPU"
python benchmarks/quality/run_needle.py --model meta-llama/Llama-3.1-8B-Instruct
python benchmarks/efficiency/bench_decode_latency.py --model meta-llama/Llama-3.1-8B-Instruct \
    --ctx 32768 65536 131072 --budget 0.0625
