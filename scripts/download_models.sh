#!/usr/bin/env bash
# Move this file out of ~/Downloads into ~/Projects/adakv/scripts/ before running.
set -euo pipefail
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./models/llama-3.1-8b-instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct       --local-dir ./models/qwen2.5-7b-instruct
