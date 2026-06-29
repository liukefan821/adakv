# AdaKV

**Query-aware, per-head-adaptive block-sparse attention for long-context LLM decode.**
Drop-in for HuggingFace models; keeps the full KV cache (no eviction) and attends to
only the blocks each query needs, with the per-head KV budget allocated adaptively.

> ⚠️ Working name — check PyPI / GitHub before publishing. Kernel is a v0 skeleton;
> validate on GPU (`tests/test_kernel_parity.py`) before trusting any speedup numbers.

```python
from transformers import AutoModelForCausalLM
from adakv.patch import patch_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
patch_model(model, block_size=16, avg_budget=8)   # decode now uses AdaKV
```

## Why

At 128K context, decode is memory-bound — latency tracks KV traffic, not FLOPs.
AdaKV cuts that traffic with three pieces (see `docs/design.md`):

1. **Centroid block estimator** — score a block by `q · mean_key`; cheaper and a
   sharper ranking signal than Quest's min/max bound.
2. **Adaptive per-head budget** — give flat "retrieval" heads more blocks, peaked
   "local" heads fewer, from one cheap entropy signal. Prior work uses fixed top-k.
3. **Fused ragged block-sparse Triton kernel** — gathers only selected blocks inside
   the kernel; handles a variable per-head block count in one launch.

## Install

```bash
conda create -n adakv python=3.11 -y && conda activate adakv
pip install -e .
# GPU extras for kernels + benchmarks:
pip install -e ".[gpu]"
```

## Validate

```bash
python tests/test_reference.py          # CPU: algorithm correctness (no GPU needed)
pytest -q tests/test_kernel_parity.py   # GPU: Triton kernel vs NumPy oracle
```

## Benchmarks

```bash
python benchmarks/quality/run_needle.py        --model <hf_id>
python benchmarks/efficiency/bench_decode_latency.py --model <hf_id> --ctx 32768 65536 131072
```

| Axis | Datasets / baselines | Target |
|---|---|---|
| Quality @ fixed KV budget | RULER, LongBench, Needle vs Full / StreamingLLM / H2O / SnapKV / **Quest** | within ~1 pt of full where Quest drops more; ≥ Quest at equal budget |
| Decode latency vs context | AdaKV vs FlashAttention-2/3 dense, vs Quest kernel | speedup grows with context; faster at 128K |
| Memory | KV bytes touched | large reduction at long context |

## Layout

```
adakv/        estimator · budget · selector · reference (NumPy oracle) · attention · patch · kernels/
benchmarks/   quality/ · efficiency/ · baselines/
tests/        CPU algorithm tests · GPU kernel-parity test
docs/         design.md (novelty + differentiation table + protocol)
```

## License

MIT
