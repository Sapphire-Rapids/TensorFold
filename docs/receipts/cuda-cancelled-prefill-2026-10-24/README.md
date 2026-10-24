# Cancelled CUDA prefill retention

An interrupted prefill can retain its latest complete snapshot in the existing
prompt cache. On the tested Qwen checkpoint, one cancellation retained 6,016
tokens; two cancellations retained 10,752. Every completed retry produced the
same 32 output tokens as the uninterrupted, cache-disabled reference.

This is a draft lifecycle-fix receipt. Device coverage is one checkpoint on
GB10; it does not qualify every changed family or establish a performance gate.

## Source and environment

- Baseline: `3f2ceb85b8cf9534bc05380f0f791682ece2e536`.
- Validated candidate: `6ad99c5fc8ab7cfdff8f2eee8c06d6f9d37ecb22`.
  [source.sha256](source.sha256) identifies all 16 changed source files. The
  later receipt commit does not change those files.
- One NVIDIA GB10, SM121, 128 GB memory class; Ubuntu 24.04.5 LTS, aarch64,
  kernel `6.17.0-1032-nvidia`, driver `580.173.02`.
- Zig 0.17.0; CUDA toolkit `13.3.1.008`, nvcc 13.3/V13.3.73. NCCL 2.30.7
  and PyTorch `2.13.0a0+9186a08b2c.nv26.7.59513937` are installed in the
  container. Inference uses Zig/CUDA, without the PyTorch or MLX runtime.
- Checkpoint and tokenizer: `mlx-community/Qwen3.5-2B-MLX-4bit`, revision
  `93760be4f1f69842a46bc13dbdc0f19e291392a3`; affine 4-bit, group size 64,
  checkpoint dtype BF16. One rank, context 131,072, two server slots.
- Baseline binary SHA256:
  `b2e56ae40433506f0c01351ff48d21c2f510cda111d62cb385a216d7056b1244`.
  Candidate binary SHA256:
  `ee253dc9533999e182b831ac9008140c8feca6d954db5dc2b0680b152681afd6`.

Both native builds used the following command. The build summary reports
`safe native` for the executable despite the requested optimization flag;
these measurements are not labelled ReleaseFast performance.

```sh
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 \
  -Doptimize=ReleaseFast -j2 --prefix <output> --summary all
```

## Generated input and exactness

The request contains 5,000 generated numbered records and asks for four values.
It uses temperature 0, seed 1234, 32 output tokens, `ignore_eos: true`, streaming
usage and thinking disabled. The rendered prompt has 104,559 tokens. No user
conversations or model weights are included.

[synthetic-request.json.gz](synthetic-request.json.gz) and
[prompt-ids.json.gz](prompt-ids.json.gz) retain the exact chat body and tokenizer
output. The four `cold-*-request.json.gz` files contain the exact JSON values
sent to `/v1/completions`, including each input token array. Gzip files have
deterministic timestamps; [SHA256SUMS](SHA256SUMS) hashes the compressed files.

Each of the four cases below starts a fresh server process. Cache-disabled
cases use fresh KV state for each request and report zero cached tokens.
Cancellation cases use a 2 GiB prompt cache. A disconnect occurs after the
request is observed running, before any text output; the probe waits for the
server to become idle before retrying.

| Candidate case | Cancellations | Cached tokens on resumed reply | Resumed TTFT |
| --- | ---: | ---: | ---: |
| One cancellation | 1 × about 1 s | 6,016 | 136.571 s |
| Repeated cancellation | 2 × about 1 s | 10,752 | 131.192 s |

Within each cancellation case, the completed resumed reply, a subsequent
`draft: false` retry, and two concurrent default retries have the same full
token SHA256 as the baseline cache-disabled reply. The latter retries are warm:
they reuse 104,448 tokens. These comparisons establish output equality, but
positive speculative-work telemetry was not captured. They do not establish
that a drafter was active or that two cold prefills ran simultaneously.

## Cache-disabled timing observations

Process order was baseline cache-off, candidate once-cancelled, candidate
twice-cancelled, candidate cache-off. Each timing cell has one observation.
The 2K/8K/32K/64K ladder runs in that order within each cache-off process.

| Input tokens | Baseline TTFT (s) | Candidate TTFT (s) | Change | Baseline decode (tok/s) | Candidate decode (tok/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 0.288 | 0.291 | +1.02% | 614.46 | 611.09 |
| 8,192 | 1.463 | 1.503 | +2.75% | 131.16 | 131.17 |
| 32,768 | 12.067 | 12.285 | +1.80% | 118.05 | 115.92 |
| 65,536 | 45.369 | 44.802 | -1.25% | 78.00 | 77.31 |
| 104,559 | 125.594 | 123.686 | -1.52% | not measured | not measured |

All five paired output token hashes match. The cold ladder uses the existing
`bench_concurrent.stream` and `bench_cache.receipt` helpers. No repeated
`bench_openai.py` decode panel was run. Short outputs and the absence of
alternating repetitions prevent a performance-regression conclusion. In
particular, the once-cancelled retry above is slower than a cache-off full
prefill; this receipt claims retained progress and exactness, not a Qwen speedup.

## Reproduction

Use Python 3.11+ and the checked-out candidate's `tools` directory. Build the
baseline and candidate separately with the command above, and download the
checkpoint at the recorded revision. Set `SOURCE` to the candidate checkout,
`MODEL` to that checkpoint snapshot and `BINARY` to the relevant native build.

```sh
RECEIPT="$SOURCE/docs/receipts/cuda-cancelled-prefill-2026-10-24"
OUT="$SOURCE/receipt-out"
mkdir -p "$OUT"
gzip -dc "$RECEIPT/synthetic-request.json.gz" > "$OUT/request.json"
```

Start a new server before each probe invocation below, with `CACHE_GIB=0` for
the baseline-off/candidate-off cases and `CACHE_GIB=2` for cancellation cases.
Stop the preceding server before starting the next. The only model process on
the test GPU should be this server.

```sh
TENSORFOLD_CUDA_KERNELS=native "$BINARY" serve "$MODEL" \
  --backend cuda --host 127.0.0.1 --port 8910 --name cuda-resume-check \
  --context 131072 --parallel 2 --prompt-cache-gib "$CACHE_GIB" \
  --temperature 0 --no-thinking --metrics-open
```

Run these in order, switching the server binary/cache setting between cases:

```sh
python3 "$RECEIPT/probe.py" --source "$SOURCE" --out "$OUT/base" \
  --mode baseline-off --payload "$OUT/request.json"
python3 "$RECEIPT/probe.py" --source "$SOURCE" --out "$OUT/once" \
  --mode candidate-cache --cancel-count 1 --payload "$OUT/request.json" \
  --reference "$OUT/base/result.json"
python3 "$RECEIPT/probe.py" --source "$SOURCE" --out "$OUT/twice" \
  --mode candidate-cache --cancel-count 2 --payload "$OUT/request.json" \
  --reference "$OUT/base/result.json" --progress-reference "$OUT/once/result.json"
python3 "$RECEIPT/probe.py" --source "$SOURCE" --out "$OUT/candidate" \
  --mode candidate-off --payload "$OUT/request.json" \
  --reference "$OUT/base/result.json"
```

The probe is the unchanged client used for these measurements. Its SHA256 is
recorded with the other receipt files. Timing and retained-token counts depend
on the machine; exact output equality and advancing retained progress are the
asserted behaviors.

## Host checks and negative controls

On Linux, both commands completed with exit 0:

```sh
zig build test test-kernel-contract test-golden fuzz-cortheon \
  test-server-security test-pull-security test-lane-api-host \
  test-glm-snapshots-host -j6 --summary all
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 \
  -Doptimize=ReleaseFast -j6 --prefix <output> --summary all
```

The first completed 321/321 build steps, with 3,518 tests passing, zero failing
and 14 skipped. The plain build completed 150/150 steps. Logs preserve the
expected fault-injection stderr; the final summaries determine the outcome.

Earlier targeted host controls ran with Zig 0.17.0 on macOS:

- Disabling interrupted-prefix retention makes both lane-host cancellation
  tests fail: 240/242 pass, two fail, exit 1.
- Disabling GLM cancellation snapshot publication makes its new rank-ordering
  test fail: 12/13 pass, one fails, exit 1; restoring it gives 13/13 pass.
- Removing cancellation forwarding from the device wrapper and recorder makes
  three tests fail: 1,672 pass, five skip, three fail, exit 1; restoring it gives
  1,675 pass and five skip, exit 0.
- Formatting passed. Lean findings remain 438 on baseline and candidate, zero
  new; the nonzero baseline checker result is not reported as clean.

The original log contents are compressed alongside this file. Private source
and toolchain paths are replaced with placeholders; the substitution counts
are recorded in [evidence.json](evidence.json).

## Remaining gates

Other changed adapter checkpoints, other CUDA architectures, Metal M1–M5,
128K/full native-window prompts, active-drafter telemetry, peak serving memory,
alternating performance repetitions, last-release comparison and a standard
server comparison were not measured. Host tests do not replace those device
gates. No GPU kernels, arithmetic or precision rules change in this patch, so
there is no new kernel-contract registration or device `SUMMARY` claim.
