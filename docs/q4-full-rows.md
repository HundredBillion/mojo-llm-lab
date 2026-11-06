# Full distinct-row qualification

The unchanged `gate-pair` Mojo kernel retained its measured gain when both projection matrices used **9728 distinct byte rows**, rather than repeating 64 rows.

| Run | Fused CUDA median | Fused Mojo median | CUDA / Mojo |
| --- | ---: | ---: | ---: |
| First | 250.416 µs | 231.056 µs | 1.08379× |
| Repeat | 250.896 µs | 230.512 µs | 1.08843× |

These are roughly 8–9% higher operation-throughput ratios, or about 7.7–8.1% less operation time. [First raw record](../results/a2000/q4-full-rows/) and [repeat](../results/a2000/q4-full-rows-repeat/) retain all samples and numerical logs. Control ratios were 1.00055 and 1.00260, within the unchanged gate. This strengthens the synthetic operation result; it does not establish whole-model acceleration.

## What changed

Only fixture row coverage and host input loading changed. The first 64 rows of both projections and all activation bytes remain identical to the earlier fixture. Additional rows receive independently generated metadata and quantized bytes. Bulk random-byte generation for the added rows avoids the CPU cost of generating every byte with a separate Python call. Original 64-row hashes remain unchanged.

Each weight file is 14,008,320 bytes. Coverage checks confirmed 9728 distinct byte rows in each, preserved original prefixes, and recorded [fixture hashes](../results/a2000/q4-full-rows/fixtures.json). Distinct bytes do not mean every row has a distinct mathematical output: the deliberate zero-gate rows still evaluate to zero.

The host accepts exactly 64 or 9728 rows, rejects truncated/trailing data, and requires gate and up files to contain the same row count. Only a 64-row input is tiled; full-row inputs are used directly. Raw measurement JSON now records `source_rows`; both new runs report 9728. The candidate PTX is unchanged from the paired-reduction experiment, as verified by its result hash.

## Correctness and remaining limits

Warmup and timed checks pass `absdiff <= 1e-5 + 1e-5 * abs(reference)`, and output canaries remain intact. Maximum absolute error is about **6.104e-5**, with 1817 of 9728 output values differing bitwise. Relative tolerance permits that absolute difference on sufficiently large outputs; the tolerance was not loosened. This remains tolerance agreement, not bit-identical agreement.

The fixture still uses power-of-two quantization factors and a fixed activation vector. General half-precision factors, varied activation distributions, real-model quality, token generation and complete accurate PDF latency remain outstanding. The fused CUDA baseline and activation-quantization work are unchanged. Do not extrapolate the isolated 1.08× ratio to complete inference or cooling savings.

## Thermal evidence, including the stopped attempt

The first byte-by-byte full-fixture preparation was stopped by the existing guard at a sampled **93°C CPU / 68°C GPU**, before any comparison. That attempt is not a performance measurement. The preparation was changed to bulk byte generation for added rows and a 10% CPU quota; it then completed at a peak of **77°C CPU / 68°C GPU**. This change concerns benchmark preparation, not LLM inference, and is not a permanent host setting.

The GPU runs peaked at CPU 62°C / GPU 71°C and CPU 63°C / GPU 72°C across their controls/comparisons. Sampled graphics clocks were 1207 MHz in the first comparison and 1200 MHz in the repeat; full samples are retained. AC and abort thresholds were unchanged. Sprite was resumed after every bounded pause, with independent resume watchdogs. No production setting changed.

## Reproduce

Use the pinned dependencies and external CPU/thermal protection for fixture generation and harness compilation:

```sh
python3 tools/generate_q4_matvec_fixtures.py --random-quants --gate-up --rows 9728 --output build/q4-full-rows
MOJO_BIN=.venv/bin/mojo bash tools/build_q4.sh gate-pair
python3 tools/build_q4_comparison.py --threads 128 --gate-up
python3 tools/run_q4_cuda_comparison.py --variant gate-pair --fixtures build/q4-full-rows --output results/local-full-rows
```

`--rows` defaults to 64 for compatibility. CPU tests verify prefix/activation preservation, row uniqueness on an extended fixture and invalid row-count rejection; the full-row coverage record and GPU comparison supply the larger-size evidence.
