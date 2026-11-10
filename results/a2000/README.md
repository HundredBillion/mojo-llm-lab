# RTX A2000 measurements

All inputs are synthetic. These results measure one cached Q operation, not whole-model inference or PDF latency.

- `result-*.json`: original investigation harness, llama.cpp revision `d2e54583c7452353eb35d40431281f6ee984332f` unless the filename contains `b12110`.
- `result-b12110-*.json`: original harness against revision `cd959bae99ff4bed22bef2604f45241c09bbb8c8`.
- `public-reproduction/`: independent rebuild and run of this repository's portable C/Mojo sources, using the pins in `versions.json` and standard NVIDIA CDI access. The portable harness initializes its own blocking CUDA context and discovers the selected device PCI address.

The public reproduction passed all three numerical cases, output canaries, and the same-graph control. Each candidate comparison measured 3.976 versus 2.072 microseconds (1.919×); the control ratio was 1.000. Sampled GPU clocks were 1207 MHz. Observed CPU/GPU peaks were 79°C/63°C; these short experiments do not establish sustained thermal improvement. Full sensor samples and artifact hashes are included.

Reproduction command after the README setup:

```sh
python3 tools/run_benchmark.py --output results/local-reproduction
```

## Renamed identifiers

On 2026-11-10 the names of a private application were removed from this repository. In recorded logs, helper scripts, folder names and write-ups, its service, container-label, path and prompt names were replaced with neutral ones (for example `llm-fan.service`, `io.local.benchmark`, `/path/to/application`, `statement-short`). The recording machine's home directory and checkout path were likewise replaced with `/home/user` and `/path/to/mojo-llm-lab`. No measurement, hash or sensor value was changed. Helper scripts under `helpers/` are records of what ran and now name services that may not exist on another machine.
