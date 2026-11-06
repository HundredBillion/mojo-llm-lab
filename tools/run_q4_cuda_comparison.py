"""Compare GGML CUDA and Mojo including identical activation quantization work."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
from run_benchmark import ROOT, Sensors, digest, run_case
from validate_ptx import validate_ptx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/local-q4-cuda')
    parser.add_argument('--variant', choices=['dp4a', 'dp4a-128', 'unrolled', 'packed', 'fixed', 'prepacked', 'reduce', 'readonly', 'gate'], default='dp4a')
    parser.add_argument('--fixtures', type=Path, default=ROOT/'build/q4-matvec')
    args = parser.parse_args()
    args.fixtures = args.fixtures.resolve()
    executable = 'q4-compare' + ('-128' if args.variant in ('dp4a-128', 'unrolled', 'packed', 'fixed', 'prepacked', 'reduce', 'readonly', 'gate') else '') + ('-split' if args.variant == 'prepacked' else '') + ('-gate' if args.variant == 'gate' else '')
    args.gpu = 0
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU compute busy')
    versions = json.loads((ROOT/'versions.json').read_text())
    ptxs = list((ROOT/'build'/('q4-'+args.variant)).glob('*.ptx'))
    if len(ptxs) != 1:
        raise RuntimeError('Build exactly one DP4A PTX artifact')
    ptx = ptxs[0]
    entry = validate_ptx(ptx.read_text(), expected_parameters=4 if args.variant in ('prepacked', 'gate') else 3)
    sensors = Sensors(0)
    result = {'status':'failed_or_stopped','variant':args.variant,'rows':9728,'input_width':2560,'scope':('Fused Q4_K gate/up with SiLU and matched Q8_1 quantization; not whole inference' if args.variant == 'gate' else 'Standalone Q4_K matvec including matched Q8_1 quantization; not fused gate/up or whole inference'),
              'fixture_sha256':{name:digest(args.fixtures/name) for name in (['weights.bin','activations.bin','gate.bin'] if args.variant == 'gate' else ['weights.bin','activations.bin'])},'versions':versions,'ptx_sha256':digest(ptx),'benchmark_sha256':digest(ROOT/'build/q4-matvec'/executable), 'comparisons':[]}
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for label, control in [('control',True),('comparison',False)]:
            row = run_case(args,sensors,versions,ptx,entry,label,'q4_matvec',control,
                           benchmark_dir=ROOT/'build/q4-matvec',fixture_dir=args.fixtures,
                           executable=executable,input_path='/fixtures')
            result['comparisons'].append(row)
            print(json.dumps(row),flush=True)
        result['status']='passed'
    finally:
        sensors.nv.nvmlShutdown()
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
