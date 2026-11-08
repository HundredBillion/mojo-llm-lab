"""Compare GGML CUDA and Mojo including identical activation quantization work."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
from run_benchmark import ROOT, Sensors, digest, run_case
from validate_ptx import validate_ptx


def validate_2b_fixtures(directory):
    manifest = json.loads((directory / 'fixtures.json').read_text())
    if (manifest.get('rows'), manifest.get('input_width'), manifest.get('gate_up')) != (6144, 2048, True):
        raise ValueError('2B qualification requires 6144 distinct gate/up rows of width 2048')
    for name, size in {'weights.bin': 6144 * 1152, 'gate.bin': 6144 * 1152, 'activations.bin': 2304}.items():
        path = directory / name
        record = manifest.get('files', {}).get(name, {})
        if path.stat().st_size != size or record.get('bytes') != size or record.get('sha256') != digest(path):
            raise ValueError('Invalid 2B fixture size/hash: ' + name)
        if name != 'activations.bin':
            payload = path.read_bytes()
            if len({payload[offset:offset+1152] for offset in range(0, size, 1152)}) != 6144:
                raise ValueError('2B qualification requires distinct rows: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/local-q4-cuda')
    parser.add_argument('--variant', choices=['dp4a', 'dp4a-128', 'unrolled', 'packed', 'fixed', 'prepacked', 'reduce', 'readonly', 'gate', 'gate-pair', 'gate-order', 'gate-order-2b', 'gate-order-2b-64', 'gate-order-2b-256'], default='dp4a')
    parser.add_argument('--fixtures', type=Path, default=ROOT/'build/q4-matvec')
    args = parser.parse_args()
    args.fixtures = args.fixtures.resolve()
    executable = 'q4-compare' + ('-128' if args.variant in ('dp4a-128', 'unrolled', 'packed', 'fixed', 'prepacked', 'reduce', 'readonly', 'gate', 'gate-pair', 'gate-order', 'gate-order-2b', 'gate-order-2b-64', 'gate-order-2b-256') else '') + ('-split' if args.variant == 'prepacked' else '') + ('-gate' if args.variant in ('gate', 'gate-pair', 'gate-order', 'gate-order-2b', 'gate-order-2b-64', 'gate-order-2b-256') else '')
    if args.variant.startswith('gate-order-2b'):
        threads = int(args.variant.rsplit('-', 1)[1]) if args.variant.endswith(('-64', '-256')) else 128
        executable = f'q4-compare-{threads}-gate-2b'
        validate_2b_fixtures(args.fixtures)
    rows, width = (6144, 2048) if args.variant.startswith('gate-order-2b') else (9728, 2560)
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
    entry = validate_ptx(ptx.read_text(), expected_parameters=4 if args.variant in ('prepacked', 'gate', 'gate-pair', 'gate-order', 'gate-order-2b', 'gate-order-2b-64', 'gate-order-2b-256') else 3)
    if args.variant.startswith('gate-order-2b') and not entry.startswith(args.variant.replace('-', '_').replace('gate_order', 'q4_k_q8_1_gate_order') + '_q4_'):
        raise ValueError('Unexpected 2B PTX entry')
    sensors = Sensors(0)
    result = {'status':'failed_or_stopped','variant':args.variant,'rows':rows,'input_width':width,'scope':('Fused Q4_K gate/up with SiLU and matched Q8_1 quantization; not whole inference' if args.variant in ('gate', 'gate-pair', 'gate-order', 'gate-order-2b', 'gate-order-2b-64', 'gate-order-2b-256') else 'Standalone Q4_K matvec including matched Q8_1 quantization; not fused gate/up or whole inference'),
              'fixture_sha256':{name:digest(args.fixtures/name) for name in (['weights.bin','activations.bin','gate.bin'] if args.variant in ('gate', 'gate-pair', 'gate-order', 'gate-order-2b', 'gate-order-2b-64', 'gate-order-2b-256') else ['weights.bin','activations.bin'])},'versions':versions,'ptx_sha256':digest(ptx),'benchmark_sha256':digest(ROOT/'build/q4-matvec'/executable), 'comparisons':[]}
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for label, control in [('control',True),('comparison',False)]:
            row = run_case(args,sensors,versions,ptx,entry,label,'q4_matvec',control,
                           benchmark_dir=ROOT/'build/q4-matvec',fixture_dir=args.fixtures,
                           executable=executable,input_path='/fixtures')
            native = json.loads((args.output / f'{label}.json').read_text())
            if (native.get('rows'), native.get('input_width')) != (rows, width):
                raise ValueError('Native benchmark dimensions do not match requested shape')
            result['comparisons'].append(row)
            print(json.dumps(row),flush=True)
        result['status']='passed'
    finally:
        sensors.nv.nvmlShutdown()
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
