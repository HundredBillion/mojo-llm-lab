"""Measure achievable GPU read bandwidth for a vocabulary-projection-sized buffer (roofline probe).

Loads the Mojo stream-read PTX, verifies its result exactly, and times it with CUDA events. Also times a driver
device-to-device copy (read + write) as a second reference. Short, guarded, numerical/timing probe only.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import re
import statistics
import subprocess as S
import time

from run_benchmark import cold_gate, safe, digest
from run_launch_observer import ModelSensors

ROOT = Path(__file__).resolve().parents[1]
BYTES = 248320 * 1680
WORDS = BYTES // 8
PATTERN = 0x5A5A5A5A


def expected_accumulators(grid, threads=128):
    """Each thread XORs ceil((WORDS - g) / stride) identical words: the pattern if that count is odd, else 0."""
    stride = grid * threads
    word = (PATTERN << 32) | PATTERN
    return [word if ((WORDS - g + stride - 1) // stride) % 2 else 0 for g in range(stride)]


def entry_name(text):
    entries = re.findall(r'\.visible\s+\.entry\s+(\w+)\s*\(', text)
    if len(entries) != 1:
        raise ValueError('Require exactly one visible entry')
    return entries[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ptx', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repetitions', type=int, default=30)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    if S.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU already in use')
    sensor = ModelSensors(0); samples = []
    read_sensor = sensor.read

    def recorded_read():
        sample = read_sensor(); samples.append(sample); return sample
    sensor.read = recorded_read
    cuda = C.CDLL('libcuda.so.1')

    def fn(name, types):
        f = getattr(cuda, name); f.argtypes = types; f.restype = C.c_int

        def call(*args):
            status = f(*args)
            if status:
                raise RuntimeError(f'{name} status {status}')
        return call
    ptr, u64, u, size = C.c_void_p, C.c_uint64, C.c_uint, C.c_size_t
    init = fn('cuInit', [u]); get = fn('cuDeviceGet', [C.POINTER(C.c_int), C.c_int])
    retain = fn('cuDevicePrimaryCtxRetain', [C.POINTER(ptr), C.c_int]); current = fn('cuCtxSetCurrent', [ptr])
    release = fn('cuDevicePrimaryCtxRelease_v2', [C.c_int]); alloc = fn('cuMemAlloc_v2', [C.POINTER(u64), size])
    free = fn('cuMemFree_v2', [u64]); memset = fn('cuMemsetD32_v2', [u64, u, size])
    dtoh = fn('cuMemcpyDtoH_v2', [ptr, u64, size]); dtod = fn('cuMemcpyDtoD_v2', [u64, u64, size])
    load = fn('cuModuleLoad', [C.POINTER(ptr), C.c_char_p]); func = fn('cuModuleGetFunction', [C.POINTER(ptr), ptr, C.c_char_p])
    launch = fn('cuLaunchKernel', [ptr, u, u, u, u, u, u, u, ptr, C.POINTER(ptr), C.POINTER(ptr)])
    sync = fn('cuCtxSynchronize', []); unload = fn('cuModuleUnload', [ptr])
    event_create = fn('cuEventCreate', [C.POINTER(ptr), u]); record = fn('cuEventRecord', [ptr, ptr])
    elapsed = fn('cuEventElapsedTime', [C.POINTER(C.c_float), ptr, ptr]); destroy = fn('cuEventDestroy_v2', [ptr])
    text = a.ptx.read_text(); entry = entry_name(text)
    result = {'status': 'failed_or_stopped', 'bytes': BYTES, 'ptx_sha256': digest(a.ptx), 'entry': entry,
              'scope': 'Read-bandwidth roofline for a 248320x1680-byte buffer; not a matrix kernel or inference result.',
              'kernels': {}}
    data = u64(); copy = u64(); output = u64(); module = ptr(); dev = C.c_int(); events = []
    try:
        cold_gate(sensor)
        init(0); get(C.byref(dev), 0); ctx = ptr(); retain(C.byref(ctx), dev); current(ctx)
        alloc(C.byref(data), BYTES); memset(data, PATTERN, BYTES // 4)
        load(C.byref(module), str(a.ptx).encode()); kernel = ptr(); func(C.byref(kernel), module, entry.encode())
        start = ptr(); end = ptr(); event_create(C.byref(start), 0); event_create(C.byref(end), 0); events += [start, end]

        def timed(run):
            values = []
            for repetition in range(a.repetitions + 3):
                if not safe(sensor.read()):
                    raise RuntimeError('Temperature or AC guard')
                record(start, None); run(); record(end, None); sync()
                ms = C.c_float(); elapsed(C.byref(ms), start, end)
                if repetition >= 3:
                    values.append(ms.value)
            return values
        for grid in (640, 1280, 2560):
            alloc(C.byref(output), grid * 128 * 8)
            args = (ptr * 2)(C.cast(C.byref(data), ptr), C.cast(C.byref(output), ptr))
            times = timed(lambda: launch(kernel, grid, 1, 1, 128, 1, 1, 0, None, args, None))
            host = (C.c_uint64 * (grid * 128))(); dtoh(host, output, C.sizeof(host))
            if list(host) != expected_accumulators(grid):
                raise RuntimeError('Stream-read result mismatch at grid %d' % grid)
            median = statistics.median(times)
            result['kernels'][f'mojo_stream_read_grid{grid}'] = {'median_ms': median, 'min_ms': min(times),
                                                                 'read_GBps_median': BYTES / median / 1e6, 'exact': True}
            free(output); output = u64()
        alloc(C.byref(copy), BYTES)
        times = timed(lambda: dtod(copy, data, BYTES))
        median = statistics.median(times)
        result['kernels']['driver_device_copy'] = {'median_ms': median, 'read_plus_write_GBps_median': 2 * BYTES / median / 1e6}
        result['status'] = 'passed'
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        for event in events:
            destroy(event)
        for buffer in (data, copy, output):
            if buffer.value:
                free(buffer)
        if module.value:
            unload(module)
        if dev.value or True:
            try:
                release(dev)
            except Exception:
                pass
        result['peaks'] = {k: max(s[k] for s in samples) for k in ['cpu_C', 'gpu_C', 'platform_C']} if samples else {}
        result['clocks_MHz'] = sorted({(s.get('sm_clock_MHz'), s.get('memory_clock_MHz')) for s in samples})[-3:]
        (out / 'sensors.json').write_text(json.dumps(samples) + '\n')
        (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'error', 'kernels', 'peaks')}, indent=1))
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
