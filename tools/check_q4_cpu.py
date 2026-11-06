"""Explicitly compare synthetic Q4_K decoding with the fetched GGML library."""
import ctypes
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from q4_reference import decode_block
from test_q4_reference import packed_block


def main():
    versions = json.loads((ROOT / 'versions.json').read_text())
    library = ROOT / 'build/upstream' / ('llama-' + versions['engine_release']) / 'libggml-base.so'
    decoder = ctypes.CDLL(str(library)).dequantize_row_q4_K
    decoder.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    decoder.restype = None
    for seed in range(64):
        data = packed_block([(seed+j*7) % 64 for j in range(8)],
                            [(63-seed+j*11) % 64 for j in range(8)])
        source = ctypes.create_string_buffer(data)
        output = (ctypes.c_float * 256)()
        decoder(source, output, 256)
        if list(output) != decode_block(data):
            raise RuntimeError(f'GGML layout mismatch for seed {seed}')
    print(json.dumps({'status': 'passed', 'blocks': 64, 'values': 16384,
                      'criterion': 'exact equality for dyadic factors',
                      'engine_revision': versions['engine_revision']}))


if __name__ == '__main__':
    main()
