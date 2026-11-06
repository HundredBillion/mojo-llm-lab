"""Independent mathematical decoder for the pinned GGML Q4_K block layout."""
import math
import struct


def decode_block(payload):
    if len(payload) != 144:
        raise ValueError('Q4_K block must have exactly 144 bytes')
    d, minimum = struct.unpack_from('<ee', payload)
    if not math.isfinite(d) or not math.isfinite(minimum):
        raise ValueError('Nonfinite block factors')
    packed = payload[4:16]
    values = []
    for group in range(8):
        if group < 4:
            scale, offset = packed[group] & 63, packed[group + 4] & 63
        else:
            scale = (packed[group + 4] & 15) | ((packed[group - 4] >> 6) << 4)
            offset = (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4)
        for lane in range(32):
            byte = payload[16 + (group // 2) * 32 + lane]
            quant = (byte >> (4 * (group % 2))) & 15
            values.append(d * scale * quant - minimum * offset)
    return values
