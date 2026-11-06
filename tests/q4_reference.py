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


def decode_q8_1(payload):
    if len(payload) != 36:
        raise ValueError('Q8_1 block must have exactly 36 bytes')
    delta, _ = struct.unpack_from('<ee', payload)
    if not math.isfinite(delta):
        raise ValueError('Nonfinite activation delta')
    return [delta * q for q in struct.unpack_from('<32b', payload, 4)]


def dot_q4_q8(weights, activations):
    if not weights or len(weights) % 144 or len(activations) != len(weights) // 144 * 288:
        raise ValueError('Mismatched Q4_K/Q8_1 block counts')
    total = []
    for block_index in range(len(weights) // 144):
        values = decode_block(weights[block_index*144:(block_index+1)*144])
        activation_values = []
        for group in range(8):
            start = (block_index*8+group)*36
            activation_values.extend(decode_q8_1(activations[start:start+36]))
        total.extend(w*a for w,a in zip(values, activation_values))
    return math.fsum(total)
