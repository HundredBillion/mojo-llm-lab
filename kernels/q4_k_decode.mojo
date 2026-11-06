from std.memory import MutPointer
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext


def decode_q4_k(
    packed: MutPointer[UInt8, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
):
    var index = thread_idx.x
    var base = Int(block_idx.x) * 144
    var group = Int(index) // 32
    var lane = Int(index) % 32
    var factors = (packed + base).bitcast[Float16]()
    var d = Float32(factors[0])
    var minimum = Float32(factors[1])
    var scale: Int
    var offset: Int
    if group < 4:
        scale = Int(packed[base + 4 + group]) & 63
        offset = Int(packed[base + 8 + group]) & 63
    else:
        scale = (Int(packed[base + 8 + group]) & 15) | ((Int(packed[base + group]) >> 6) << 4)
        offset = (Int(packed[base + 8 + group]) >> 4) | ((Int(packed[base + 4 + group]) >> 6) << 4)
    var byte = Int(packed[base + 16 + (group // 2) * 32 + lane])
    var quant = (byte >> (4 * (group % 2))) & 15
    output[Int(block_idx.x) * 256 + Int(index)] = d * Float32(scale) * Float32(quant) - minimum * Float32(offset)


# AOT specialization only; the host stub must not be executed with uninitialized inputs.
def main() raises:
    var ctx = DeviceContext()
    var packed = ctx.enqueue_create_buffer[DType.uint8](144)
    var output = ctx.enqueue_create_buffer[DType.float32](256)
    ctx.enqueue_function[decode_q4_k](packed, output, grid_dim=1, block_dim=256)
