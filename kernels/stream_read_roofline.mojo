# Bandwidth roofline probe: stream-read a buffer the size of the Qwen3.5-2B Q6_K vocabulary projection
# (248320 rows x 1680 bytes = 417,177,600 bytes) with coalesced 8-byte loads and no reuse.
# Each 128-thread block reads one 1680-byte-aligned span per grid-stride step, mirroring one-row-per-block reads.
# Not a matrix kernel; it bounds what any vocabulary-projection kernel could achieve on this GPU.
from std.memory import MutPointer
from max.gpu import block_idx, thread_idx, grid_dim
from max.gpu.host import DeviceContext

comptime WORDS = 417177600 // 8


def stream_read_roofline(
    data: MutPointer[UInt8, MutAnyOrigin],
    output: MutPointer[UInt64, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x) * 128
    var index = Int(block_idx.x) * 128 + tid
    var words = data.bitcast[UInt64]()
    var acc = UInt64(0)
    while index + 3 * stride < WORDS:
        acc ^= words[index] ^ words[index + stride] ^ words[index + 2 * stride] ^ words[index + 3 * stride]
        index += 4 * stride
    while index < WORDS:
        acc ^= words[index]
        index += stride
    output[Int(block_idx.x) * 128 + tid] = acc


def main() raises:
    var ctx = DeviceContext()
    var data = ctx.enqueue_create_buffer[DType.uint8](1680)
    var output = ctx.enqueue_create_buffer[DType.uint64](128)
    ctx.enqueue_function[stream_read_roofline](data, output, grid_dim=1, block_dim=128)
