from std.memory import MutPointer
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives import block


def q4_k_q8_1_matvec(
    weights: MutPointer[UInt8, MutAnyOrigin],
    activations: MutPointer[UInt8, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
):
    var lane = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var partial = Float32(0)
    if lane < 16:
        var pair = lane // 4
        var offset = (lane % 4) * 4
        for k in range(10):
            var wbase = (row * 10 + k) * 144
            var factors = (weights + wbase).bitcast[Float16]()
            var scaled_dot = Float32(0)
            var scaled_sum = Float32(0)
            for half in range(2):
                var group = pair * 2 + half
                var scale: Int
                var minimum: Int
                if group < 4:
                    scale = Int(weights[wbase + 4 + group]) & 63
                    minimum = Int(weights[wbase + 8 + group]) & 63
                else:
                    scale = (Int(weights[wbase + 8 + group]) & 15) | ((Int(weights[wbase + group]) >> 6) << 4)
                    minimum = (Int(weights[wbase + 8 + group]) >> 4) | ((Int(weights[wbase + 4 + group]) >> 6) << 4)
                var abase = (k * 8 + group) * 36
                var delta = Float32((activations + abase).bitcast[Float16]()[0])
                var dot = Int32(0)
                var total = Int32(0)
                for segment in range(2):
                    for byte in range(4):
                        var index = offset + segment * 16 + byte
                        var q = (Int(weights[wbase + 16 + pair * 32 + index]) >> (half * 4)) & 15
                        var a = Int32((activations + abase + 4).bitcast[Int8]()[index])
                        dot += Int32(q) * a
                        total += a
                scaled_dot += delta * Float32(dot * Int32(scale))
                scaled_sum += delta * Float32(total * Int32(minimum))
            partial += Float32(factors[0]) * scaled_dot - Float32(factors[1]) * scaled_sum
    var result = block.sum[block_size=32](partial)
    if lane == 0:
        output[row] = result


# AOT-only specialization: fixed input width 2560, arbitrary grid-sized output rows.
def main() raises:
    var ctx = DeviceContext()
    var weights = ctx.enqueue_create_buffer[DType.uint8](1440)
    var activations = ctx.enqueue_create_buffer[DType.uint8](2880)
    var output = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[q4_k_q8_1_matvec](weights, activations, output, grid_dim=1, block_dim=32)
