from std.memory import MutPointer
from std.math import cos, exp2, log2, rsqrt, sin
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives import block


def fused_q_imrope_runtime_fastpow(
    q: MutPointer[Float32, MutAnyOrigin],
    gamma: MutPointer[Float32, MutAnyOrigin],
    positions: MutPointer[Int32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    theta_scale_input: MutPointer[Float32, MutAnyOrigin],
):
    var head = block_idx.x
    var lane = thread_idx.x
    var value = q[head * 128 + lane]
    var energy = block.sum[block_size=128](value * value)
    var reciprocal_norm = rsqrt(energy / Float32(128.0) + Float32(0.000001))
    if lane < 64:
        var axis = 3
        if lane % 3 == 1 and lane < 60:
            axis = 1
        elif lane % 3 == 2 and lane < 60:
            axis = 2
        elif lane % 3 == 0 and lane < 72:
            axis = 0
        # A runtime load preserves the CUDA fast-pow rounding behavior at long positions.
        var theta_scale = theta_scale_input[0]
        var angle = Float32(positions[axis]) * exp2(log2(theta_scale) * Float32(lane))
        var c = cos(angle)
        var s = sin(angle)
        var first = q[head * 128 + lane] * reciprocal_norm * gamma[lane]
        var second = q[head * 128 + lane + 64] * reciprocal_norm * gamma[lane + 64]
        output[head * 128 + lane] = first * c - second * s
        output[head * 128 + lane + 64] = first * s + second * c


# This host stub exposes the GPU specialization for AOT compilation; its inputs are uninitialized.
def main() raises:
    var ctx = DeviceContext()
    var q = ctx.enqueue_create_buffer[DType.float32](4096)
    var gamma = ctx.enqueue_create_buffer[DType.float32](128)
    var positions = ctx.enqueue_create_buffer[DType.int32](4)
    var output = ctx.enqueue_create_buffer[DType.float32](4096)
    var theta = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[fused_q_imrope_runtime_fastpow](
        q, gamma, positions, output, theta, grid_dim=32, block_dim=128
    )
