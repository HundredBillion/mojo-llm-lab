from std.memory import MutPointer
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives import block
from max.gpu.intrinsics import inlined_assembly


@always_inline
def read_u32(ptr: MutPointer[UInt8, MutAnyOrigin]) -> UInt32:
    return inlined_assembly["ld.global.nc.u32 $0, [$1];", UInt32,
        constraints="=r,l", has_side_effect=True](Int(ptr))


@always_inline
def read_i32(ptr: MutPointer[UInt8, MutAnyOrigin]) -> Int32:
    return inlined_assembly["ld.global.nc.s32 $0, [$1];", Int32,
        constraints="=r,l", has_side_effect=True](Int(ptr))


def q4_k_q8_1_matvec_readonly(
    weights: MutPointer[UInt8, MutAnyOrigin],
    activations: MutPointer[UInt8, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    var lane = tid % 16
    var row = Int(block_idx.x)
    var partial = Float32(0)
    if tid < 128:
        var pair = lane // 4
        var offset = (lane % 4) * 4
        comptime for step in range(2):
            var k = tid // 16 + step * 8
            if k < 10:
                var wbase = (row * 10 + k) * 144
                var factors = (weights + wbase).bitcast[Float16]()
                var scaled_dot = Float32(0)
                var scaled_sum = Float32(0)
                var scales_ptr = (weights + wbase + 4).bitcast[UInt16]()
                var index = pair & 1
                var s0 = UInt32(scales_ptr[index])
                var s2 = UInt32(scales_ptr[index + 2])
                var s4 = UInt32(scales_ptr[index + 4])
                var high = UInt32(0) - UInt32(pair >= 2)
                var scales = ((s0 & UInt32(0x3f3f)) & ~high) | (((s4 & UInt32(0x0f0f)) | ((s0 & UInt32(0xc0c0)) >> 2)) & high)
                var minima = ((s2 & UInt32(0x3f3f)) & ~high) | ((((s4 >> 4) & UInt32(0x0f0f)) | ((s2 & UInt32(0xc0c0)) >> 2)) & high)
                comptime for half in range(2):
                    var group = pair * 2 + half
                    var scale = Int32((scales >> UInt32(half * 8)) & UInt32(255))
                    var minimum = Int32((minima >> UInt32(half * 8)) & UInt32(255))
                    var abase = (k * 8 + group) * 36
                    var delta = Float32((activations + abase).bitcast[Float16]()[0])
                    var dot = Int32(0)
                    var total = Int32(0)
                    comptime for segment in range(2):
                        var index = offset + segment * 16
                        var packed_q = read_u32(weights + wbase + 16 + pair * 32 + index)
                        var q = Int32((packed_q >> UInt32(half * 4)) & UInt32(0x0f0f0f0f))
                        var a = read_i32(activations + abase + 4 + index)
                        dot = inlined_assembly["dp4a.s32.s32 $0, $1, $2, $3;", Int32,
                            constraints="=r,r,r,r", has_side_effect=False](q, a, dot)
                        total = inlined_assembly["dp4a.s32.s32 $0, $1, $2, $3;", Int32,
                            constraints="=r,r,r,r", has_side_effect=False](Int32(0x01010101), a, total)
                    scaled_dot += delta * Float32(dot * Int32(scale))
                    scaled_sum += delta * Float32(total * Int32(minimum))
                partial += Float32(factors[0]) * scaled_dot - Float32(factors[1]) * scaled_sum
    var result = block.sum[block_size=128, broadcast=False](partial)
    if tid == 0:
        output[row] = result


# AOT-only specialization: fixed input width 2560, arbitrary grid-sized output rows.
def main() raises:
    var ctx = DeviceContext()
    var weights = ctx.enqueue_create_buffer[DType.uint8](1440)
    var activations = ctx.enqueue_create_buffer[DType.uint8](2880)
    var output = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[q4_k_q8_1_matvec_readonly](weights, activations, output, grid_dim=1, block_dim=128)
