#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

struct Fusion {
    const void *x_bias, *gate, *gate_bias, *x_scale, *gate_scale;
    int glu_op;
    float glu_limit;
    const void *shared_up, *shared_gate;
    float *shared_dst;
    uint32_t shared_stride_col_dst;
};
static_assert(sizeof(Fusion) == 80);
static_assert(offsetof(Fusion,gate)==8 && offsetof(Fusion,glu_op)==40 && offsetof(Fusion,shared_dst)==64);
template<class T> static T argument(void **args, int i) {
    T value;
    std::memcpy(&value, args[i], sizeof(value));
    return value;
}
static bool eligible(void **args) {
    const auto f = argument<Fusion>(args, 3);
    return argument<uint32_t>(args,5)==2560 && argument<uint32_t>(args,9)==9728 &&
        argument<uint32_t>(args,7)==10 && argument<uint32_t>(args,8)==80 &&
        !argument<const void *>(args,2) && argument<const void *>(args,0) &&
        argument<const void *>(args,1) && argument<void *>(args,4) && f.gate &&
        !f.x_bias && !f.gate_bias && !f.x_scale && !f.gate_scale &&
        !f.shared_up && !f.shared_gate && !f.shared_dst && f.glu_op==2 && f.glu_limit==0;
}
