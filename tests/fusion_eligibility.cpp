#include "../integration/fusion_eligibility.h"
#include <cassert>
#include <initializer_list>

int main() {
    int storage=0;
    const void *up=&storage,*input=&storage,*ids=nullptr;
    void *output=&storage;
    Fusion fusion{}; fusion.gate=&storage; fusion.glu_op=2;
    uint32_t width=2560,rows=9728,ws=10,as=80;
    void *args[19]{};
    args[0]=&up; args[1]=&input; args[2]=&ids; args[3]=&fusion;
    args[4]=&output; args[5]=&width; args[7]=&ws; args[8]=&as; args[9]=&rows;
    assert(eligible(args));
    for (auto p : {&width,&rows,&ws,&as}) {
        auto saved=*p; ++*p; assert(!eligible(args)); *p=saved;
    }
    for (auto p : {&up,&input,&fusion.gate}) {
        auto saved=*p; *p=nullptr; assert(!eligible(args)); *p=saved;
    }
    output=nullptr; assert(!eligible(args)); output=&storage;
    for (auto p : {&ids,&fusion.x_bias,&fusion.gate_bias,&fusion.x_scale,
                   &fusion.gate_scale,&fusion.shared_up,&fusion.shared_gate}) {
        *p=&storage; assert(!eligible(args)); *p=nullptr;
    }
    float dummy=0;
    fusion.shared_dst=&dummy; assert(!eligible(args)); fusion.shared_dst=nullptr;
    fusion.glu_op=1; assert(!eligible(args)); fusion.glu_op=2;
    fusion.glu_limit=1; assert(!eligible(args)); fusion.glu_limit=0;
    assert(eligible(args));
    width=2048; rows=6144; ws=8; as=64;
    assert(!eligible(args));
    assert(eligible(args,layout_2b));
    for (auto p : {&width,&rows,&ws,&as}) {
        auto saved=*p; ++*p; assert(!eligible(args,layout_2b)); *p=saved;
    }
}
