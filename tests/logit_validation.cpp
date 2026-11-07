#include "../integration/logit_validation.h"
#include <cassert>
#include <limits>
int main() {
    float base[]={0,1,-2},close[]={0.00005f,1,-2},bad[]={0.01f,1,-2};
    LogitValidation v(3,2);
    v.observe(true,0,close,3); v.observe(false,0,base,3);
    assert(v.passed(1,1,1)); assert(v.candidate.values==3);
    v.observe(false,0,base,3); v.observe(true,0,close,3);
    assert(v.passed(1,2,2)); assert(!v.passed(2,2,2));
    v.observe(true,0,bad,3); assert(!v.passed(1,3,2));
    LogitValidation missing(3); missing.observe(true,0,base,3);
    assert(!missing.passed(1,1,1));
    LogitValidation unstable(3); unstable.observe(false,0,base,3);
    unstable.observe(false,0,bad,3); unstable.observe(true,0,base,3);
    assert(!unstable.passed(1,1,2));
    bool threw=false;
    try { v.observe(false,0,base,2); } catch(const std::runtime_error &) { threw=true; }
    assert(threw);
    float nan[]={0,std::numeric_limits<float>::quiet_NaN(),1};threw=false;
    try { v.observe(true,0,nan,3); } catch(const std::runtime_error &) { threw=true; }
    assert(threw);
    float tied[]={1,1,0},flipped[]={1,1.000001f,0};
    LogitValidation top(3);top.observe(false,0,tied,3);top.observe(true,0,flipped,3);
    assert(!top.passed(1,1,1));
}
