#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

class LogitValidation {
public:
    struct Stats {
        unsigned long long values=0, positions=0, violations=0, top_mismatches=0;
        double max_abs=0, max_tolerance_ratio=0;
    } candidate, control;
    static constexpr double atol=1e-4, rtol=1e-5;
    explicit LogitValidation(size_t vocabulary, size_t capacity=131)
        : vocabulary_(vocabulary), reference_(capacity), pending_(capacity), by_position_(capacity) {
        if (!vocabulary || !capacity) throw std::runtime_error("invalid logit validation dimensions");
    }
    void observe(bool is_candidate, size_t position, const float *values, size_t count) {
        if (!values || position>=reference_.size() || count!=vocabulary_)
            throw std::runtime_error("logit observation dimensions mismatch");
        for (size_t i=0;i<count;++i)
            if (!std::isfinite(values[i])) throw std::runtime_error("nonfinite validation logit");
        auto &ref=reference_[position]; auto &pending=pending_[position];
        if (is_candidate) {
            if (ref.empty()) {
                if (!pending.empty()) throw std::runtime_error("candidate reference still missing");
                pending.assign(values,values+count);
            } else compare(ref,values,candidate,position);
        } else if (ref.empty()) {
            ref.assign(values,values+count);
            if (!pending.empty()) { compare(ref,pending.data(),candidate,position); pending.clear(); }
        } else compare(ref,values,control,position);
    }
    bool passed(size_t positions, size_t candidate_trials, size_t cuda_trials) const {
        if (!candidate_trials || !cuda_trials || !positions || positions>reference_.size()) return false;
        for (size_t i=0;i<reference_.size();++i)
            if (!pending_[i].empty() || (i<positions)!=!reference_[i].empty()) return false;
        return candidate.positions==candidate_trials*positions
            && control.positions==(cuda_trials-1)*positions
            && candidate.values==candidate.positions*vocabulary_
            && control.values==control.positions*vocabulary_
            && !candidate.violations && !control.violations
            && !candidate.top_mismatches && !control.top_mismatches;
    }
    void report(FILE *out, bool valid, size_t positions) const {
        for (size_t i=0;i<by_position_.size();++i) {
            const auto &s=by_position_[i];
            if (s.positions) std::fprintf(out,"mojo_logit_position {\"position\":%zu,\"comparisons\":%llu,\"violations\":%llu,\"max_abs\":%.9g,\"max_tolerance_ratio\":%.9g}\n",
                i,s.positions,s.violations,s.max_abs,s.max_tolerance_ratio);
        }
        std::fprintf(out,"mojo_logits {\"passed\":%s,\"vocabulary\":%zu,\"positions\":%zu,\"atol\":%.9g,\"rtol\":%.9g,"
            "\"candidate_values\":%llu,\"candidate_positions\":%llu,\"candidate_violations\":%llu,\"candidate_top_mismatches\":%llu,\"candidate_max_abs\":%.9g,\"candidate_max_tolerance_ratio\":%.9g,"
            "\"control_values\":%llu,\"control_positions\":%llu,\"control_violations\":%llu,\"control_top_mismatches\":%llu,\"control_max_abs\":%.9g,\"control_max_tolerance_ratio\":%.9g}\n",
            valid?"true":"false",vocabulary_,positions,atol,rtol,
            candidate.values,candidate.positions,candidate.violations,candidate.top_mismatches,candidate.max_abs,candidate.max_tolerance_ratio,
            control.values,control.positions,control.violations,control.top_mismatches,control.max_abs,control.max_tolerance_ratio);
    }
private:
    size_t vocabulary_;
    std::vector<std::vector<float>> reference_,pending_;
    std::vector<Stats> by_position_;
    void compare(const std::vector<float> &ref, const float *values, Stats &stats, size_t position) {
        size_t ref_top=0, actual_top=0;
        for (size_t i=0;i<vocabulary_;++i) {
            double error=std::fabs(double(ref[i])-values[i]);
            double ratio=error/(atol+rtol*std::fabs(double(ref[i])));
            stats.max_abs=std::max(stats.max_abs,error);
            stats.max_tolerance_ratio=std::max(stats.max_tolerance_ratio,ratio);
            stats.violations+=ratio>1;
            if (&stats==&candidate) {
                auto &s=by_position_[position];
                s.max_abs=std::max(s.max_abs,error);
                s.max_tolerance_ratio=std::max(s.max_tolerance_ratio,ratio);
                s.violations+=ratio>1;
            }
            if (ref[i]>ref[ref_top]) ref_top=i;
            if (values[i]>values[actual_top]) actual_top=i;
        }
        stats.top_mismatches+=ref_top!=actual_top;
        stats.values+=vocabulary_; ++stats.positions;
        if (&stats==&candidate) ++by_position_[position].positions;
    }
};
