#include "outage_detector.hpp"

namespace nav_core {

OutageDetector::OutageDetector(const OutageDetectorConfig& cfg)
    : config_(cfg),
      current_status_(NAV_STATUS_GOOD),
      last_obs_time_(-1.0),
      pending_candidate_(NAV_STATUS_GOOD),
      candidate_counter_(0),
      recovering_counter_(0) {}

NavigationStatus OutageDetector::evaluateInstantCandidate(
    bool has_fix,
    double current_time,
    double h_acc_m,
    double hdop,
    double cn0
) {
    if (!has_fix) {
        return NAV_STATUS_OUTAGE;
    }

    if (last_obs_time_ >= 0.0) {
        double gap = current_time - last_obs_time_;
        if (gap > config_.max_timestamp_gap_sec) {
            return NAV_STATUS_OUTAGE;
        }
    }

    if (h_acc_m > 0.0) {
        if (h_acc_m >= config_.outage_horizontal_acc_m) return NAV_STATUS_OUTAGE;
        if (h_acc_m >= config_.degraded_horizontal_acc_m) return NAV_STATUS_DEGRADED;
    }

    if (hdop > 0.0) {
        if (hdop >= config_.outage_hdop) return NAV_STATUS_OUTAGE;
        if (hdop >= config_.degraded_hdop) return NAV_STATUS_DEGRADED;
    }

    if (cn0 > 0.0) {
        if (cn0 <= config_.outage_cn0) return NAV_STATUS_OUTAGE;
        if (cn0 <= config_.degraded_cn0) return NAV_STATUS_DEGRADED;
    }

    return NAV_STATUS_GOOD;
}

NavigationStatus OutageDetector::processObservation(
    bool has_gnss_fix,
    double current_time,
    double h_acc_m,
    double hdop,
    double cn0
) {
    NavigationStatus candidate = evaluateInstantCandidate(has_gnss_fix, current_time, h_acc_m, hdop, cn0);

    if (has_gnss_fix && current_time >= 0.0) {
        last_obs_time_ = current_time;
    }

    if (candidate == current_status_) {
        candidate_counter_ = 0;
        pending_candidate_ = current_status_;
        if (current_status_ == NAV_STATUS_RECOVERING) {
            recovering_counter_++;
            if (recovering_counter_ >= config_.persistence_count) {
                current_status_ = NAV_STATUS_GOOD;
                recovering_counter_ = 0;
            }
        }
        return current_status_;
    }

    if (candidate == pending_candidate_) {
        candidate_counter_++;
    } else {
        pending_candidate_ = candidate;
        candidate_counter_ = 1;
    }

    if (candidate_counter_ >= config_.persistence_count) {
        NavigationStatus old_status = current_status_;
        NavigationStatus target_status = pending_candidate_;

        if (old_status == NAV_STATUS_OUTAGE && target_status == NAV_STATUS_GOOD) {
            current_status_ = NAV_STATUS_RECOVERING;
            recovering_counter_ = 1;
        } else {
            current_status_ = target_status;
            recovering_counter_ = 0;
        }

        candidate_counter_ = 0;
        pending_candidate_ = current_status_;
    }

    return current_status_;
}

void OutageDetector::reset() {
    current_status_ = NAV_STATUS_GOOD;
    last_obs_time_ = -1.0;
    pending_candidate_ = NAV_STATUS_GOOD;
    candidate_counter_ = 0;
    recovering_counter_ = 0;
}

} // namespace nav_core
