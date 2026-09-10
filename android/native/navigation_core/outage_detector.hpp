#ifndef NAV_OUTAGE_DETECTOR_HPP
#define NAV_OUTAGE_DETECTOR_HPP

#include "nav_state.hpp"

namespace nav_core {

struct OutageDetectorConfig {
    double max_timestamp_gap_sec;
    double degraded_horizontal_acc_m;
    double outage_horizontal_acc_m;
    double degraded_hdop;
    double outage_hdop;
    double degraded_cn0;
    double outage_cn0;
    int persistence_count;

    OutageDetectorConfig()
        : max_timestamp_gap_sec(2.0),
          degraded_horizontal_acc_m(5.0),
          outage_horizontal_acc_m(15.0),
          degraded_hdop(2.5),
          outage_hdop(5.0),
          degraded_cn0(35.0),
          outage_cn0(25.0),
          persistence_count(2) {}
};

class OutageDetector {
public:
    OutageDetector(const OutageDetectorConfig& cfg = OutageDetectorConfig());

    NavigationStatus processObservation(
        bool has_gnss_fix,
        double current_time,
        double h_acc_m = -1.0,
        double hdop = -1.0,
        double cn0 = -1.0
    );

    NavigationStatus getStatus() const { return current_status_; }
    void reset();

private:
    OutageDetectorConfig config_;
    NavigationStatus current_status_;
    double last_obs_time_;
    NavigationStatus pending_candidate_;
    int candidate_counter_;
    int recovering_counter_;

    NavigationStatus evaluateInstantCandidate(
        bool has_fix,
        double current_time,
        double h_acc_m,
        double hdop,
        double cn0
    );
};

} // namespace nav_core

#endif // NAV_OUTAGE_DETECTOR_HPP
