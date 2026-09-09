#include "traffic_models.h"

#include <math.h>
#include <stdlib.h>

/* This is the standalone form of fe_compute_core() in the supplied
 * feature_extractor.c. Keep constants and arithmetic in sync with that file. */
#define RF_EPS 1e-6
#define RF_CAP 20.0
#define RF_MID_LO 300.0
#define RF_MID_HI 1200.0
#define RF_MTU_CAP 1514.0

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double percentile_sorted(const double *values, int count, double p) {
    double position = p * (count - 1);
    int low = (int)position;
    double fraction = position - low;
    if (low + 1 >= count) return values[low];
    return values[low] * (1.0 - fraction) + values[low + 1] * fraction;
}

static double ratio_feature(double numerator, double denominator) {
    double value = numerator / (fabs(denominator) + 1e-3);
    if (value > 100.0) return 100.0;
    if (value < -100.0) return -100.0;
    return value;
}

int traffic_rf_extract(const uint64_t *timestamp_us, const uint32_t *packet_size,
                       size_t packet_count, float *features) {
    if (!timestamp_us || !packet_size || !features ||
        packet_count != TRAFFIC_GRU_PACKETS) return TRAFFIC_INVALID_INPUT;

    enum { N = TRAFFIC_GRU_PACKETS, M = TRAFFIC_GRU_PACKETS - 1 };
    double sizes[N], intervals[M], big_intervals[M];
    int tiny = 0, middle = 0, large = 0, back_to_back = 0;
    int over_1ms = 0, over_5ms = 0, around_20ms = 0, big_count = 0;
    double size_sum = 0.0, idle_over_10ms = 0.0;

    for (int index = 0; index < N; ++index) {
        if (index && timestamp_us[index] < timestamp_us[index - 1])
            return TRAFFIC_INVALID_INPUT;
        double size = (double)packet_size[index];
        if (size > RF_MTU_CAP) size = RF_MTU_CAP;
        sizes[index] = size;
        size_sum += size;
        if (size < 100.0) ++tiny;
        if (size > RF_MID_LO && size <= RF_MID_HI) ++middle;
        if (size > RF_MID_HI) ++large;
    }
    for (int index = 0; index < M; ++index) {
        uint64_t delta = timestamp_us[index + 1] - timestamp_us[index];
        intervals[index] = (double)delta;
        if (delta < 50) ++back_to_back;
        if (delta > 1000) ++over_1ms;
        if (delta > 5000) ++over_5ms;
        if (delta >= 15000 && delta <= 25000) ++around_20ms;
        if (delta > 10000) idle_over_10ms += (double)delta;
    }

    qsort(sizes, N, sizeof(sizes[0]), compare_double);
    qsort(intervals, M, sizeof(intervals[0]), compare_double);
    double p05 = percentile_sorted(sizes, N, 0.05);
    double p10 = percentile_sorted(sizes, N, 0.10);
    double p25 = percentile_sorted(sizes, N, 0.25);
    double p75 = percentile_sorted(sizes, N, 0.75);
    double p95 = percentile_sorted(sizes, N, 0.95);
    double iat_median = percentile_sorted(intervals, M, 0.50);
    double iat_max = intervals[M - 1];
    double span = (double)(timestamp_us[N - 1] - timestamp_us[0]);
    if (span < 1.0) span = 1.0;

    double length_mean = size_sum / N;
    double fraction_tiny = (double)tiny / N;
    double fraction_middle = (double)middle / N;
    double fraction_large = (double)large / N;
    double fraction_back_to_back = (double)back_to_back / M;
    double fraction_1ms = (double)over_1ms / M;
    double fraction_5ms = (double)over_5ms / M;
    double fraction_20ms = (double)around_20ms / M;
    double duty = 1.0 - idle_over_10ms / (span + RF_EPS);
    double iat_max_median = iat_max / (iat_median + RF_EPS);
    if (iat_max_median > 1e3) iat_max_median = 1e3;

    double threshold = 2.0 * iat_median;
    if (threshold < 1000.0) threshold = 1000.0;
    for (int index = 0; index < M; ++index) {
        uint64_t delta = timestamp_us[index + 1] - timestamp_us[index];
        /* The original compares the previous packet against p75.  sizes is
         * sorted, so use packet_size again with the same 1514-byte cap. */
        double prior_size = (double)packet_size[index];
        if (prior_size > RF_MTU_CAP) prior_size = RF_MTU_CAP;
        if (prior_size > p75) big_intervals[big_count++] = (double)delta;
    }
    double median_after_big = 0.0;
    if (big_count) {
        qsort(big_intervals, (size_t)big_count, sizeof(big_intervals[0]), compare_double);
        median_after_big = percentile_sorted(big_intervals, big_count, 0.50);
    }
    double after_big_ratio = median_after_big / (iat_median + RF_EPS);
    if (after_big_ratio > RF_CAP) after_big_ratio = RF_CAP;

    features[0] = (float)fraction_middle;
    features[1] = (float)(p95 / (p05 + RF_EPS));
    features[2] = (float)length_mean;
    features[3] = (float)ratio_feature(iat_max_median, fraction_1ms);
    features[4] = (float)p10;
    features[5] = (float)((p75 / (p25 + RF_EPS)) * duty);
    features[6] = (float)(fraction_20ms * fraction_back_to_back);
    features[7] = (float)after_big_ratio;
    features[8] = (float)(iat_max * 1e-6);
    features[9] = (float)(4.0 * fraction_tiny * fraction_large);
    features[10] = (float)ratio_feature(fraction_5ms, fraction_20ms);
    for (int index = 0; index < TRAFFIC_RF_FEATURES; ++index)
        if (!isfinite(features[index])) return TRAFFIC_INVALID_INPUT;
    return TRAFFIC_OK;
}
