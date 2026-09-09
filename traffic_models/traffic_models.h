#ifndef TRAFFIC_MODELS_H
#define TRAFFIC_MODELS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define TRAFFIC_RF_FEATURES 11
#define TRAFFIC_GRU_PACKETS 90
#define TRAFFIC_GRU_CHANNELS 3
#define TRAFFIC_GRU_FEATURES 270
#define TRAFFIC_CLASSES 5
#define TRAFFIC_OK 0
#define TRAFFIC_INVALID_INPUT -1
#define TRAFFIC_MODEL_UNAVAILABLE -2
#define TRAFFIC_INFERENCE_ERROR -3
typedef struct {
    float scores[TRAFFIC_CLASSES];
    int label;
    double latency_ms;
} traffic_result;
/* RF is always available. GRU is optional and must be initialized separately.
 * Calls are sequential, not thread-safe. */
int traffic_rf_predict(const float *input, size_t count, traffic_result *result);
/* Reproduces the supplied feature_extractor.c / featgen7.py RF contract.
 * Input is exactly one training-aligned window of 90 packets: timestamps are
 * monotonically non-decreasing microseconds; packet_size is IPv4 total length.
 * The capture/flow layer must remove the same first 10 flow packets as train
 * before supplying this window. Direction is deliberately not used by RF.
 * On success, features contains the 11 values accepted by traffic_rf_predict.
 */
int traffic_rf_extract(const uint64_t *timestamp_us, const uint32_t *packet_size,
                       size_t packet_count, float *features);
int traffic_gru_init(void);
void traffic_gru_deinit(void);
/* Normalized packet-major [90][3]: delta_time, direction, packet_length. */
int traffic_gru_predict(const float *input, size_t count, traffic_result *result);
/* Raw packet-major float array -> normalized array. Supports in-place use.
 * Call exactly once; do not normalize an already normalized tensor. */
int traffic_gru_prepare(const float *raw, size_t count, float *normalized);
/* Production packet-window adapter for TinyGRU. The SDK creates
 * [delta_time_seconds, direction, ip_total_length] and normalizes it exactly
 * once. direction must be +1 for outbound/local-source, -1 for inbound.
 * Supply the same training-aligned 90-packet window used for RF extraction. */
int traffic_gru_prepare_packets(const uint64_t *timestamp_us, const int8_t *direction,
                                const uint32_t *ip_total_length, size_t packet_count,
                                float *normalized);
const char *traffic_class_name(int label);
#ifdef __cplusplus
}
#endif
#endif
