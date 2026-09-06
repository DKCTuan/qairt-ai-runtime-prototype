#ifndef TRAFFIC_MODELS_H
#define TRAFFIC_MODELS_H
#include <stddef.h>
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
int traffic_gru_init(void);
void traffic_gru_deinit(void);
/* Normalized packet-major [90][3]: delta_time, direction, packet_length. */
int traffic_gru_predict(const float *input, size_t count, traffic_result *result);
/* Raw packet-major float array -> normalized array. Supports in-place use.
 * Call exactly once; do not normalize an already normalized tensor. */
int traffic_gru_prepare(const float *raw, size_t count, float *normalized);
const char *traffic_class_name(int label);
#ifdef __cplusplus
}
#endif
#endif
