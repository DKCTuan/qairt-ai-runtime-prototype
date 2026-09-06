#define _POSIX_C_SOURCE 200809L
#include "traffic_models.h"
#include <math.h>
#include <string.h>
#include <time.h>
#ifdef TRAFFIC_WITH_GRU
#include "ai_model.h"
static int gru_ready;
#endif
/* rf_model.h defines the forest data; include it only in vendor/rf_model.c. */
extern int predict_forest(float input[], float output_probs[]);
static int valid(const float *input, size_t count, size_t expected) {
    if (!input || count != expected) return 0;
    for (size_t i = 0; i < count; ++i) if (!isfinite(input[i])) return 0;
    return 1;
}
const char *traffic_class_name(int label) {
    static const char *names[] = {"Background", "Game", "RTVideo", "VStream", "Voice"};
    return label >= 0 && label < TRAFFIC_CLASSES ? names[label] : "Unknown";
}
int traffic_rf_predict(const float *input, size_t count, traffic_result *result) {
    if (!result) return TRAFFIC_INVALID_INPUT;
    memset(result, 0, sizeof(*result)); result->label = -1;
    if (!valid(input, count, TRAFFIC_RF_FEATURES)) return TRAFFIC_INVALID_INPUT;
    float features[TRAFFIC_RF_FEATURES];
    memcpy(features, input, sizeof(features));
    struct timespec start, end;
    if (clock_gettime(CLOCK_MONOTONIC, &start)) return TRAFFIC_INFERENCE_ERROR;
    result->label = predict_forest(features, result->scores);
    if (clock_gettime(CLOCK_MONOTONIC, &end)) return TRAFFIC_INFERENCE_ERROR;
    result->latency_ms = (end.tv_sec-start.tv_sec)*1000.0 + (end.tv_nsec-start.tv_nsec)/1e6;
    return TRAFFIC_OK;
}
int traffic_gru_prepare(const float *raw, size_t count, float *out) {
    if (!out || !valid(raw, count, TRAFFIC_GRU_FEATURES)) return TRAFFIC_INVALID_INPUT;
    for (size_t i = 0; i < count; i += 3) {
        out[i] = (raw[i] - 0.005491858348250389f) / 0.07258545607328415f;
        out[i+1] = raw[i+1];
        out[i+2] = (raw[i+2] - 626.3450317382812f) / 594.463134765625f;
    }
    return valid(out, count, TRAFFIC_GRU_FEATURES) ? TRAFFIC_OK : TRAFFIC_INVALID_INPUT;
}
int traffic_gru_init(void) {
#ifdef TRAFFIC_WITH_GRU
    if (gru_ready) return TRAFFIC_OK;
    if (ai_model_init()) return TRAFFIC_INFERENCE_ERROR;
    gru_ready = 1;
    return TRAFFIC_OK;
#else
    return TRAFFIC_MODEL_UNAVAILABLE;
#endif
}
void traffic_gru_deinit(void) {
#ifdef TRAFFIC_WITH_GRU
    if (gru_ready) ai_model_deinit();
    gru_ready = 0;
#endif
}
int traffic_gru_predict(const float *input, size_t count, traffic_result *result) {
    if (!result) return TRAFFIC_INVALID_INPUT;
    memset(result, 0, sizeof(*result)); result->label = -1;
    if (!valid(input, count, TRAFFIC_GRU_FEATURES)) return TRAFFIC_INVALID_INPUT;
#ifdef TRAFFIC_WITH_GRU
    if (!gru_ready) return TRAFFIC_MODEL_UNAVAILABLE;
    size_t outputs = 0;
    if (ai_model_predict(input, count, result->scores, TRAFFIC_CLASSES,
                         &outputs, &result->label, &result->latency_ms) ||
        outputs != TRAFFIC_CLASSES) return TRAFFIC_INFERENCE_ERROR;
    return TRAFFIC_OK;
#else
    return TRAFFIC_MODEL_UNAVAILABLE;
#endif
}
