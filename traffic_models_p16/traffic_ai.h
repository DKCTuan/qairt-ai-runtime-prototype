#ifndef TRAFFIC_AI_H
#define TRAFFIC_AI_H

/* Public header for the unified Random Forest + TinyGRU P16 SDK. */
#include <stddef.h>
#include <stdint.h>
#include "model_contract.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TRAFFIC_RF_FEATURES 11
#define TRAFFIC_CLASSES 5

typedef struct {
    float scores[TRAFFIC_CLASSES];
    int label;
    double latency_ms;
} traffic_result;

/* RF label order: Background, Game, RTVideo, VStream, Voice. */
const char *traffic_class_name(int label);
int traffic_rf_predict(const float *input, size_t count, traffic_result *result);
int traffic_rf_extract(const uint64_t *timestamp_us, const uint32_t *packet_size,
                       size_t packet_count, float *features);

/* The selected model profile is generated at build time. These accessors let
 * an application log/verify exactly which model contract was packaged. */
const traffic_model_contract_t *traffic_ai_model_contract(void);
const char *traffic_ai_model_id(void);
const char *traffic_ai_preprocessing_id(void);
size_t traffic_ai_skip_packets(void);
float traffic_ai_accept_threshold(void);

#include "traffic_p16.h"
#include "traffic_p16_deployment_config.h"

#ifdef __cplusplus
}
#endif

#endif
