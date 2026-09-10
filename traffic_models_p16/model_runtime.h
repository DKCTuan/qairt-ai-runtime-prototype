#ifndef TRAFFIC_MODEL_RUNTIME_H
#define TRAFFIC_MODEL_RUNTIME_H

#include <stddef.h>
#include "model_contract.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TRAFFIC_MODEL_MAX_TENSORS 8

typedef struct {
    void *data;
    size_t bytes;
} traffic_model_buffer_t;

enum {
    TRAFFIC_MODEL_RUNTIME_OK = 0,
    TRAFFIC_MODEL_RUNTIME_INVALID = -1,
    TRAFFIC_MODEL_RUNTIME_UNAVAILABLE = -2,
    TRAFFIC_MODEL_RUNTIME_ABI_MISMATCH = -3,
    TRAFFIC_MODEL_RUNTIME_INFERENCE_ERROR = -4,
};

int traffic_model_runtime_init(const traffic_model_contract_t *contract);
void traffic_model_runtime_deinit(void);
int traffic_model_runtime_predict(const traffic_model_buffer_t *inputs,
                                  size_t input_count,
                                  traffic_model_buffer_t *outputs,
                                  size_t output_count,
                                  double *latency_ms);
const traffic_model_contract_t *traffic_model_runtime_contract(void);

#ifdef __cplusplus
}
#endif

#endif
