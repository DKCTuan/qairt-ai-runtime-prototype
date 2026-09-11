#ifndef TRAFFIC_MODEL_CONTRACT_H
#define TRAFFIC_MODEL_CONTRACT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TRAFFIC_MODEL_DTYPE_FLOAT32 = 1,
    TRAFFIC_MODEL_DTYPE_INT32 = 2,
} traffic_model_dtype_t;

typedef struct {
    const char *name;
    traffic_model_dtype_t dtype;
    size_t element_count;
} traffic_model_tensor_spec_t;

typedef struct {
    const char *model_id;
    const char *preprocessing_id;
    const traffic_model_tensor_spec_t *inputs;
    size_t input_count;
    const traffic_model_tensor_spec_t *outputs;
    size_t output_count;
    const char *const *class_names;
    size_t class_count;
    size_t skip_packets;
    float softmax_temperature;
    float accept_threshold;
} traffic_model_contract_t;

#ifdef __cplusplus
}
#endif

#endif
