#ifndef AI_MODEL_H
#define AI_MODEL_H

#include <stddef.h>
#include "ai_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Model-facing API exported by a generated static model library. */
int ai_model_init(void);
/* Return flattened float element counts for the single model input/output.
 * Call this after ai_model_init() so a client need not hard-code 240, 270,
 * or any other model-specific feature count. */
int ai_model_get_io_count(size_t *input_count, size_t *output_count);
int ai_model_predict(const float *input, size_t input_count,
                     float *scores, size_t score_capacity,
                     size_t *score_count, int *label, double *latency_ms);
void ai_model_deinit(void);

/* Multi-tensor ABI. Use this API for models such as TinyGRU+/16 that have
 * more than one input or whose inputs are not all float32. The caller owns
 * the tensor buffers; byte_size must exactly match the discovered metadata. */
int ai_model_get_tensor_count(int *input_tensor_count, int *output_tensor_count);
int ai_model_get_input_info(int index, dl_tensor_info_t *info);
int ai_model_get_output_info(int index, dl_tensor_info_t *info);
int ai_model_predict_tensors(const dl_tensor_t *inputs, int input_tensor_count,
                             dl_tensor_t *outputs, int output_tensor_count,
                             double *latency_ms);

#ifdef __cplusplus
}
#endif

#endif
