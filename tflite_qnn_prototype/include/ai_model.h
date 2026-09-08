#ifndef AI_MODEL_H
#define AI_MODEL_H

#include <stddef.h>

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

#ifdef __cplusplus
}
#endif

#endif
