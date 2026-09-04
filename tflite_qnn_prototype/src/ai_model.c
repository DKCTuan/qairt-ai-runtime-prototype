#include "ai_model.h"

#include "ai_runtime.h"

int ai_model_init(void)
{
    return dl_init();
}

int ai_model_predict(const float *input, size_t input_count,
                     float *scores, size_t score_capacity,
                     size_t *score_count, int *label, double *latency_ms)
{
    int expected_inputs = 0;
    int expected_outputs = 0;
    dl_result_t result;

    if (input == NULL || scores == NULL || score_count == NULL || label == NULL) {
        return -1;
    }
    if (dl_get_io_count(&expected_inputs, &expected_outputs) != 0 ||
        expected_inputs != (int)input_count ||
        expected_outputs < 0 || score_capacity < (size_t)expected_outputs) {
        return -1;
    }
    if (dl_inference_ex(input, &result) != 0) {
        return -1;
    }
    for (int i = 0; i < result.score_count; ++i) {
        scores[i] = result.scores[i];
    }
    *score_count = (size_t)result.score_count;
    *label = result.label;
    if (latency_ms != NULL) {
        *latency_ms = result.latency_ms;
    }
    return 0;
}

void ai_model_deinit(void)
{
    dl_deinit();
}
