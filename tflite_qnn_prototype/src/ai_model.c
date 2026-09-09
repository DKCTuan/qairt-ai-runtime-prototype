#include "ai_model.h"

#include "ai_runtime.h"

static dl_runtime_t *g_runtime;

static int get_counts(int *inputs, int *outputs)
{
    return g_runtime == NULL ? -1 :
        (dl_runtime_get_tensor_count(g_runtime, inputs, outputs) == DL_OK ? 0 : -1);
}

int ai_model_init(void)
{
    if (g_runtime != NULL) return 0;
    return dl_runtime_create(&g_runtime) == DL_OK ? 0 : -1;
}

int ai_model_get_io_count(size_t *input_count, size_t *output_count)
{
    int inputs = 0, outputs = 0;
    dl_tensor_info_t input_info, output_info;

    if (input_count == NULL || output_count == NULL ||
        get_counts(&inputs, &outputs) != 0 || inputs != 1 || outputs != 1 ||
        dl_runtime_get_input_info(g_runtime, 0, &input_info) != DL_OK ||
        dl_runtime_get_output_info(g_runtime, 0, &output_info) != DL_OK ||
        input_info.dtype != DL_DTYPE_FLOAT32 || output_info.dtype != DL_DTYPE_FLOAT32) {
        return -1;
    }
    *input_count = input_info.element_count;
    *output_count = output_info.element_count;
    return 0;
}

int ai_model_predict(const float *input, size_t input_count,
                     float *scores, size_t score_capacity,
                     size_t *score_count, int *label, double *latency_ms)
{
    size_t expected_inputs = 0, expected_outputs = 0;
    dl_tensor_t input_tensor, output_tensor;
    double measured_ms = 0.0;

    if (input == NULL || scores == NULL || score_count == NULL || label == NULL) {
        return -1;
    }
    if (ai_model_get_io_count(&expected_inputs, &expected_outputs) != 0 ||
        expected_inputs != input_count || score_capacity < expected_outputs) {
        return -1;
    }
    input_tensor.data = (void *)input;
    input_tensor.byte_size = input_count * sizeof(*input);
    output_tensor.data = scores;
    output_tensor.byte_size = expected_outputs * sizeof(*scores);
    if (ai_model_predict_tensors(&input_tensor, 1, &output_tensor, 1, &measured_ms) != 0) {
        return -1;
    }
    int best = 0;
    for (size_t i = 1; i < expected_outputs; ++i) {
        if (scores[i] > scores[best]) best = (int)i;
    }
    *score_count = expected_outputs;
    *label = best;
    if (latency_ms != NULL) {
        *latency_ms = measured_ms;
    }
    return 0;
}

void ai_model_deinit(void)
{
    if (g_runtime != NULL) dl_runtime_destroy(g_runtime);
    g_runtime = NULL;
}

int ai_model_get_tensor_count(int *input_tensor_count, int *output_tensor_count)
{
    return get_counts(input_tensor_count, output_tensor_count);
}

int ai_model_get_input_info(int index, dl_tensor_info_t *info)
{
    return g_runtime != NULL && dl_runtime_get_input_info(g_runtime, index, info) == DL_OK ? 0 : -1;
}

int ai_model_get_output_info(int index, dl_tensor_info_t *info)
{
    return g_runtime != NULL && dl_runtime_get_output_info(g_runtime, index, info) == DL_OK ? 0 : -1;
}

int ai_model_predict_tensors(const dl_tensor_t *inputs, int input_tensor_count,
                             dl_tensor_t *outputs, int output_tensor_count,
                             double *latency_ms)
{
    return g_runtime != NULL &&
        dl_runtime_execute(g_runtime, inputs, input_tensor_count, outputs,
                           output_tensor_count, latency_ms) == DL_OK ? 0 : -1;
}
