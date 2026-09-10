#include "ai_model.h"

#include <string.h>

static int initialized;

int ai_model_init(void)
{
    initialized = 1;
    return 0;
}

void ai_model_deinit(void)
{
    initialized = 0;
}

int ai_model_get_tensor_count(int *input_count, int *output_count)
{
    if (!initialized || input_count == NULL || output_count == NULL) return -1;
    *input_count = 2;
    *output_count = 1;
    return 0;
}

int ai_model_get_input_info(int index, dl_tensor_info_t *info)
{
    if (!initialized || info == NULL || (index != 0 && index != 1)) return -1;
    memset(info, 0, sizeof(*info));
    info->dtype = index == 0 ? DL_DTYPE_FLOAT32 : DL_DTYPE_INT32;
    info->element_count = index == 0 ? 270U : 90U;
    info->byte_size = info->element_count * 4U;
    return 0;
}

int ai_model_get_output_info(int index, dl_tensor_info_t *info)
{
    if (!initialized || info == NULL || index != 0) return -1;
    memset(info, 0, sizeof(*info));
    info->dtype = DL_DTYPE_FLOAT32;
    info->element_count = 5U;
    info->byte_size = 20U;
    return 0;
}

int ai_model_predict_tensors(const dl_tensor_t *inputs, int input_count,
                             dl_tensor_t *outputs, int output_count,
                             double *latency_ms)
{
    static const float scores[5] = {0.05f, 0.1f, 0.7f, 0.1f, 0.05f};
    if (!initialized || inputs == NULL || outputs == NULL ||
        input_count != 2 || output_count != 1 ||
        inputs[0].byte_size != 1080U || inputs[1].byte_size != 360U ||
        outputs[0].byte_size != sizeof(scores))
        return -1;
    memcpy(outputs[0].data, scores, sizeof(scores));
    if (latency_ms != NULL) *latency_ms = 0.25;
    return 0;
}

int ai_model_get_io_count(size_t *input_count, size_t *output_count)
{
    (void)input_count;
    (void)output_count;
    return -1;
}

int ai_model_predict(const float *input, size_t input_count,
                     float *scores, size_t score_capacity,
                     size_t *score_count, int *label, double *latency_ms)
{
    (void)input;
    (void)input_count;
    (void)scores;
    (void)score_capacity;
    (void)score_count;
    (void)label;
    (void)latency_ms;
    return -1;
}
