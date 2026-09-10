#include "model_runtime.h"

#include <string.h>

#ifdef TRAFFIC_AI_WITH_MODEL
#include "ai_model.h"
#endif

static const traffic_model_contract_t *active_contract;

static size_t dtype_size(traffic_model_dtype_t dtype)
{
    switch (dtype) {
    case TRAFFIC_MODEL_DTYPE_FLOAT32:
    case TRAFFIC_MODEL_DTYPE_INT32:
        return 4;
    default:
        return 0;
    }
}

#ifdef TRAFFIC_AI_WITH_MODEL
static int dtype_matches(traffic_model_dtype_t expected, dl_tensor_dtype_t actual)
{
    return (expected == TRAFFIC_MODEL_DTYPE_FLOAT32 && actual == DL_DTYPE_FLOAT32) ||
           (expected == TRAFFIC_MODEL_DTYPE_INT32 && actual == DL_DTYPE_INT32);
}

static int info_matches(const traffic_model_tensor_spec_t *expected,
                        const dl_tensor_info_t *actual)
{
    return dtype_matches(expected->dtype, actual->dtype) &&
           expected->element_count == actual->element_count;
}
#endif

static int valid_contract(const traffic_model_contract_t *contract)
{
    if (contract == NULL || contract->model_id == NULL ||
        contract->preprocessing_id == NULL || contract->inputs == NULL ||
        contract->outputs == NULL || contract->input_count == 0 ||
        contract->output_count == 0 ||
        contract->input_count > TRAFFIC_MODEL_MAX_TENSORS ||
        contract->output_count > TRAFFIC_MODEL_MAX_TENSORS)
        return 0;
    for (size_t i = 0; i < contract->input_count; ++i)
        if (contract->inputs[i].name == NULL ||
            dtype_size(contract->inputs[i].dtype) == 0 ||
            contract->inputs[i].element_count == 0)
            return 0;
    for (size_t i = 0; i < contract->output_count; ++i)
        if (contract->outputs[i].name == NULL ||
            dtype_size(contract->outputs[i].dtype) == 0 ||
            contract->outputs[i].element_count == 0)
            return 0;
    return 1;
}

int traffic_model_runtime_init(const traffic_model_contract_t *contract)
{
    if (!valid_contract(contract)) return TRAFFIC_MODEL_RUNTIME_INVALID;
#ifdef TRAFFIC_AI_WITH_MODEL
    int input_count = 0, output_count = 0;
    if (active_contract != NULL)
        return active_contract == contract ? TRAFFIC_MODEL_RUNTIME_OK :
                                             TRAFFIC_MODEL_RUNTIME_INVALID;
    if (ai_model_init() != 0 ||
        ai_model_get_tensor_count(&input_count, &output_count) != 0 ||
        input_count != (int)contract->input_count ||
        output_count != (int)contract->output_count) {
        ai_model_deinit();
        return TRAFFIC_MODEL_RUNTIME_ABI_MISMATCH;
    }
    for (size_t i = 0; i < contract->input_count; ++i) {
        dl_tensor_info_t info;
        if (ai_model_get_input_info((int)i, &info) != 0 ||
            !info_matches(&contract->inputs[i], &info)) {
            ai_model_deinit();
            return TRAFFIC_MODEL_RUNTIME_ABI_MISMATCH;
        }
    }
    for (size_t i = 0; i < contract->output_count; ++i) {
        dl_tensor_info_t info;
        if (ai_model_get_output_info((int)i, &info) != 0 ||
            !info_matches(&contract->outputs[i], &info)) {
            ai_model_deinit();
            return TRAFFIC_MODEL_RUNTIME_ABI_MISMATCH;
        }
    }
    active_contract = contract;
    return TRAFFIC_MODEL_RUNTIME_OK;
#else
    (void)contract;
    return TRAFFIC_MODEL_RUNTIME_UNAVAILABLE;
#endif
}

void traffic_model_runtime_deinit(void)
{
#ifdef TRAFFIC_AI_WITH_MODEL
    if (active_contract != NULL) ai_model_deinit();
#endif
    active_contract = NULL;
}

int traffic_model_runtime_predict(const traffic_model_buffer_t *inputs,
                                  size_t input_count,
                                  traffic_model_buffer_t *outputs,
                                  size_t output_count,
                                  double *latency_ms)
{
    if (active_contract == NULL) return TRAFFIC_MODEL_RUNTIME_UNAVAILABLE;
    if (inputs == NULL || outputs == NULL || input_count != active_contract->input_count ||
        output_count != active_contract->output_count)
        return TRAFFIC_MODEL_RUNTIME_INVALID;
    for (size_t i = 0; i < input_count; ++i) {
        size_t expected = active_contract->inputs[i].element_count *
                          dtype_size(active_contract->inputs[i].dtype);
        if (inputs[i].data == NULL || inputs[i].bytes != expected)
            return TRAFFIC_MODEL_RUNTIME_INVALID;
    }
    for (size_t i = 0; i < output_count; ++i) {
        size_t expected = active_contract->outputs[i].element_count *
                          dtype_size(active_contract->outputs[i].dtype);
        if (outputs[i].data == NULL || outputs[i].bytes != expected)
            return TRAFFIC_MODEL_RUNTIME_INVALID;
    }
#ifdef TRAFFIC_AI_WITH_MODEL
    dl_tensor_t model_inputs[TRAFFIC_MODEL_MAX_TENSORS];
    dl_tensor_t model_outputs[TRAFFIC_MODEL_MAX_TENSORS];
    for (size_t i = 0; i < input_count; ++i) {
        model_inputs[i].data = inputs[i].data;
        model_inputs[i].byte_size = inputs[i].bytes;
    }
    for (size_t i = 0; i < output_count; ++i) {
        model_outputs[i].data = outputs[i].data;
        model_outputs[i].byte_size = outputs[i].bytes;
    }
    return ai_model_predict_tensors(model_inputs, (int)input_count,
                                    model_outputs, (int)output_count,
                                    latency_ms) == 0 ?
           TRAFFIC_MODEL_RUNTIME_OK : TRAFFIC_MODEL_RUNTIME_INFERENCE_ERROR;
#else
    (void)latency_ms;
    return TRAFFIC_MODEL_RUNTIME_UNAVAILABLE;
#endif
}

const traffic_model_contract_t *traffic_model_runtime_contract(void)
{
    return active_contract;
}
