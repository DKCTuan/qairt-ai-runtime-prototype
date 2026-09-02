#include "backend.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Mock backend gia lap shape cua traffic_qos_model (240 input, 5 output/
 * class) de test flow application ma khong can QAIRT SDK. Day la fixture
 * co dinh cho muc dich test nhanh, khong phai backend "generic" - neu doi
 * model that thi dung backend qnn_cli/qnn_api, khong sua so o day. */
#define MOCK_INPUT_COUNT 240
#define MOCK_OUTPUT_COUNT 5
#define MOCK_SECOND_INPUT_COUNT 3
#define MOCK_SECOND_OUTPUT_COUNT 2

static int g_backend_ready = 0;
static int g_multi_io = 0;

int backend_init(void)
{
    const char *multi = getenv("DL_MOCK_MULTI_IO");
    g_multi_io = multi != NULL && multi[0] != '\0' && strcmp(multi, "0") != 0;
    g_backend_ready = 1;
    return 0;
}

int backend_get_io_count(int *input_count, int *output_count)
{
    if (!g_backend_ready || g_multi_io) {
        return -1;
    }
    if (input_count != NULL) {
        *input_count = MOCK_INPUT_COUNT;
    }
    if (output_count != NULL) {
        *output_count = MOCK_OUTPUT_COUNT;
    }
    return 0;
}

int backend_get_tensor_count(int *input_tensor_count, int *output_tensor_count)
{
    if (!g_backend_ready || input_tensor_count == NULL || output_tensor_count == NULL) return -1;
    *input_tensor_count = g_multi_io ? 2 : 1;
    *output_tensor_count = g_multi_io ? 2 : 1;
    return 0;
}

int backend_get_tensor_info(int is_input, int index, dl_tensor_info_t *info)
{
    int tensor_count = g_multi_io ? 2 : 1;
    if (!g_backend_ready || index < 0 || index >= tensor_count || info == NULL) return -1;
    int count = is_input
        ? (index == 0 ? MOCK_INPUT_COUNT : MOCK_SECOND_INPUT_COUNT)
        : (index == 0 ? MOCK_OUTPUT_COUNT : MOCK_SECOND_OUTPUT_COUNT);
    memset(info, 0, sizeof(*info));
    snprintf(info->name, sizeof(info->name), "%s_%d", is_input ? "mock_input" : "mock_output", index);
    info->dtype = DL_DTYPE_FLOAT32;
    info->rank = 1;
    info->dimensions[0] = (uint32_t)count;
    info->element_count = (size_t)count;
    info->byte_size = (size_t)count * sizeof(float);
    info->scale = 1.0f;
    info->quantized_axis = -1;
    return 0;
}

int backend_execute(const float *input, int input_count, float *output, int output_count)
{
    if (!g_backend_ready || input == NULL || output == NULL || input_count <= 0 || output_count < 5) {
        return -1;
    }

    /* Mock output copied from the validated QNN CPU result. */
    output[0] = 4.4335314e-04f;
    output[1] = 2.2159459e-03f;
    output[2] = 9.9732780e-01f;
    output[3] = 1.2404647e-05f;
    output[4] = 4.2676922e-07f;

    return 0;
}

void backend_deinit(void)
{
    g_backend_ready = 0;
    g_multi_io = 0;
}

/* THEM MOI (dong bo voi backend.h/tflite_qnn_prototype): backend nay chua
 * ho tro doc dtype dong (chi lam viec voi float32 tu truoc gio), nen tra
 * ve -1 - ai_runtime.c se tu hieu la "khong ho tro", giu nguyen mac dinh
 * DL_DTYPE_FLOAT32 va duong code cu (backend_execute()). KHONG doi hanh
 * vi hien tai cua backend nay. */
int backend_get_io_dtype(dl_tensor_dtype_t *input_dtype, float *input_scale, int *input_zero_point,
                          dl_tensor_dtype_t *output_dtype, float *output_scale, int *output_zero_point)
{
    (void)input_dtype; (void)input_scale; (void)input_zero_point;
    (void)output_dtype; (void)output_scale; (void)output_zero_point;
    return -1;
}

/* Backend nay khong dung duong "raw" (chi co model float32) - khong bao
 * gio duoc goi thuc te vi backend_get_io_dtype() da tra -1 o tren, nhung
 * van dinh nghia de link OK. */
int backend_execute_raw(const void *input, int input_count, void *output, int output_count)
{
    return backend_execute((const float *)input, input_count,
                           (float *)output, output_count);
}

int backend_execute_tensors(const dl_tensor_t *inputs, int input_tensor_count,
                            dl_tensor_t *outputs, int output_tensor_count)
{
    int expected_tensors = g_multi_io ? 2 : 1;
    if (inputs == NULL || outputs == NULL || input_tensor_count != expected_tensors ||
        output_tensor_count != expected_tensors ||
        inputs[0].byte_size != (size_t)MOCK_INPUT_COUNT * sizeof(float) ||
        outputs[0].byte_size != (size_t)MOCK_OUTPUT_COUNT * sizeof(float)) return -1;
    if (backend_execute((const float *)inputs[0].data, MOCK_INPUT_COUNT,
                        (float *)outputs[0].data, MOCK_OUTPUT_COUNT) != 0) return -1;
    if (g_multi_io) {
        if (inputs[1].data == NULL || outputs[1].data == NULL ||
            inputs[1].byte_size != (size_t)MOCK_SECOND_INPUT_COUNT * sizeof(float) ||
            outputs[1].byte_size != (size_t)MOCK_SECOND_OUTPUT_COUNT * sizeof(float)) return -1;
        const float *second_input = (const float *)inputs[1].data;
        float *second_output = (float *)outputs[1].data;
        second_output[0] = second_input[0] + second_input[1] + second_input[2];
        second_output[1] = second_input[2] - second_input[0];
    }
    return 0;
}
