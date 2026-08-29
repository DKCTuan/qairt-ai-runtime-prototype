#include <stdio.h>
#include <stdlib.h>

#include "ai_runtime.h"

#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

int main(void)
{
    dl_runtime_t *runtime = NULL;
    CHECK(dl_runtime_create(NULL) == DL_ERROR_INVALID_ARGUMENT, "create validates output pointer");
    CHECK(dl_runtime_create(&runtime) == DL_OK && runtime != NULL, "create runtime");

    dl_runtime_t *second = NULL;
    CHECK(dl_runtime_create(&second) == DL_ERROR_BUSY, "reject second backend owner");

    int input_tensors = 0, output_tensors = 0;
    CHECK(dl_runtime_get_tensor_count(runtime, &input_tensors, &output_tensors) == DL_OK,
          "read tensor count");
    CHECK(input_tensors == 1 && output_tensors == 1, "mock tensor count");

    dl_tensor_info_t input_info, output_info;
    CHECK(dl_runtime_get_input_info(runtime, 0, &input_info) == DL_OK, "read input metadata");
    CHECK(dl_runtime_get_output_info(runtime, 0, &output_info) == DL_OK, "read output metadata");
    CHECK(input_info.dtype == DL_DTYPE_FLOAT32 && input_info.element_count == 240,
          "mock input metadata");
    CHECK(output_info.dtype == DL_DTYPE_FLOAT32 && output_info.element_count == 5,
          "mock output metadata");

    float *input = (float *)calloc(input_info.element_count, sizeof(float));
    float *output = (float *)calloc(output_info.element_count, sizeof(float));
    CHECK(input != NULL && output != NULL, "allocate test buffers");
    dl_tensor_t inputs[] = {{input, input_info.byte_size}};
    dl_tensor_t outputs[] = {{output, output_info.byte_size}};

    dl_tensor_t bad_input = {input, input_info.byte_size - 1};
    CHECK(dl_runtime_execute(runtime, &bad_input, 1, outputs, 1, NULL) == DL_ERROR_SIZE_MISMATCH,
          "reject wrong input byte size");
    CHECK(dl_runtime_execute(runtime, inputs, 1, outputs, 1, NULL) == DL_OK,
          "execute raw tensor API");
    CHECK(output[2] > output[0] && output[2] > output[1], "mock output values");

    dl_benchmark_config_t config = {2, 8};
    dl_benchmark_result_t benchmark;
    CHECK(dl_runtime_benchmark(runtime, inputs, 1, outputs, 1, &config, &benchmark) == DL_OK,
          "benchmark API");
    CHECK(benchmark.completed_runs == 8 && benchmark.min_ms <= benchmark.mean_ms &&
          benchmark.mean_ms <= benchmark.max_ms, "benchmark statistics");

    free(input);
    free(output);
    dl_runtime_destroy(runtime);
    printf("PASS: runtime API, metadata, lifecycle, size validation and benchmark\n");
    return 0;
}
