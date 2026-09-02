#include <stdio.h>
#include <stdlib.h>

#include "ai_runtime.h"

#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

int main(void)
{
    dl_runtime_t *runtime = NULL;
    CHECK(dl_runtime_create(&runtime) == DL_OK, "create multi-I/O runtime");

    int input_count = 0, output_count = 0;
    CHECK(dl_runtime_get_tensor_count(runtime, &input_count, &output_count) == DL_OK,
          "get tensor counts");
    CHECK(input_count == 2 && output_count == 2, "discover two inputs and two outputs");

    dl_tensor_info_t in_info[2], out_info[2];
    for (int i = 0; i < 2; ++i) {
        CHECK(dl_runtime_get_input_info(runtime, i, &in_info[i]) == DL_OK, "get input metadata");
        CHECK(dl_runtime_get_output_info(runtime, i, &out_info[i]) == DL_OK, "get output metadata");
    }
    CHECK(in_info[1].element_count == 3 && out_info[1].element_count == 2,
          "read non-primary tensor metadata");

    float *in0 = (float *)calloc(in_info[0].element_count, sizeof(float));
    float in1[3] = {2.0f, 3.0f, 7.0f};
    float *out0 = (float *)calloc(out_info[0].element_count, sizeof(float));
    float out1[2] = {0.0f, 0.0f};
    CHECK(in0 != NULL && out0 != NULL, "allocate buffers");
    dl_tensor_t inputs[2] = {{in0, in_info[0].byte_size}, {in1, sizeof(in1)}};
    dl_tensor_t outputs[2] = {{out0, out_info[0].byte_size}, {out1, sizeof(out1)}};

    CHECK(dl_runtime_execute(runtime, inputs, 1, outputs, 2, NULL) == DL_ERROR_INVALID_ARGUMENT,
          "reject wrong tensor count");
    CHECK(dl_runtime_execute(runtime, inputs, 2, outputs, 2, NULL) == DL_OK,
          "execute multiple tensors");
    CHECK(out0[2] > out0[0] && out1[0] == 12.0f && out1[1] == 5.0f,
          "verify all outputs");

    free(in0);
    free(out0);
    dl_runtime_destroy(runtime);
    printf("PASS: multi-input/output metadata and execution\n");
    return 0;
}
