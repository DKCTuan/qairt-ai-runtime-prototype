#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>

#include "ai_runtime.h"

static int read_exact(const char *path, void *data, size_t byte_size)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) return -1;
    size_t read_size = fread(data, 1, byte_size, fp);
    int extra = fgetc(fp);
    fclose(fp);
    return read_size == byte_size && extra == EOF ? 0 : -1;
}

static int write_exact(const char *path, const void *data, size_t byte_size)
{
    FILE *fp = fopen(path, "wb");
    if (fp == NULL) return -1;
    size_t written = fwrite(data, 1, byte_size, fp);
    int close_status = fclose(fp);
    return written == byte_size && close_status == 0 ? 0 : -1;
}

static void cleanup(dl_runtime_t *runtime, dl_tensor_t *inputs, int input_count,
                    dl_tensor_t *outputs, int output_count)
{
    if (inputs != NULL) for (int i = 0; i < input_count; ++i) free(inputs[i].data);
    if (outputs != NULL) for (int i = 0; i < output_count; ++i) free(outputs[i].data);
    free(inputs);
    free(outputs);
    dl_runtime_destroy(runtime);
}

int main(void)
{
    const char *output_dir = getenv("DL_OUTPUT_DIR");
    if (output_dir == NULL || output_dir[0] == '\0') {
        fprintf(stderr, "run_model: DL_OUTPUT_DIR is required\n");
        return 1;
    }
    dl_runtime_t *runtime = NULL;
    dl_status_t status = dl_runtime_create(&runtime);
    if (status != DL_OK) {
        fprintf(stderr, "run_model: %s\n", dl_status_string(status));
        return 1;
    }
    int input_count = 0, output_count = 0;
    if (dl_runtime_get_tensor_count(runtime, &input_count, &output_count) != DL_OK) {
        cleanup(runtime, NULL, 0, NULL, 0); return 1;
    }
    dl_tensor_t *inputs = (dl_tensor_t *)calloc((size_t)input_count, sizeof(*inputs));
    dl_tensor_t *outputs = (dl_tensor_t *)calloc((size_t)output_count, sizeof(*outputs));
    if (inputs == NULL || outputs == NULL) {
        cleanup(runtime, inputs, input_count, outputs, output_count); return 1;
    }

    for (int i = 0; i < input_count; ++i) {
        dl_tensor_info_t info;
        char env_name[64];
        snprintf(env_name, sizeof(env_name), "DL_INPUT_%d_RAW", i);
        const char *path = getenv(env_name);
        if (path == NULL || dl_runtime_get_input_info(runtime, i, &info) != DL_OK) {
            fprintf(stderr, "run_model: %s is required\n", env_name);
            cleanup(runtime, inputs, input_count, outputs, output_count); return 1;
        }
        inputs[i].byte_size = info.byte_size;
        inputs[i].data = malloc(info.byte_size);
        if (inputs[i].data == NULL || read_exact(path, inputs[i].data, info.byte_size) != 0) {
            fprintf(stderr, "run_model: input %d must contain exactly %zu bytes: %s\n",
                    i, info.byte_size, path);
            cleanup(runtime, inputs, input_count, outputs, output_count); return 1;
        }
    }
    for (int i = 0; i < output_count; ++i) {
        dl_tensor_info_t info;
        if (dl_runtime_get_output_info(runtime, i, &info) != DL_OK) {
            cleanup(runtime, inputs, input_count, outputs, output_count); return 1;
        }
        outputs[i].byte_size = info.byte_size;
        outputs[i].data = malloc(info.byte_size);
        if (outputs[i].data == NULL) {
            cleanup(runtime, inputs, input_count, outputs, output_count); return 1;
        }
    }

    double latency_ms = 0.0;
    status = dl_runtime_execute(runtime, inputs, input_count, outputs, output_count, &latency_ms);
    if (status != DL_OK) {
        fprintf(stderr, "run_model: %s\n", dl_status_string(status));
        cleanup(runtime, inputs, input_count, outputs, output_count); return 1;
    }
    printf("{\"latency_ms\":%.6f,\"outputs\":[", latency_ms);
    for (int i = 0; i < output_count; ++i) {
        char path[4096];
        snprintf(path, sizeof(path), "%s/output_%d.raw", output_dir, i);
        if (write_exact(path, outputs[i].data, outputs[i].byte_size) != 0) {
            fprintf(stderr, "run_model: cannot write %s\n", path);
            cleanup(runtime, inputs, input_count, outputs, output_count); return 1;
        }
        printf("%s{\"index\":%d,\"path\":\"output_%d.raw\",\"byte_size\":%zu}",
               i == 0 ? "" : ",", i, i, outputs[i].byte_size);
    }
    printf("]}\n");
    cleanup(runtime, inputs, input_count, outputs, output_count);
    return 0;
}
