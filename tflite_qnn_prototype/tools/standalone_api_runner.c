/*
 * Generated standalone application entry point.
 * It knows only ai_runtime.h; TensorFlow Lite is confined to the backend.
 */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ai_runtime.h"

static int read_exact(const char *path, void *data, size_t bytes) {
  FILE *fp = fopen(path, "rb");
  size_t got;
  int extra;
  if (fp == NULL) {
    fprintf(stderr, "standalone_runner: cannot open %s: %s\n", path, strerror(errno));
    return -1;
  }
  got = fread(data, 1, bytes, fp);
  extra = fgetc(fp);
  fclose(fp);
  return got == bytes && extra == EOF ? 0 : -1;
}

static int write_exact(const char *path, const void *data, size_t bytes) {
  FILE *fp = fopen(path, "wb");
  if (fp == NULL || fwrite(data, 1, bytes, fp) != bytes || fclose(fp) != 0) {
    fprintf(stderr, "standalone_runner: cannot write %s\n", path);
    return -1;
  }
  return 0;
}

static void destroy(dl_runtime_t *runtime, dl_tensor_t *inputs, int input_count,
                    dl_tensor_t *outputs, int output_count) {
  int i;
  for (i = 0; i < input_count; ++i) free(inputs[i].data);
  for (i = 0; i < output_count; ++i) free(outputs[i].data);
  free(inputs);
  free(outputs);
  dl_runtime_destroy(runtime);
}

int main(int argc, char **argv) {
  const char *output_dir;
  dl_runtime_t *runtime = NULL;
  dl_tensor_t *inputs = NULL;
  dl_tensor_t *outputs = NULL;
  int input_count = 0, output_count = 0, i;
  double latency_ms = 0.0;

  if (argc < 4 || strcmp(argv[1], "--output-dir") != 0) {
    fprintf(stderr, "Usage: %s --output-dir DIR INPUT_0.raw [INPUT_1.raw ...]\n", argv[0]);
    return 2;
  }
  output_dir = argv[2];
  if (dl_runtime_create(&runtime) != DL_OK ||
      dl_runtime_get_tensor_count(runtime, &input_count, &output_count) != DL_OK ||
      argc - 3 != input_count) {
    fprintf(stderr, "standalone_runner: model expects %d input file(s)\n", input_count);
    destroy(runtime, inputs, 0, outputs, 0);
    return 2;
  }
  inputs = calloc((size_t)input_count, sizeof(*inputs));
  outputs = calloc((size_t)output_count, sizeof(*outputs));
  if (inputs == NULL || outputs == NULL) {
    destroy(runtime, inputs, input_count, outputs, output_count);
    return 1;
  }
  for (i = 0; i < input_count; ++i) {
    dl_tensor_info_t info;
    if (dl_runtime_get_input_info(runtime, i, &info) != DL_OK ||
        (inputs[i].data = malloc(info.byte_size)) == NULL ||
        read_exact(argv[i + 3], inputs[i].data, info.byte_size) != 0) {
      fprintf(stderr, "standalone_runner: input %d has incorrect size\n", i);
      destroy(runtime, inputs, input_count, outputs, output_count);
      return 1;
    }
    inputs[i].byte_size = info.byte_size;
  }
  for (i = 0; i < output_count; ++i) {
    dl_tensor_info_t info;
    if (dl_runtime_get_output_info(runtime, i, &info) != DL_OK ||
        (outputs[i].data = malloc(info.byte_size)) == NULL) {
      destroy(runtime, inputs, input_count, outputs, output_count);
      return 1;
    }
    outputs[i].byte_size = info.byte_size;
  }
  if (dl_runtime_execute(runtime, inputs, input_count, outputs, output_count, &latency_ms) != DL_OK) {
    fprintf(stderr, "standalone_runner: dl_runtime_execute failed\n");
    destroy(runtime, inputs, input_count, outputs, output_count);
    return 1;
  }
  printf("{\"mode\":\"ai_runtime_static_tflite_cpu\",\"latency_ms\":%.6f,\"outputs\":[", latency_ms);
  for (i = 0; i < output_count; ++i) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/output_%d.raw", output_dir, i);
    if (write_exact(path, outputs[i].data, outputs[i].byte_size) != 0) {
      destroy(runtime, inputs, input_count, outputs, output_count);
      return 1;
    }
    printf("%s{\"index\":%d,\"path\":\"output_%d.raw\",\"byte_size\":%zu}",
           i == 0 ? "" : ",", i, i, outputs[i].byte_size);
  }
  printf("]}\n");
  destroy(runtime, inputs, input_count, outputs, output_count);
  return 0;
}
