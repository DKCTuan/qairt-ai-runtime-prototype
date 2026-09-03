/* Example application source for standalone-tflite --app-source.
 * It includes only the project's API, never a TensorFlow header. */
#include <stdio.h>
#include <stdlib.h>

#include "ai_runtime.h"

static int read_floats(const char *path, float *data, int count) {
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) return -1;
    size_t read_count = fread(data, sizeof(*data), (size_t)count, fp);
    int extra = fgetc(fp);
    fclose(fp);
    return read_count == (size_t)count && extra == EOF ? 0 : -1;
}

int main(int argc, char **argv) {
    int feature_num = 0;
    int output_num = 0;
    float *input;
    int label;

    if (argc != 2 || dl_init() != 0 ||
        dl_get_io_count(&feature_num, &output_num) != 0 || feature_num <= 0) {
        fprintf(stderr, "Usage: %s input_float32.raw\n", argv[0]);
        return 1;
    }
    input = malloc((size_t)feature_num * sizeof(*input));
    if (input == NULL || read_floats(argv[1], input, feature_num) != 0) {
        fprintf(stderr, "input must contain exactly %d float32 values\n", feature_num);
        free(input);
        dl_deinit();
        return 1;
    }
    label = dl_inference(input);
    printf("%d\n", label);
    free(input);
    dl_deinit();
    return label < 0 ? 1 : 0;
}
