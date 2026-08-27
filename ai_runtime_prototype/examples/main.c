#include <stdio.h>
#include <stdlib.h>

#include "ai_runtime.h"

static int load_input(const char *path, float *input, int input_count)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return -1;
    }

    size_t read_count = fread(input, sizeof(float), (size_t)input_count, fp);
    fclose(fp);

    return read_count == (size_t)input_count ? 0 : -1;
}

int main(void)
{
    const char *input_path = getenv("DL_INPUT_RAW");
    if (input_path == NULL || input_path[0] == '\0') {
        input_path = "/home/congtuan/model_test/input_seq.raw";
    }

    if (dl_init() != 0) {
        fprintf(stderr, "dl_init failed\n");
        return 1;
    }

    /* Kich thuoc input/output khong con hardcode - lay dong tu model that
     * sau khi backend da doc metadata (dl_init()). Nho vay app nay chay
     * duoc voi bat ky model nao da convert vao pipeline, khong chi rieng
     * traffic_qos_model, ma khong can sua/rebuild main.c. */
    int input_count = 0;
    int output_count = 0;
    if (dl_get_io_count(&input_count, &output_count) != 0) {
        fprintf(stderr, "dl_get_io_count failed\n");
        dl_deinit();
        return 1;
    }

    float *input = (float *)malloc(sizeof(float) * (size_t)input_count);
    if (input == NULL) {
        fprintf(stderr, "failed to allocate input buffer (%d floats)\n", input_count);
        dl_deinit();
        return 1;
    }

    if (load_input(input_path, input, input_count) != 0) {
        fprintf(stderr, "failed to load input: %s\n", input_path);
        free(input);
        dl_deinit();
        return 1;
    }

    dl_result_t result;
    if (dl_inference_ex(input, &result) != 0) {
        fprintf(stderr, "dl_inference failed\n");
        free(input);
        dl_deinit();
        return 1;
    }
    free(input);

    printf("label=%d\n", result.label);
    printf("scores=");
    for (int i = 0; i < result.score_count; ++i) {
        printf("%s%.9g", i == 0 ? "" : ",", result.scores[i]);
    }
    printf("\n");
    printf("latency_ms=%.3f\n", result.latency_ms);

    dl_deinit();
    return 0;
}
