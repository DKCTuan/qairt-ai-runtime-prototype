#include <stdio.h>
#include <stdlib.h>

#include "ai_runtime.h"

static int get_file_size(const char *path, long *size_bytes)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return -1;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }
    long size = ftell(fp);
    fclose(fp);
    if (size < 0) {
        return -1;
    }
    *size_bytes = size;
    return 0;
}

static int load_input(const char *path, float *input, int input_count)
{
    long file_size = 0;
    const long expected_size = (long)sizeof(float) * (long)input_count;
    if (get_file_size(path, &file_size) != 0) {
        fprintf(stderr, "failed to stat input: %s\n", path);
        return -1;
    }
    if (file_size != expected_size) {
        fprintf(stderr,
                "input raw size mismatch: path=%s provided=%ld bytes expected=%ld bytes (%d float32)\n",
                path, file_size, expected_size, input_count);
        return -1;
    }

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

    int scores_to_print = result.score_count < 32 ? result.score_count : 32;
    printf("label=%d\n", result.label);
    printf("score_count=%d\n", result.score_count);
    printf("scores_head=");
    for (int i = 0; i < scores_to_print; ++i) {
        printf("%s%.9g", i == 0 ? "" : ",", result.scores[i]);
    }
    if (scores_to_print < result.score_count) {
        printf(",...");
    }
    printf("\n");
    printf("latency_ms=%.3f\n", result.latency_ms);
    dl_deinit();
    return 0;
}
