/*
 * Generic client for a generated libai_model.so.
 *
 * Usage:
 *   ./ai_model_runner INPUT.raw
 *   producer_that_writes_float32_raw | ./ai_model_runner -
 *
 * INPUT.raw contains one or more consecutive input tensors in native
 * little-endian float32 form.  The runner asks the embedded model for its
 * element counts after ai_model_init(), so it works for both the legacy
 * QoS model (240 floats) and TinyGRU (270 floats) without recompilation.
 * It deliberately does not perform feature extraction or normalization:
 * callers must supply tensors following the model's documented contract.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ai_model.h"

static void print_scores(const float *scores, size_t score_count)
{
    printf("scores=");
    for (size_t index = 0; index < score_count; ++index) {
        printf("%s%.9g", index == 0 ? "" : ",", scores[index]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    FILE *input_file = NULL;
    float *input = NULL;
    float *scores = NULL;
    size_t input_count = 0;
    size_t output_capacity = 0;
    size_t sample = 0;
    int exit_code = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s INPUT.raw|-\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "-") == 0) {
        input_file = stdin;
    } else {
        input_file = fopen(argv[1], "rb");
        if (input_file == NULL) {
            fprintf(stderr, "cannot open %s: %s\n", argv[1], strerror(errno));
            return 1;
        }
    }

    if (ai_model_init() != 0 ||
        ai_model_get_io_count(&input_count, &output_capacity) != 0 ||
        input_count == 0 || output_capacity == 0) {
        fprintf(stderr, "cannot initialize model or determine its I/O contract\n");
        goto done;
    }
    input = malloc(input_count * sizeof(*input));
    scores = malloc(output_capacity * sizeof(*scores));
    if (input == NULL || scores == NULL) {
        fprintf(stderr, "cannot allocate model I/O buffers\n");
        goto done;
    }

    for (;;) {
        size_t read_count = fread(input, sizeof(*input), input_count, input_file);
        size_t score_count = 0;
        int label = -1;
        double latency_ms = 0.0;

        if (read_count == 0) {
            if (ferror(input_file)) {
                fprintf(stderr, "input read error: %s\n", strerror(errno));
                goto done;
            }
            break;
        }
        if (read_count != input_count) {
            fprintf(stderr,
                    "incomplete input tensor: expected %zu float32 values (%zu bytes), got %zu\n",
                    input_count, input_count * sizeof(*input), read_count);
            goto done;
        }
        if (ai_model_predict(input, input_count, scores, output_capacity,
                             &score_count, &label, &latency_ms) != 0) {
            fprintf(stderr, "ai_model_predict failed for sample %zu\n", sample);
            goto done;
        }
        printf("sample=%zu\ninput_count=%zu\nscore_count=%zu\n",
               sample, input_count, score_count);
        print_scores(scores, score_count);
        printf("label=%d\nlatency_ms=%.3f\n", label, latency_ms);
        ++sample;
    }
    if (sample == 0) {
        fprintf(stderr, "input contains no complete float32 tensor\n");
        goto done;
    }
    exit_code = 0;

done:
    free(scores);
    free(input);
    ai_model_deinit();
    if (input_file != NULL && input_file != stdin) {
        fclose(input_file);
    }
    return exit_code;
}
