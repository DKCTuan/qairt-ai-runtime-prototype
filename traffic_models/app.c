#include <stdio.h>
#include <string.h>
#include "traffic_models.h"

static void print_result(const char *model, int count, const traffic_result *r) {
    printf("model=%s\ninput_count=%d\nscores=", model, count);
    for (int i = 0; i < TRAFFIC_CLASSES; ++i)
        printf("%s%.9g", i ? "," : "", r->scores[i]);
    printf("\nlabel=%d\nclass=%s\nlatency_ms=%.3f\n", r->label,
           traffic_class_name(r->label), r->latency_ms);
}

static int run_rf(void) {
    /* Replace with 11 features from your feature extractor, in vendor header order.
     * This sample comes from RF_MODEL_TEST in the supplied rf_model.c. */
    float input[TRAFFIC_RF_FEATURES] = {
        0, 26.9784f, 832.711f, 100, 60, 19.7368f, 0, 0, 0.000524f, 0.995556f, 0
    };
    traffic_result result;
    if (traffic_rf_predict(input, TRAFFIC_RF_FEATURES, &result)) {
        fprintf(stderr, "RF inference failed\n"); return 1;
    }
    print_result("RandomForest", TRAFFIC_RF_FEATURES, &result);
    return 0;
}

static int run_gru(const char *input_path, int already_normalized) {
    float raw[TRAFFIC_GRU_FEATURES];
    float input[TRAFFIC_GRU_FEATURES];
    if (input_path) {
        /* A file passed to "gru" is raw [90][3] data.  Normalize it here so
         * this test app has the same contract as a production caller. */
        FILE *fp = fopen(input_path, "rb");
        if (!fp) { perror(input_path); return 1; }
        size_t n = fread(raw, sizeof(float), TRAFFIC_GRU_FEATURES, fp);
        int extra = fgetc(fp);
        int failed = ferror(fp);
        fclose(fp);
        if (n != TRAFFIC_GRU_FEATURES || extra != EOF || failed) {
            fprintf(stderr, "GRU input file must contain exactly 1080 bytes\n");
            return 1;
        }
        if (already_normalized) {
            memcpy(input, raw, sizeof(input));
        } else if (traffic_gru_prepare(raw, TRAFFIC_GRU_FEATURES, input)) {
            fprintf(stderr, "GRU preprocessing failed\n");
            return 1;
        }
    } else {
        /* Synthetic packets for smoke testing only. Replace with caller's
         * packet-major features. Not the same sample as the RF test vector. */
        for (int p = 0; p < TRAFFIC_GRU_PACKETS; ++p) {
            input[3*p] = 0.001f * (float)(p % 7);
            input[3*p+1] = p % 2 ? -1.0f : 1.0f;
            input[3*p+2] = 60.0f + (float)((p * 37) % 1400);
        }
        if (traffic_gru_prepare(input, TRAFFIC_GRU_FEATURES, input)) return 1;
    }
    int status = traffic_gru_init();
    if (status) {
        fprintf(stderr, "GRU init failed: %d (RF-only build returns -2)\n", status);
        return 1;
    }
    traffic_result result;
    status = traffic_gru_predict(input, TRAFFIC_GRU_FEATURES, &result);
    traffic_gru_deinit();
    if (status) { fprintf(stderr, "GRU prediction failed: %d\n", status); return 1; }
    print_result("TinyGRU", TRAFFIC_GRU_FEATURES, &result);
    return 0;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "rf";
    if (argc > 3 || (strcmp(mode, "rf") && strcmp(mode, "gru") &&
                     strcmp(mode, "gru-normalized") && strcmp(mode, "both")) ||
        (argc == 3 && !strcmp(mode, "rf"))) {
        fprintf(stderr, "Usage: %s [rf|gru|gru-normalized|both] [gru_input.raw]\n", argv[0]);
        return 1;
    }
    if ((!strcmp(mode, "rf") || !strcmp(mode, "both")) && run_rf()) return 1;
    if (!strcmp(mode, "gru") || !strcmp(mode, "gru-normalized") || !strcmp(mode, "both"))
        return run_gru(argc == 3 ? argv[2] : NULL, !strcmp(mode, "gru-normalized"));
    return 0;
}
