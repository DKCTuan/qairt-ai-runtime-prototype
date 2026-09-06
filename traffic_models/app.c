#include <stdio.h>
#include "traffic_models.h"

int main(void) {
    /* Replace with 11 features from your feature extractor, in vendor header order.
     * This sample comes from RF_MODEL_TEST in the supplied rf_model.c. */
    float input[TRAFFIC_RF_FEATURES] = {
        0, 26.9784f, 832.711f, 100, 60, 19.7368f, 0, 0, 0.000524f, 0.995556f, 0
    };
    traffic_result result;
    if (traffic_rf_predict(input, TRAFFIC_RF_FEATURES, &result)) {
        fprintf(stderr, "RF inference failed\n"); return 1;
    }
    printf("model=RandomForest\ninput_count=%d\nscores=", TRAFFIC_RF_FEATURES);
    for (int i = 0; i < TRAFFIC_CLASSES; ++i)
        printf("%s%.9g", i ? "," : "", result.scores[i]);
    printf("\nlabel=%d\nclass=%s\nlatency_ms=%.3f\n", result.label,
           traffic_class_name(result.label), result.latency_ms);
    return 0;
}
