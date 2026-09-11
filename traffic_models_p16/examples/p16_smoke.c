#include "traffic_ai.h"

#include <inttypes.h>
#include <stdio.h>

int main(void)
{
    traffic_p16_packet_t packets[TRAFFIC_P16_WINDOW_SIZE];
    traffic_p16_result_t result;
    for (size_t i = 0; i < TRAFFIC_P16_WINDOW_SIZE; ++i) {
        packets[i].timestamp_us = (uint64_t)i * 20000U;
        packets[i].packet_length = 1200U;
        packets[i].source_ipv4 = 0xc0a8ef26U;      /* 192.168.239.38 */
        packets[i].destination_ipv4 = 0x14ca327bU; /* 20.202.50.123 */
    }
    if (traffic_p16_model_init() != TRAFFIC_P16_OK) {
        fprintf(stderr, "traffic_p16_model_init failed\n");
        return 1;
    }
    int status = traffic_p16_predict_packets(packets, TRAFFIC_P16_WINDOW_SIZE,
                                             traffic_p16_default_config(), &result);
    traffic_p16_model_deinit();
    if (status != TRAFFIC_P16_OK) {
        fprintf(stderr, "traffic_p16_predict_packets failed: %d\n", status);
        return 1;
    }
    printf("model=%s\npreprocessing=%s\nskip_packets=%zu\nlabel=%d\nclass=%s\n"
           "confidence=%.8g\naccepted=%d\nscores=",
           traffic_ai_model_id(), traffic_ai_preprocessing_id(),
           traffic_ai_skip_packets(), result.label, traffic_p16_class_name(result.label),
           result.confidence, result.accepted);
    for (int i = 0; i < TRAFFIC_P16_CLASS_COUNT; ++i)
        printf("%s%.8g", i == 0 ? "" : ",", result.scores[i]);
    printf("\nlatency_ms=%.3f\n", result.latency_ms);
    return 0;
}
