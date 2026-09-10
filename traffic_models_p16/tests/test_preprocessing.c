#include "traffic_p16.h"
#include "traffic_p16_deployment_config.h"

#include <assert.h>
#include <math.h>
#include <string.h>

int main(void)
{
    traffic_p16_packet_t packets[TRAFFIC_P16_WINDOW_SIZE];
    traffic_p16_vocab_entry_t vocab[] = {{0x14ca, 17}, {0x5db8, 23}};
    traffic_p16_config_t config = {{1.0f, 2.0f, 0.0f}, {2.0f, 4.0f, 1.0f}, vocab, 2, 24};
    float numeric[TRAFFIC_P16_NUMERIC_ELEMENTS];
    int32_t ids[TRAFFIC_P16_WINDOW_SIZE];
    traffic_p16_directional_packet_t directional[TRAFFIC_P16_WINDOW_SIZE];
    memset(packets, 0, sizeof(packets));
    for (int i = 0; i < TRAFFIC_P16_WINDOW_SIZE; ++i) {
        packets[i].timestamp_us = (uint64_t)i * 2000000U;
        packets[i].packet_length = 99;
        packets[i].source_ipv4 = 0xc0a8010aU;      /* 192.168.1.10 */
        packets[i].destination_ipv4 = 0x14ca327bU; /* 20.202.50.123 */
    }
    assert(traffic_p16_prepare_packets(packets, TRAFFIC_P16_WINDOW_SIZE, &config,
                                       numeric, ids) == TRAFFIC_P16_OK);
    assert(ids[0] == 17);
    assert(fabsf(numeric[0] - ((log1pf(99.0f) - 1.0f) / 2.0f)) < 1e-6f);
    assert(fabsf(numeric[1] - ((log1pf(0.0f) - 2.0f) / 4.0f)) < 1e-6f);
    assert(numeric[2] == 1.0f);
    assert(fabsf(numeric[4] - ((log1pf(2.0f) - 2.0f) / 4.0f)) < 1e-6f);
    packets[1].source_ipv4 = 0x14ca327bU;
    packets[1].destination_ipv4 = 0xc0a8010aU;
    assert(traffic_p16_prepare_packets(packets, TRAFFIC_P16_WINDOW_SIZE, &config,
                                       numeric, ids) == TRAFFIC_P16_OK);
    assert(ids[1] == 17 && numeric[5] == -1.0f);
    packets[1].timestamp_us = packets[0].timestamp_us - 1;
    assert(traffic_p16_prepare_packets(packets, TRAFFIC_P16_WINDOW_SIZE, &config,
                                       numeric, ids) == TRAFFIC_P16_INVALID_INPUT);
    packets[1].timestamp_us = 2000000;
    packets[1].source_ipv4 = 0x64400001U; /* 100.64.0.1 is neither private nor global. */
    packets[1].destination_ipv4 = 0x14ca327bU;
    assert(traffic_p16_prepare_packets(packets, TRAFFIC_P16_WINDOW_SIZE, &config,
                                       numeric, ids) == TRAFFIC_P16_OK);
    assert(ids[1] == 0 && numeric[5] == 1.0f);
    const traffic_p16_config_t *deployed = traffic_p16_default_config();
    packets[1].source_ipv4 = 0xc0a8010aU;
    packets[1].destination_ipv4 = 0x14ca327bU;
    assert(traffic_p16_prepare_packets(packets, TRAFFIC_P16_WINDOW_SIZE, deployed,
                                       numeric, ids) == TRAFFIC_P16_OK);
    assert(deployed->vocabulary_count == 141 && deployed->vocabulary_size_including_unk == 142);
    assert(ids[1] == 47);
    for (int i = 0; i < TRAFFIC_P16_WINDOW_SIZE; ++i) {
        directional[i].timestamp_us = (uint64_t)i * 2000000U;
        directional[i].packet_length = 99;
        directional[i].direction = i % 2 ? -1 : 1;
    }
    assert(traffic_p16_prepare_directional_packets(
               directional, TRAFFIC_P16_WINDOW_SIZE, 0x14ca327bU, &config,
               numeric, ids) == TRAFFIC_P16_OK);
    assert(ids[0] == 17 && ids[89] == 17);
    assert(numeric[2] == 1.0f && numeric[5] == -1.0f);
    directional[1].direction = 0;
    assert(traffic_p16_prepare_directional_packets(
               directional, TRAFFIC_P16_WINDOW_SIZE, 0x14ca327bU, &config,
               numeric, ids) == TRAFFIC_P16_INVALID_INPUT);
    directional[1].direction = -1;
    assert(traffic_p16_prepare_directional_packets(
               directional, TRAFFIC_P16_WINDOW_SIZE, 0x64400001U, &config,
               numeric, ids) == TRAFFIC_P16_INVALID_INPUT);
    assert(strcmp(traffic_p16_class_name(3), "Voice") == 0);
    assert(strcmp(traffic_p16_class_name(4), "VStream") == 0);
    return 0;
}
