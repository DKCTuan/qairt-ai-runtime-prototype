#include "traffic_p16.h"

#include <math.h>
#include <string.h>

#ifdef TRAFFIC_P16_WITH_GRU
#include "ai_model.h"
static int p16_ready;
#endif

static int in_range(uint32_t ip, uint32_t network, unsigned int prefix_bits)
{
    uint32_t mask = prefix_bits == 0 ? 0U : 0xffffffffU << (32U - prefix_bits);
    return (ip & mask) == network;
}

/* Python ipaddress.is_private semantics used by the training notebook. The
 * ranges beyond RFC1918 matter because is_private determines direction. */
static int ipv4_private(uint32_t ip)
{
    return in_range(ip, 0x00000000U, 8) || in_range(ip, 0x0a000000U, 8) ||
           in_range(ip, 0x7f000000U, 8) || in_range(ip, 0xa9fe0000U, 16) ||
           in_range(ip, 0xac100000U, 12) ||
           (in_range(ip, 0xc0000000U, 24) && ip != 0xc0000009U && ip != 0xc000000aU) ||
           in_range(ip, 0xc0000200U, 24) ||
           in_range(ip, 0xc01fc400U, 24) || in_range(ip, 0xc034c100U, 24) ||
           in_range(ip, 0xc0586300U, 24) || in_range(ip, 0xc0a80000U, 16) ||
           in_range(ip, 0xc0af3000U, 24) || in_range(ip, 0xc6120000U, 15) ||
           in_range(ip, 0xc6336400U, 24) || in_range(ip, 0xcb007100U, 24) ||
           in_range(ip, 0xe0000000U, 4);
}

static int ipv4_global_unicast(uint32_t ip)
{
    /* ipaddress.is_global is false for the carrier-grade shared range even
     * though that range is not is_private. Multicast is excluded separately. */
    return !ipv4_private(ip) && !in_range(ip, 0x64400000U, 10) &&
           !in_range(ip, 0xe0000000U, 4);
}

static int32_t lookup_prefix16(const traffic_p16_config_t *config, uint16_t prefix16)
{
    size_t low = 0, high = config->vocabulary_count;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        uint16_t candidate = config->vocabulary[mid].prefix16;
        if (candidate == prefix16) return config->vocabulary[mid].embedding_id;
        if (candidate < prefix16) low = mid + 1;
        else high = mid;
    }
    return 0;
}

static int valid_config(const traffic_p16_config_t *config)
{
    if (config == NULL || config->vocabulary_size_including_unk < 1 ||
        (config->vocabulary_count != 0 && config->vocabulary == NULL)) return 0;
    for (size_t i = 0; i < TRAFFIC_P16_NUMERIC_CHANNELS; ++i) {
        if (!isfinite(config->mean[i]) || !isfinite(config->std[i]) || config->std[i] <= 0.0f)
            return 0;
    }
    for (size_t i = 0; i < config->vocabulary_count; ++i) {
        int32_t id = config->vocabulary[i].embedding_id;
        if (id <= 0 || id >= config->vocabulary_size_including_unk ||
            (i != 0 && config->vocabulary[i - 1].prefix16 >= config->vocabulary[i].prefix16))
            return 0;
    }
    return 1;
}

int traffic_p16_prepare_packets(const traffic_p16_packet_t *packets,
                                size_t packet_count,
                                const traffic_p16_config_t *config,
                                float numeric[TRAFFIC_P16_NUMERIC_ELEMENTS],
                                int32_t p16_ids[TRAFFIC_P16_WINDOW_SIZE])
{
    if (packets == NULL || numeric == NULL || p16_ids == NULL ||
        packet_count != TRAFFIC_P16_WINDOW_SIZE || !valid_config(config))
        return TRAFFIC_P16_INVALID_INPUT;

    for (size_t i = 0; i < packet_count; ++i) {
        uint32_t remote = 0;
        int src_local = ipv4_private(packets[i].source_ipv4);
        int dst_local = ipv4_private(packets[i].destination_ipv4);
        float direction;
        float iat_seconds;
        float values[TRAFFIC_P16_NUMERIC_CHANNELS];

        if (i != 0 && packets[i].timestamp_us < packets[i - 1].timestamp_us)
            return TRAFFIC_P16_INVALID_INPUT;

        if (src_local && ipv4_global_unicast(packets[i].destination_ipv4)) {
            remote = packets[i].destination_ipv4;
            direction = 1.0f;
        } else if (dst_local && ipv4_global_unicast(packets[i].source_ipv4)) {
            remote = packets[i].source_ipv4;
            direction = -1.0f;
        } else if (src_local && !dst_local) {
            direction = 1.0f;
        } else if (dst_local && !src_local) {
            direction = -1.0f;
        } else {
            direction = 1.0f;
        }

        iat_seconds = i == 0 ? 0.0f :
            (float)((packets[i].timestamp_us - packets[i - 1].timestamp_us) / 1000000.0);
        if (iat_seconds < 0.0f) iat_seconds = 0.0f;
        values[0] = log1pf((float)packets[i].packet_length);
        values[1] = log1pf(iat_seconds);
        values[2] = direction;
        for (size_t channel = 0; channel < TRAFFIC_P16_NUMERIC_CHANNELS; ++channel) {
            float normalized = (values[channel] - config->mean[channel]) / config->std[channel];
            if (!isfinite(normalized)) return TRAFFIC_P16_INVALID_INPUT;
            numeric[i * TRAFFIC_P16_NUMERIC_CHANNELS + channel] = normalized;
        }
        p16_ids[i] = remote == 0 ? 0 : lookup_prefix16(config, (uint16_t)(remote >> 16));
    }
    return TRAFFIC_P16_OK;
}

int traffic_p16_model_init(void)
{
#ifdef TRAFFIC_P16_WITH_GRU
    int inputs = 0, outputs = 0;
    dl_tensor_info_t numeric, p16, scores;
    if (p16_ready) return TRAFFIC_P16_OK;
    if (ai_model_init() != 0 || ai_model_get_tensor_count(&inputs, &outputs) != 0 ||
        inputs != 2 || outputs != 1 || ai_model_get_input_info(0, &numeric) != 0 ||
        ai_model_get_input_info(1, &p16) != 0 || ai_model_get_output_info(0, &scores) != 0 ||
        numeric.dtype != DL_DTYPE_FLOAT32 || numeric.element_count != TRAFFIC_P16_NUMERIC_ELEMENTS ||
        p16.dtype != DL_DTYPE_INT32 || p16.element_count != TRAFFIC_P16_WINDOW_SIZE ||
        scores.dtype != DL_DTYPE_FLOAT32 || scores.element_count != TRAFFIC_P16_CLASS_COUNT) {
        ai_model_deinit();
        return TRAFFIC_P16_INFERENCE_ERROR;
    }
    p16_ready = 1;
    return TRAFFIC_P16_OK;
#else
    return TRAFFIC_P16_MODEL_UNAVAILABLE;
#endif
}

void traffic_p16_model_deinit(void)
{
#ifdef TRAFFIC_P16_WITH_GRU
    if (p16_ready) ai_model_deinit();
    p16_ready = 0;
#endif
}

int traffic_p16_predict(const float numeric[TRAFFIC_P16_NUMERIC_ELEMENTS],
                        const int32_t p16_ids[TRAFFIC_P16_WINDOW_SIZE],
                        traffic_p16_result_t *result)
{
    if (numeric == NULL || p16_ids == NULL || result == NULL) return TRAFFIC_P16_INVALID_INPUT;
    memset(result, 0, sizeof(*result));
    result->label = -1;
    for (size_t i = 0; i < TRAFFIC_P16_NUMERIC_ELEMENTS; ++i)
        if (!isfinite(numeric[i])) return TRAFFIC_P16_INVALID_INPUT;
#ifdef TRAFFIC_P16_WITH_GRU
    if (!p16_ready) return TRAFFIC_P16_MODEL_UNAVAILABLE;
    dl_tensor_t inputs[2] = {
        {(void *)numeric, sizeof(float) * TRAFFIC_P16_NUMERIC_ELEMENTS},
        {(void *)p16_ids, sizeof(int32_t) * TRAFFIC_P16_WINDOW_SIZE},
    };
    dl_tensor_t outputs[1] = {{result->scores, sizeof(result->scores)}};
    if (ai_model_predict_tensors(inputs, 2, outputs, 1, &result->latency_ms) != 0)
        return TRAFFIC_P16_INFERENCE_ERROR;
    result->label = 0;
    for (int i = 1; i < TRAFFIC_P16_CLASS_COUNT; ++i)
        if (result->scores[i] > result->scores[result->label]) result->label = i;
    return TRAFFIC_P16_OK;
#else
    return TRAFFIC_P16_MODEL_UNAVAILABLE;
#endif
}

int traffic_p16_predict_packets(const traffic_p16_packet_t *packets,
                                size_t packet_count,
                                const traffic_p16_config_t *config,
                                traffic_p16_result_t *result)
{
    float numeric[TRAFFIC_P16_NUMERIC_ELEMENTS];
    int32_t p16_ids[TRAFFIC_P16_WINDOW_SIZE];
    int status = traffic_p16_prepare_packets(packets, packet_count, config, numeric, p16_ids);
    if (status != TRAFFIC_P16_OK) return status;
    return traffic_p16_predict(numeric, p16_ids, result);
}
