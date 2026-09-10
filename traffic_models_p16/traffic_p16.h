#ifndef TRAFFIC_P16_H
#define TRAFFIC_P16_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TRAFFIC_P16_WINDOW_SIZE 90
#define TRAFFIC_P16_NUMERIC_CHANNELS 3
#define TRAFFIC_P16_NUMERIC_ELEMENTS \
    (TRAFFIC_P16_WINDOW_SIZE * TRAFFIC_P16_NUMERIC_CHANNELS)
#define TRAFFIC_P16_CLASS_COUNT 5

enum {
    TRAFFIC_P16_OK = 0,
    TRAFFIC_P16_INVALID_INPUT = -1,
    TRAFFIC_P16_MODEL_UNAVAILABLE = -2,
    TRAFFIC_P16_INFERENCE_ERROR = -3,
};

/* IPv4 addresses are host-order, for example 192.168.1.10 is 0xc0a8010a.
 * packet_length must use the same definition as the training CSV's Length
 * column; it is not necessarily an IPv4 header total-length field. */
typedef struct {
    uint64_t timestamp_us;
    uint32_t packet_length;
    uint32_t source_ipv4;
    uint32_t destination_ipv4;
} traffic_p16_packet_t;

/* Use this form when the caller already owns flow direction and all packets
 * in the 90-packet window have the same global remote endpoint. direction is
 * +1 for local-to-remote and -1 for remote-to-local. */
typedef struct {
    uint64_t timestamp_us;
    uint32_t packet_length;
    int8_t direction;
} traffic_p16_directional_packet_t;

/* prefix16 is the high 16 bits of an IPv4 address, e.g. 20.202.0.0/16 is
 * 0x14ca. Entries must be strictly sorted by prefix16; ID 0 is reserved for
 * unknown or invalid remote endpoints. */
typedef struct {
    uint16_t prefix16;
    int32_t embedding_id;
} traffic_p16_vocab_entry_t;

typedef struct {
    float mean[TRAFFIC_P16_NUMERIC_CHANNELS];
    float std[TRAFFIC_P16_NUMERIC_CHANNELS];
    const traffic_p16_vocab_entry_t *vocabulary;
    size_t vocabulary_count;
    int32_t vocabulary_size_including_unk;
} traffic_p16_config_t;

typedef struct {
    float scores[TRAFFIC_P16_CLASS_COUNT];
    int label;
    double latency_ms;
} traffic_p16_result_t;

/* Implements exactly the P16 training contract:
 * numeric[:,0] = Z(log1p(packet_length))
 * numeric[:,1] = Z(log1p(max(IAT_seconds, 0)))
 * numeric[:,2] = Z(direction)
 * p16_ids[:]    = vocabulary ID of the global-unicast remote IPv4 /16,
 *                  or zero (UNK).
 *
 * The application passes the post-skip 90-packet window. Therefore with
 * SKIP_PACKETS=10, its first entry is original flow packet 11. */
int traffic_p16_prepare_packets(const traffic_p16_packet_t *packets,
                                size_t packet_count,
                                const traffic_p16_config_t *config,
                                float numeric[TRAFFIC_P16_NUMERIC_ELEMENTS],
                                int32_t p16_ids[TRAFFIC_P16_WINDOW_SIZE]);

/* remote_global_ipv4 is host-order and is applied to every packet in this
 * window. It must be global-unicast; an unlisted /16 still maps to ID 0
 * (UNK), matching the trained model contract. */
int traffic_p16_prepare_directional_packets(
    const traffic_p16_directional_packet_t *packets, size_t packet_count,
    uint32_t remote_global_ipv4, const traffic_p16_config_t *config,
    float numeric[TRAFFIC_P16_NUMERIC_ELEMENTS],
    int32_t p16_ids[TRAFFIC_P16_WINDOW_SIZE]);

/* These functions use the generated model library's multi-tensor ABI.
 * The model must expose exactly float32 [1,90,3], int32 [1,90], and float32
 * [1,5], in that discovered tensor order. */
int traffic_p16_model_init(void);
void traffic_p16_model_deinit(void);
int traffic_p16_predict(const float numeric[TRAFFIC_P16_NUMERIC_ELEMENTS],
                        const int32_t p16_ids[TRAFFIC_P16_WINDOW_SIZE],
                        traffic_p16_result_t *result);

/* Convenience operation for an SDK that owns preprocessing. */
int traffic_p16_predict_packets(const traffic_p16_packet_t *packets,
                                size_t packet_count,
                                const traffic_p16_config_t *config,
                                traffic_p16_result_t *result);

int traffic_p16_predict_directional_packets(
    const traffic_p16_directional_packet_t *packets, size_t packet_count,
    uint32_t remote_global_ipv4, const traffic_p16_config_t *config,
    traffic_p16_result_t *result);

#ifdef __cplusplus
}
#endif

#endif
