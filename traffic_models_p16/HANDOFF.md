# Bàn giao SDK Traffic AI: Random Forest + TinyGRU P16

## Thành phần

| Thư viện | Vai trò |
| --- | --- |
| `libtraffic_ai.so` | **SDK công khai duy nhất:** Random Forest và API TinyGRU P16 |
| `libtiny_gru_p16_float32.so` | Model TinyGRU P16 hai input |

Không có `libtraffic_models.so`, `libtraffic_p16.so` hay model TinyGRU cũ trong
bundle unified này. Random Forest được biên dịch trực tiếp vào
`libtraffic_ai.so`; chỉ model TinyGRU P16 được giữ thành dependency riêng.

RF và P16 có class order khác nhau. Luôn dùng hàm/tài liệu của đúng API để đổi
label sang tên class:

```text
RF:  Background, Game, RTVideo, VStream, Voice
P16: Background, Game, RTVideo, Voice, VStream
```

## Runtime target

Bundle này dành cho Linux ARM64 OpenWrt/musl. Sao chép nguyên thư mục `lib/`
vào cùng deployment với ứng dụng. `libgcc_s.so.1` được kèm để standalone; có
thể bỏ nó chỉ khi firmware đã có `/lib/libgcc_s.so.1` tương thích.

Kiểm tra integrity sau khi giải nén:

```bash
sha256sum -c SHA256SUMS
```

## Random Forest

Giữ nguyên tích hợp RF hiện có. Một window là 90 packet sau khi bỏ 10 packet
đầu của flow; timestamp là microseconds không giảm. Input application và input
thật của API RF là:

| Field | C type | Số lượng | Đơn vị/ý nghĩa |
| --- | --- | ---: | --- |
| `timestamp_us` | `uint64_t` | 90 | microseconds, non-decreasing |
| `packet_length` | `uint32_t` | 90 | cùng định nghĩa với dữ liệu train RF |

SDK tự sinh vector 11 feature; application không tự tính 11 feature. API:

```c
float features[TRAFFIC_RF_FEATURES];
traffic_result rf;

traffic_rf_extract(timestamp_us, packet_length, 90, features);
traffic_rf_predict(features, TRAFFIC_RF_FEATURES, &rf);
```

`packet_length` phải cùng định nghĩa với lúc train RF. RF không sử dụng
direction hay IPv4 address.

## TinyGRU P16

Không đưa trực tiếp vector `[90][3]` cho P16. SDK tạo hai input model từ 90
packet thô sau khi bỏ 10 packet đầu. Một flow cần tối thiểu 100 packet: bỏ
packet gốc 1–10, rồi packet 11–100 trở thành window đầu tiên. Window tiếp theo
dùng stride 90.

### Input application → SDK: mode một remote IP cho một window

Đây là mode dùng cho app của anh Kiên: app truyền 90 bộ
`timestamp + length + direction` (tương đương 90×3 field) và **một** remote
global IPv4 dùng chung cho cả window. `timestamp_us` là timestamp tuyệt đối,
không phải IAT; SDK tự tính IAT từ các timestamp này.

```c
traffic_p16_directional_packet_t packets[TRAFFIC_P16_WINDOW_SIZE];
traffic_p16_result_t p16;
uint32_t remote_global_ipv4 = 0x14ca327bU; /* 20.202.50.123, host-order */

/* Per packet:
 * timestamp_us       : microseconds, non-decreasing within flow
 * packet_length      : same meaning as the training CSV's Length column
 * direction          : +1 local -> remote; -1 remote -> local
 */
if (traffic_p16_model_init() != TRAFFIC_P16_OK) {
    /* model/library loading error */
}

int status = traffic_p16_predict_directional_packets(
    packets, TRAFFIC_P16_WINDOW_SIZE,
    remote_global_ipv4,
    traffic_p16_default_config(), &p16);

const char *name = traffic_p16_class_name(p16.label);
traffic_p16_model_deinit();
```

`remote_global_ipv4` phải là IPv4 global-unicast ở **host-order** và phải thật
sự giống nhau cho tất cả 90 packet. SDK lookup `/16` một lần, rồi dùng ID đó
cho toàn bộ tensor `[1,90]`. Nếu window có nhiều remote IP, không dùng API
này; dùng API `traffic_p16_predict_packets()` với source/destination IPv4 cho
từng packet.

### API source/destination (tùy chọn)

API cũ `traffic_p16_predict_packets()` vẫn hỗ trợ khi app muốn truyền
source/destination IPv4 cho từng packet. SDK sẽ tự suy ra direction và remote
IP. Không dùng cả hai API cho cùng một window.

### Hai input thật của `.tflite`

SDK tạo đúng hai tensor này; application **không** tự tạo chúng:

| TFLite input | Shape / dtype | Giá trị ở packet `i` |
| --- | --- | --- |
| Numeric input #0 | `[1,90,3] float32` | channel 0: `(log1p(packet_length)-6.0075965)/1.3676786` |
|  |  | channel 1: `(log1p(max(IAT_seconds,0))-0.0020584022)/0.021586655` |
|  |  | channel 2: `(direction-(-0.16327207))/0.9865821` |
| P16 input #1 | `[1,90] int32` | embedding ID của remote global-unicast IPv4 `/16`; không có prefix hợp lệ là `0` (UNK) |
| Output #0 | `[1,5] float32` | five logits/scores; lấy index score lớn nhất làm `label` |

`IAT_seconds` tại packet đầu của window là `0`; các packet sau là chênh lệch
timestamp hiện tại với timestamp trước đó, đổi từ microseconds sang seconds.

IPv4 phải là **host-order**. Nếu capture header cung cấp network-order,
chuyển bằng `ntohl()` trước khi gán:

```c
packets[i].source_ipv4 = ntohl(ip_header->saddr);
packets[i].destination_ipv4 = ntohl(ip_header->daddr);
```

Direction và remote IP được quyết định như sau:

| Source | Destination | Direction | Remote dùng lookup `/16` |
| --- | --- | ---: | --- |
| private/local | global-unicast | `+1` | destination |
| global-unicast | private/local | `-1` | source |
| Các trường hợp khác | — | fallback theo local endpoint, hoặc `+1` | UNK `0` nếu không có remote global-unicast |

P16 tự làm toàn bộ direction, IAT, `log1p`, Z-score và lookup `/16` sang
embedding ID. Application không normalize hay lookup vocabulary.

## Link ứng dụng riêng

Ví dụ application nằm ở `bin/customer_app` và bundle nằm cạnh nó:

```bash
aarch64-openwrt-linux-musl-gcc customer_app.c \
  -I../include -L../lib \
  -ltraffic_ai \
  -Wl,-rpath,'$ORIGIN/../lib' \
  -o customer_app
```

Application chỉ cần `#include "traffic_ai.h"`; header này gom API RF và P16.

`bin/app_arm64` chỉ là smoke test P16, không thay thế test bằng
capture thật hoặc app khách hàng.
