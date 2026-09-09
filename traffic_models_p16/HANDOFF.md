# Bàn giao SDK Traffic AI: Random Forest + TinyGRU P16

## Thành phần

| Thư viện | Vai trò |
| --- | --- |
| `libtraffic_models.so` | Random Forest và API RF hiện có |
| `libtiny_gru_float32.so` | Dependency của SDK RF hiện có |
| `libtraffic_p16.so` | Preprocessing và API TinyGRU P16 |
| `libtiny_gru_p16_float32.so` | Model TinyGRU P16 hai input |

Hai GRU có class order khác nhau. Luôn dùng hàm/tài liệu của đúng SDK để đổi
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
đầu của flow; timestamp là microseconds không giảm. API:

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
packet thô sau khi bỏ 10 packet đầu:

```c
traffic_p16_packet_t packets[TRAFFIC_P16_WINDOW_SIZE];
traffic_p16_result_t p16;

/* Per packet: timestamp_us, packet_length, source_ipv4, destination_ipv4. */
if (traffic_p16_model_init() != TRAFFIC_P16_OK) {
    /* model/library loading error */
}

int status = traffic_p16_predict_packets(
    packets, TRAFFIC_P16_WINDOW_SIZE,
    traffic_p16_default_config(), &p16);

const char *name = traffic_p16_class_name(p16.label);
traffic_p16_model_deinit();
```

IPv4 phải là **host-order**. Nếu capture header cung cấp network-order,
chuyển bằng `ntohl()` trước khi gán:

```c
packets[i].source_ipv4 = ntohl(ip_header->saddr);
packets[i].destination_ipv4 = ntohl(ip_header->daddr);
```

P16 tự làm `direction`, `IAT`, `log1p`, Z-score ba channel và lookup remote
global-unicast IPv4 `/16` sang embedding ID. `packet_length` phải có đúng ý
nghĩa của cột `Length` dùng lúc train P16.

## Link ứng dụng riêng

Ví dụ application nằm ở `bin/customer_app` và bundle nằm cạnh nó:

```bash
aarch64-openwrt-linux-musl-gcc customer_app.c \
  -I../include -L../lib \
  -ltraffic_models -ltraffic_p16 \
  -Wl,-rpath,'$ORIGIN/../lib' \
  -o customer_app
```

`bin/app_p16_arm64` chỉ là smoke test P16, không thay thế test bằng capture
thật hoặc app khách hàng.
