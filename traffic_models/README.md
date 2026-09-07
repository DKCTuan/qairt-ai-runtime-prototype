# RF và Tiny GRU trên iGate

Luồng triển khai mới thay example model traffic_qos 240 số cũ. Các artifact/model
cũ được giữ để đối chiếu, không được dùng bởi Makefile này. Source app chính:
`traffic_models/app.c`; thay mảng input bằng đặc trưng của caller.

## Build

Từ root project:
```sh
make -C traffic_models \
  CC="$HOME/toolchains/gcc-arm-10.2-aarch64/bin/aarch64-none-linux-gnu-gcc" \
  STRIP="$HOME/toolchains/gcc-arm-10.2-aarch64/bin/aarch64-none-linux-gnu-strip"
qemu-aarch64 -L "$HOME/toolchains/gcc-arm-10.2-aarch64/aarch64-none-linux-gnu/libc" dist/traffic_models/app_arm64
```
Chép `dist/traffic_models/app_arm64` và `libtraffic_models.so` vào cùng thư mục
trên iGate. Chạy `./app_arm64`. RF không cần TFLite, model cũ hoặc glibc riêng.
Header public: `traffic_models/traffic_models.h`. RF gọi predict nhiều lần,
không yêu cầu khởi tạo runtime.

## Input/output

- RF: float[11], 20 cây, thứ tự: frac_mid, len_p95_p05, len_mean,
  r__iat_max_med__frac_1ms, len_p10, x__len_p75_p25__duty,
  x__frac_20ms__frac_b2b, iat_after_big_ratio, iat_max, len_bimod,
  r__frac_5ms__frac_20ms. Không áp dụng mean/std GRU vào RF.
- GRU dự kiến: float32[1,90,3], packet-major: delta_time, direction, packet_length.
  traffic_gru_prepare chuẩn hóa kênh 0 và 2 theo metadata, giữ direction.
  Nếu đã chuẩn hóa thì truyền thẳng vào predict, không chuẩn hóa lần nữa.
- Output: 5 score, label 0 Background, 1 Game, 2 RTVideo, 3 VStream, 4 Voice.
- RF latency đo suy luận, không gồm feature extraction. API kiểm tra số phần tử
  và giá trị hữu hạn. Các lời gọi tuần tự, chưa hỗ trợ GRU đa luồng.

## GRU đã có model TFLite

Đã kiểm tra tiny_gru_float32.tflite: input [1,90,3] float32, output [1,5]
float32. Output là scores/logits, không phải xác suất. Xem [DUAL_MODELS.md](DUAL_MODELS.md)
để build và nạp cùng RF. Artifact dual ở dist/dual_models; app chọn rf/gru/both.
Mặc định Makefile vẫn build RF-only nếu không truyền GRU_LIBRARY; khi đó API
GRU trả -2. Không fallback về model cũ. Gọi init một lần, predict mỗi mẫu,
deinit khi kết thúc. Đã chạy model mới bằng QEMU; chờ kiểm thử iGate.

## Nguồn và kiểm thử

vendor chứa nguyên rf_model.c/.h và preprocessing_metadata.json từ gói
Tuan/tflite_export_12_08_2026/tflite_export_12_08_2026 người dùng cung cấp.
Chưa tích hợp extractor: còn thiếu types.h, metadata_extractor.h; header bên cung
cấp cảnh báo skip packet 90 so với 10 lúc train và yêu cầu đủ cửa sổ 90 packet.
Cần xác nhận các quy tắc này khi nối dữ liệu thực tế.
Mẫu RF lấy từ RF_MODEL_TEST của source, không phải đánh giá accuracy.

```sh
gcc -std=c11 -Wall -Wextra -Itraffic_models traffic_models/tests/test_api.c \
  traffic_models/traffic_models.c traffic_models/vendor/rf_model.c -lm -o /tmp/traffic_api_test
/tmp/traffic_api_test
```
