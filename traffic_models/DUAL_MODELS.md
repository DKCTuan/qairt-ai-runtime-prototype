# Chạy RF và Tiny GRU bằng cùng một app

Source app: `traffic_models/app.c`. RF nhận 11 feature; Tiny GRU mới nhận
float32 `[1,90,3]`, trả float32 `[1,5]`. Output GRU là scores/logits, không phải
xác suất: giữ nguyên giá trị và lấy argmax để chọn label. Mapping dùng metadata
các anh cung cấp: Background, Game, RTVideo, VStream, Voice.
Không dùng model traffic_qos 240 số trong bản dual này; bản cũ vẫn ở
`dist/old_tflite_minimal` để giữ kết quả lịch sử.

## Build trên WSL

Chạy từ root repository (`~/qairt_sdk` ở máy cũ hoặc `~/workspace/qairt_sdk`
ở máy WSL mới). Cần model do người dùng cung cấp, không có trong GitHub.

```sh
export ARM64_TOOLCHAIN="$HOME/toolchains/gcc-arm-10.2-aarch64"
mkdir -p ~/model_test/tiny_gru
cp "/mnt/c/Users/DINH KIEU CONG TUAN/Downloads/tiny_gru_float32.tflite" ~/model_test/tiny_gru/

python3 ai_runtime_prototype/tools/model_deploy.py static-library \
  --shared --quantize none \
  --model ~/model_test/tiny_gru/tiny_gru_float32.tflite \
  --output dist/dual_models/libtiny_gru.so \
  --tensorflow-root ~/tensorflow \
  --aarch64-toolchain "$ARM64_TOOLCHAIN" --force

"$ARM64_TOOLCHAIN/bin/aarch64-none-linux-gnu-strip" --strip-unneeded dist/dual_models/libtiny_gru.so

make -C traffic_models OUT=../dist/dual_models \
  GRU_LIBRARY=../dist/dual_models/libtiny_gru.so \
  CC="$ARM64_TOOLCHAIN/bin/aarch64-none-linux-gnu-gcc" \
  STRIP="$ARM64_TOOLCHAIN/bin/aarch64-none-linux-gnu-strip"

qemu-aarch64 -L "$ARM64_TOOLCHAIN/aarch64-none-linux-gnu/libc" dist/dual_models/app_arm64 both
```

Nếu WSL cũ dùng config toolchain riêng, thêm
`--aarch64-toolchain-config "$HOME/toolchains/tf-arm10-bazel-config"` vào lệnh
model_deploy. Không cần config riêng nếu TensorFlow đã được cấu hình đúng GCC 10.2.
Makefile build lại wrapper/app mỗi lần để việc bật/tắt GRU không giữ bản cũ.

## Nạp iGate

Trên WSL, phục vụ đúng thư mục chứa ba artifact; tắt server port 8000 cũ trước:
```sh
python3 -m http.server 8000 --bind 0.0.0.0 --directory dist/dual_models
```
Trên iGate (thay 192.168.1.100 nếu Ethernet Windows đổi IP; giữ port proxy
Windows tới WSL như quy trình trước). Dừng app cũ trước khi tải lại:
```sh
mkdir -p /tmp/dual_models
/userfs/bin/wget -O /tmp/dual_models/app_arm64 http://192.168.1.100:8000/app_arm64
/userfs/bin/wget -O /tmp/dual_models/libtraffic_models.so http://192.168.1.100:8000/libtraffic_models.so
/userfs/bin/wget -O /tmp/dual_models/libtiny_gru.so http://192.168.1.100:8000/libtiny_gru.so
chmod 755 /tmp/dual_models/app_arm64
cd /tmp/dual_models
./app_arm64 rf
./app_arm64 gru
./app_arm64 both
```
Không đối chiếu accuracy giữa hai kết quả mẫu: hai nhánh đang dùng input mẫu
khác nhau. GRU dùng packet giả trong `run_gru`, chuẩn hóa đúng một lần bằng
`traffic_gru_prepare`. Người dùng thay phần điền mảng bằng feature extractor.
`./app_arm64` mặc định chạy RF. API thư viện có thể gọi GRU nhiều lần sau một
lần init; deinit khi kết thúc. Các lời gọi GRU hiện cần tuần tự.

Có thể kiểm thử GRU bằng file 270 float32 đã chuẩn hóa, packet-major:
`./app_arm64 gru input.raw`. File phải đúng 1080 byte, không có header/text.
API `traffic_gru_predict` nhận thẳng con trỏ array, không yêu cầu file.

## Kết quả build hiện tại

- Model: 614,692 byte; libtiny_gru.so: 2,211,224 byte.
- libtraffic_models.so: 608,240 byte; app_arm64: 10,408 byte.
- Ba artifact tổng 2,829,872 byte (khoảng 2.70 MiB).
- App cùng hai thư viện đã chạy QEMU ARM64. Cần lấy log iGate mới để ghi
  latency board. Không sử dụng latency QEMU làm hiệu năng iGate.
- RF sample trả Background; GRU synthetic sample trả RTVideo. Đây chỉ là
  smoke test, chưa xác nhận accuracy hay parity với model PyTorch gốc.

Đối chiếu ARM64 với host TFLite trên ba tensor test (môi trường test cần TF/numpy):
```sh
python traffic_models/tests/verify_gru.py \
  --model ~/model_test/tiny_gru/tiny_gru_float32.tflite \
  --app dist/dual_models/app_arm64 \
  --sysroot "$ARM64_TOOLCHAIN/aarch64-none-linux-gnu/libc"
```
