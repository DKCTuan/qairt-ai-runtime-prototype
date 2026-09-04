# Thiết lập và chạy AI Runtime Tool từ đầu đến ARM64 board

Tài liệu này là checklist thao tác để dựng môi trường WSL, dùng cả hai hướng
QNN Native và TFLite + QNN Delegate, sau đó tạo executable TFLite CPU static
và chạy trên Orange Pi/iGate. Các lệnh chạy trên WSL trừ khi có ghi rõ
PowerShell, Orange Pi hoặc iGate.

## 1. Phạm vi kết quả

Đầu vào của nhánh TFLite là `.tflite`. Kết quả cuối của mode static là một
executable ARM64 gồm application C, AI Runtime, TFLite CPU runtime và model.
Board chỉ nhận executable cùng dữ liệu input thay đổi theo lần suy luận.

```text
model.tflite -> test (ARM64 static) -> Orange Pi / iGate
```

Hai nhánh Qualcomm cũng được giữ trong project:

```text
QNN Native:        model -> DLC -> Context Binary -> QNN C API
TFLite Delegate:   model.tflite -> TFLite -> QNN External Delegate hoặc CPU fallback
```

## 2. Cài WSL và package hệ thống

Mở PowerShell Windows và kiểm tra WSL:

```powershell
wsl -l -v
```

Mở Ubuntu WSL. Môi trường đã dùng là Ubuntu 24.04.4:

```bash
cat /etc/os-release
```

Cài package chung cho QAIRT/QNN, TensorFlow/Bazel, cross-build ARM64, QEMU và
truyền file tới board:

```bash
sudo apt update

sudo apt install -y \
  wget curl unzip zip git \
  build-essential software-properties-common \
  openjdk-11-jdk \
  libgl1 clang libc++-dev libc++abi-dev \
  flatbuffers-compiler libflatbuffers-dev rename pkg-config \
  qemu-user \
  gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
  binutils-aarch64-linux-gnu \
  openssh-client inetutils-telnet
```

Không cần `apt upgrade` toàn hệ thống để làm project.

## 3. Cài Python 3.10 cho QAIRT

Ubuntu 24.04 mặc định dùng Python 3.12. SDK QAIRT đang dùng được kiểm tra với
Python 3.10, do đó thêm PPA và cài Python 3.10:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

sudo apt install -y python3.10 python3.10-venv python3.10-dev

curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

python3.10 --version
python3.10 -m pip --version
```

Kết quả cần có Python `3.10.x`.

## 4. Lấy source project

Clone project tại Linux filesystem:

```bash
git clone https://github.com/DKCTuan/qairt-ai-runtime-prototype.git ~/qairt_sdk
cd ~/qairt_sdk
git log -1 --oneline
```

## 5. Tải và setup QAIRT SDK

Đăng nhập Qualcomm Package Manager, tải Qualcomm AI Runtime SDK Linux Community
Edition. Ví dụ bản đang dùng là `2.44.0.260225`. Copy file ZIP từ Windows vào
Linux filesystem rồi giải nén vào source project; không build trực tiếp dưới
`/mnt/c`.

```bash
cp "/mnt/c/Users/<Windows-user>/Downloads/v2.44.0.260225.zip" \
  ~/qairt_sdk_download.zip

unzip ~/qairt_sdk_download.zip -d ~/qairt_sdk

export QNN_SDK_ROOT="$HOME/qairt_sdk/qairt/2.44.0.260225"
echo 'export QNN_SDK_ROOT="$HOME/qairt_sdk/qairt/2.44.0.260225"' >> ~/.bashrc

ls "$QNN_SDK_ROOT"
```

Tạo virtual environment của SDK:

```bash
cd "$QNN_SDK_ROOT"
python3.10 -m venv qairt_env --without-pip
source qairt_env/bin/activate
python3 -m ensurepip --upgrade
```

Kiểm tra dependency SDK:

```bash
cd "$QNN_SDK_ROOT"
bash bin/check-linux-dependency.sh
```

Nếu Ubuntu 24.04 báo thiếu `libtinfo5` hoặc `libncurses5`, cài cặp package
legacy cùng version:

```bash
cd ~
wget http://ftp.us.debian.org/debian/pool/main/n/ncurses/libtinfo5_6.4-4_amd64.deb
wget http://ftp.us.debian.org/debian/pool/main/n/ncurses/libncurses5_6.4-4_amd64.deb
sudo dpkg -i libtinfo5_6.4-4_amd64.deb libncurses5_6.4-4_amd64.deb
```

Với một số SDK cũ, `bin/check-python-dependency` chưa nhận Ubuntu 24.04.
Kiểm tra file trước; chỉ patch nếu chưa có key `24.04`:

```bash
grep -n '24.04' "$QNN_SDK_ROOT/bin/check-python-dependency" || \
sed -i 's/"22.04": version310.version,/"22.04": version310.version,\n  "24.04": version310.version,/' \
  "$QNN_SDK_ROOT/bin/check-python-dependency"

source "$QNN_SDK_ROOT/qairt_env/bin/activate"
python3 "$QNN_SDK_ROOT/bin/check-python-dependency"
```

Mỗi terminal dùng QNN cần:

```bash
source "$QNN_SDK_ROOT/qairt_env/bin/activate"
source "$QNN_SDK_ROOT/bin/envsetup.sh"
```

Build sample để kiểm tra SDK:

```bash
cd "$QNN_SDK_ROOT/examples/QNN/SampleApp/SampleApp"
make all_x86
./bin/x86_64-linux-clang/qnn-sample-app --help
```

Nếu project đã tồn tại, thay `git clone` ở bước 4 bằng `git pull`. Sau khi giải
nén, SDK phải nằm ở:

```text
~/qairt_sdk/qairt/2.44.0.260225
```

Chuẩn bị model và input test:

```text
~/model_test/traffic_qos_model.onnx
~/model_test/tflite_output/traffic_qos_model_float32.tflite
~/model_test/input_seq.raw
```

`input_seq.raw` có 240 giá trị `float32` đã chuẩn hóa, theo thứ tự logic
`[80 packet][3 kênh]`, nên có kích thước 960 byte. Đây là input ngẫu nhiên
dùng để kiểm tra runtime, không phải một flow traffic thu thật.

## 6. Hướng A — QNN Native

Kích hoạt QAIRT session:

```bash
export QNN_SDK_ROOT="$HOME/qairt_sdk/qairt/2.44.0.260225"
source "$QNN_SDK_ROOT/qairt_env/bin/activate"
source "$QNN_SDK_ROOT/bin/envsetup.sh"
cd ~/qairt_sdk/ai_runtime_prototype
```

Convert ONNX sang DLC:

```bash
python3 tools/model_deploy.py convert \
  ~/model_test/traffic_qos_model.onnx
```

Tạo DLC và Context Binary trong một lệnh:

```bash
python3 tools/model_deploy.py prepare \
  ~/model_test/traffic_qos_model.onnx \
  --backend cpu
```

Chạy QNN C API trên host với Context Binary sinh ra:

```bash
python3 tools/model_deploy.py run-api \
  --context /tmp/model_deploy_tool/traffic_qos_model/model.bin \
  --input-raw ~/model_test/input_seq.raw \
  --backend cpu
```

Hướng này không chạy trên Orange Pi hoặc iGate vì hai board không có runtime
QNN Qualcomm. Khi deploy QNN/HTP lên board Qualcomm, context binary/app cần
runtime do BSP cung cấp: `libQnn*.so`, HTP Stub và HTP Skel đúng version chip.

## 7. Hướng B — TFLite Runtime + QNN External Delegate

Hướng này giữ artifact là `.tflite`. Application chỉ gọi `ai_runtime.h`; backend
khởi tạo TensorFlow Lite C API và thử QNN External Delegate.

```text
Application -> AI Runtime -> TFLite Interpreter -> QNN Delegate / TFLite CPU
                                         -> model.tflite
```

Build hai shared library cần cho mode dynamic:

```bash
mkdir -p ~/bin
wget -O ~/bin/bazelisk \
  https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x ~/bin/bazelisk

git clone --branch v2.15.0 --depth 1 \
  https://github.com/tensorflow/tensorflow.git ~/tensorflow

cd ~/tensorflow
./configure

~/bin/bazelisk build -c opt \
  //tensorflow/lite/c:libtensorflowlite_c.so \
  //tensorflow/lite/delegates/external:external_delegate
```

Copy library vào project theo kiến trúc host:

```bash
mkdir -p ~/qairt_sdk/tflite_qnn_prototype/third_party/lib-x86_64

cp ~/tensorflow/bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so \
   ~/tensorflow/bazel-bin/tensorflow/lite/delegates/external/libexternal_delegate.so \
   ~/qairt_sdk/tflite_qnn_prototype/third_party/lib-x86_64/
```

Build application TFLite dynamic:

```bash
cd ~/qairt_sdk/tflite_qnn_prototype
make ARCH=x86_64 TFLITE_ROOT=third_party
```

CPU fallback là mode đã kiểm chứng. Nó chủ động bỏ qua Delegate:

```bash
export DL_DISABLE_QNN_DELEGATE=1
export DL_TFLITE_MODEL_PATH=~/model_test/tflite_output/traffic_qos_model_float32.tflite
export DL_INPUT_RAW=~/model_test/input_seq.raw
LD_LIBRARY_PATH=third_party/lib-x86_64 ./build/tflite_delegate-x86_64/traffic_app
```

Để thử QNN Delegate, không đặt `DL_DISABLE_QNN_DELEGATE`; phải cung cấp đúng
`DL_QNN_DELEGATE_LIB`, `DL_QNN_BACKEND_LIB`, `DL_QNN_BACKEND_TYPE` và, với HTP,
`DL_QNN_SKEL_DIR`. Có thể bật strict mode để cấm fallback âm thầm:

```bash
export DL_QNN_STRICT_DELEGATE=1
```

Delegate QNN/HTP chỉ được coi là pass khi log báo mode `qnn_delegate` trên board
Qualcomm thật. CPU fallback pass không phải bằng chứng HTP/NPU đang chạy.

## 8. Hướng C — đóng gói TFLite CPU static

Hướng này sử dụng cùng TFLite backend nhưng build profile
`DL_TFLITE_STATIC_CPU`: Delegate bị bỏ qua, model được nhúng vào executable và
TensorFlow Lite C API được link tĩnh. Nó dành cho CPU ARM64 Linux portable.

TensorFlow phải ở đúng tag `v2.15.0`; Bazelisk tự đọc `.bazelversion` (6.1.0).
Nếu đã clone TensorFlow ở bước 7 thì dùng lại. Nếu chưa có, thực hiện lệnh clone
và `./configure` ở bước 7.

Build executable classifier nhận input khi chạy:

```bash
cd ~/qairt_sdk

python3 ai_runtime_prototype/tools/model_deploy.py standalone-tflite \
  --model ~/model_test/tflite_output/traffic_qos_model_float32.tflite \
  --app-source tflite_qnn_prototype/examples/embedded_traffic_classifier/main.c \
  --output dist/test \
  --tensorflow-root ~/tensorflow
```

Kết quả cần có `"status": "success"`. Kiểm tra artifact:

```bash
file dist/test
readelf -d dist/test
```

Kết quả yêu cầu: `ARM aarch64`, `statically linked`, và không có dynamic
section. Đưa một mẫu vào input inbox rồi test ABI/logic trên host:

```bash
cp ~/model_test/input_seq.raw /tmp/traffic_input.raw.tmp
mv /tmp/traffic_input.raw.tmp /tmp/traffic_input.raw
timeout --signal=TERM 2s qemu-aarch64 dist/test
```

Kết quả classifier traffic đã kiểm chứng là `label=2`, tương ứng `RTVideo`
theo label mapping trong checkpoint. Đây chỉ là smoke test bằng input ngẫu
nhiên; nó không chứng minh đã nhận diện một phiên RTVideo thực tế. Latency
QEMU không dùng để đánh giá board thật.

## 9. Chép và chạy static binary trên Orange Pi

Orange Pi thử nghiệm có IP `192.168.239.119`. Từ WSL:

```bash
cd ~/qairt_sdk
scp dist/test root@192.168.239.119:/root/
scp ~/model_test/input_seq.raw root@192.168.239.119:/tmp/traffic_input.raw
```

SSH vào board:

```bash
ssh root@192.168.239.119
```

Trên Orange Pi:

```sh
chmod +x /root/test
/root/test
```

Không chép `.tflite`, `.so`, `ld-linux` hoặc `libc` cho mode static.

## 10. Chép và chạy static binary trên iGate

iGate có Linux Buildroot ARM64 tối giản, Telnet và `/userfs/bin/wget`; root
filesystem read-only, còn `/tmp` writable. Không cài package lên iGate.

Trên WSL, tạo payload và HTTP server tạm:

```bash
mkdir -p /tmp/igate_static_payload
cp ~/qairt_sdk/dist/test ~/model_test/input_seq.raw \
  /tmp/igate_static_payload/

cd /tmp/igate_static_payload
python3 -m http.server 8000 --bind 0.0.0.0
```

Trong thử nghiệm, iGate nhìn thấy card Ethernet Windows tại `192.168.1.100`.
Nếu WSL chạy NAT, cấu hình Windows port proxy/firewall để chuyển `192.168.1.100:8000`
vào HTTP server WSL trước khi tiếp tục.

Mở terminal khác và Telnet:

```bash
telnet 192.168.1.1 23
```

Trên iGate:

```sh
/userfs/bin/wget -O /tmp/test \
  http://192.168.1.100:8000/test

/userfs/bin/wget -O /tmp/traffic_input.raw.tmp \
  http://192.168.1.100:8000/input_seq.raw

mv /tmp/traffic_input.raw.tmp /tmp/traffic_input.raw
chmod +x /tmp/test
/tmp/test
```

`wget` chỉ dùng để chép file. Inference chạy cục bộ trong executable; không gọi
web API. File trong `/tmp` bị mất sau reboot, không làm thay đổi firmware.

## 11. Application static với input động

File application dùng cho luồng này là:

```text
tflite_qnn_prototype/examples/embedded_traffic_classifier/main.c
```

Tên thư mục cho biết model/runtime được nhúng vào executable; **input không
được nhúng**. `main.c` chạy không cần đối số, giữ model trong bộ nhớ và chờ
các mẫu mới tại `/tmp/traffic_input.raw`.

Notebook `wifi-qos (6)_fixed.ipynb` cho thấy model này phân loại năm lớp:
`Background`, `Game`, `RTVideo`, `VStream`, `Voice`. Label index lần lượt là
`0..4` theo đúng thứ tự trên.

Mỗi mẫu không phải 240 đặc trưng độc lập. Nó là một cửa sổ 80 packet, mỗi
packet có ba kênh:

- `log1p(packet length)`;
- `log1p(inter-arrival time)`;
- `direction` (`+1` hoặc `-1`).

Ba kênh phải được chuẩn hóa bằng thống kê của checkpoint trước khi đưa vào
executable: `x_mean = [6.02560806, 0.00192464957, -0.167983967]` và
`x_std = [1.36892608, 0.02101696, 0.98579073]`. Mỗi mẫu vẫn gồm 240 giá trị
`float32`, tương đương 960 byte, nhưng thứ tự bên ngoài là `[80][3]`:

```text
[packet0_ch0, packet0_ch1, packet0_ch2, ..., packet79_ch2]
```

File TFLite hiện tại công bố input `[1,3,80]`, trong khi ONNX gốc là
`[1,80,3]`. `main.c` chuyển layout `[80,3]` sang `[3,80]` trước khi gọi AI
Runtime. Nếu bỏ bước này, byte size vẫn đúng nhưng các giá trị nằm sai vị trí
và score không còn khớp output ONNX tham chiếu.

Chương trình thực hiện vòng đời sau:

```text
dl_init() một lần
  -> chờ input inbox /tmp/traffic_input.raw
  -> đọc một mẫu [80][3] đã chuẩn hóa vào float input[240]
  -> xóa file đã nhận để chờ mẫu tiếp theo
  -> đổi layout sang [3][80] của TFLite hiện tại
  -> dl_inference_ex(input, &result)
  -> in scores, label và latency
  -> tiếp tục đọc mẫu kế tiếp
dl_deinit() khi nhận SIGINT/SIGTERM hoặc chương trình kết thúc
```

Build executable ARM64 static có model nhúng nhưng input động:

```bash
cd ~/qairt_sdk

python3 ai_runtime_prototype/tools/model_deploy.py standalone-tflite \
  --model ~/model_test/tflite_output/traffic_qos_model_float32.tflite \
  --app-source tflite_qnn_prototype/examples/embedded_traffic_classifier/main.c \
  --output dist/test \
  --tensorflow-root ~/tensorflow \
  --force
```

Đưa thủ công một mẫu vào inbox bằng cách ghi file tạm rồi đổi tên:

```bash
cp ~/model_test/input_seq.raw /tmp/traffic_input.raw.tmp
mv /tmp/traffic_input.raw.tmp /tmp/traffic_input.raw
./dist/test
```

Code bắt gói/tiền xử lý có thể tự động xuất từng mảng `float32[240]` theo
cùng giao thức:

```c
FILE *fp = fopen("/tmp/traffic_input.raw.tmp", "wb");
fwrite(input, sizeof(float), 240, fp);
fclose(fp);
rename("/tmp/traffic_input.raw.tmp", "/tmp/traffic_input.raw");
```

Producer phải ghi đủ 960 byte vào file tạm và chỉ publish khi inbox cũ đã được
tiêu thụ; nếu ghi đè một inbox chưa xử lý thì có thể làm mất mẫu. Notebook hiện
đọc từng CSV session,
bỏ 10 packet đầu rồi chia các cửa sổ 80 packet không chồng lấn; nó chưa cài
đặt bộ thu packet trực tiếp hoặc bảng flow theo 5-tuple. Thành phần đó phải
được tích hợp sau và phải tái tạo đúng ba kênh cùng phép chuẩn hóa ở trên.

Trên iGate, chép `test` và input thử vào `/tmp`, sau đó chạy:

```sh
chmod +x /tmp/test
/tmp/test
```

Với input kiểm thử hiện tại, output tham chiếu đã đối chiếu giữa ONNX, QNN CPU,
TFLite CPU host và bản ARM64 qua QEMU là năm score, trong đó score tại index 2
lớn nhất:

```text
sample=0
input_feature_count=240
output_score_count=5
scores=0.000443353347,0.00221594353,0.997327805,0.0000124046483,0.000000426769219
label=2
class=RTVideo
latency_ms=<đo lại trên iGate sau khi chép bản đã sửa layout>
```

Score trên khớp output ONNX tham chiếu với sai số float rất nhỏ. Tuy nhiên,
vì `input_seq.raw` là dữ liệu ngẫu nhiên trong không gian đã chuẩn hóa nên chỉ
được dùng để kiểm tra tính đúng của pipeline; không được ghi trong báo cáo là
model đã nhận diện một luồng RTVideo thu từ mạng thật. Kết quả iGate cũ dùng
input chưa đổi layout nên cần chạy lại, không dùng score/latency cũ để báo cáo.

## 12. Giới hạn cần ghi rõ

- `test` hiện là executable ARM64 Linux; muốn x86, ARM32 hoặc
  RISC-V cần build artifact theo target tương ứng.
- Application mẫu dùng `dl_inference(float *)`, phù hợp classifier một input
  float32. Model quantized hoặc multi-input/output dùng `dl_runtime_*` API.
- Preprocess (đọc sensor, resize, normalize) và postprocess (class name, NMS)
  thuộc application, không tự sinh chung được cho mọi model.
- Riêng model traffic hiện tại, phép chuẩn hóa chưa được nhúng vào graph
  TFLite; nguồn input phải áp dụng đúng `log1p`, `x_mean` và `x_std`.
- TFLite CPU static portable nhưng không dùng NPU. QNN Delegate/HTP nhanh hơn
  khi hỗ trợ, nhưng phụ thuộc thư viện/driver/BSP Qualcomm đúng target.
