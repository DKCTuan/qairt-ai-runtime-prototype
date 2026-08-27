# 18. Tool tong the: Model (.onnx/.tflite/...) -> chay tren embedded device qua API

## Muc tieu

Dua vao 1 model da train xong, bat ke framework nao xuat ra (.onnx tu
PyTorch/sklearn, .tflite tu TensorFlow...), co 1 tool tu dong hoa toan bo
buoc trung gian, va ung dung nhung chi can goi qua 1 API don gian de lay
ket qua inference - khong can biet chi tiet ben trong QNN/QAIRT lam gi.

```text
model.onnx / model.tflite / model.dlc
        |
        v
   [ tool convert + chuan bi ]   <- Qualcomm QAIRT SDK
        |
        v
   application goi qua API:
        dl_init()
        dl_inference_ex(input, &result)
        dl_deinit()
        |
        v
   label + scores + latency_ms
```

Application (vi du chay tren embedded device) khong bao gio goi truc tiep
converter hay CLI tool cua Qualcomm. Application chi include 1 file header
(`ai_runtime.h`) va link voi 1 static library (`libai_runtime.a`).

## Trang thai hien tai (da kiem chung, chua kiem chung)

Da kiem chung tren host (may Ubuntu/WSL2, kien truc x86_64), CPU backend:

```text
- Da convert + chay dung 2 model rat khac nhau:
    traffic_qos_model (classifier nho, 5 lop output)
    best.onnx (YOLO, ~1.23M tham so, ~151.200 so output)
- Runtime tu doc dung shape input/output tu model, khong hardcode
- Do duoc latency inference (rieng phan chay model, khong tinh overhead)
```

Chua kiem chung:

```text
- Chua chay tren thiet bi Qualcomm that (Networking Pro/IPQ Wi-Fi
  router - day la thiet bi dich cuoi cung)
- Chua build bang dung toolchain cross-compile cua thiet bi do (moi
  chi thu voi toolchain aarch64 generic cua Ubuntu, ra dung kien truc
  ARM64 nhung co the khong khop libc cua firmware that - xem Buoc 6)
- Chua thu backend HTP (NPU) - moi dung CPU backend
```

## Cong cu can co truoc khi bat dau

```text
Qualcomm AI Runtime SDK (QAIRT) - da cai va verify o cac phan truoc
Python 3.10 + venv rieng cho SDK (qairt_env)
clang, cac lib he thong SDK yeu cau (xem Phan 1-9)
```

---

## Buoc 1: Xac dinh model dau vao co dinh dang gi

Tool convert ho tro 3 dinh dang input:

```text
.onnx     -> converter: qairt-converter (hoac qnn-onnx-converter)
.tflite   -> converter: qairt-converter (hoac qnn-tflite-converter)
.dlc      -> khong can convert, day da la dinh dang QNN roi (copy thang)
```

Neu model dang o dinh dang khac (.h5, checkpoint PyTorch...), phai tu
export sang 1 trong 3 dinh dang tren truoc (ngoai pham vi tool nay).

## Buoc 2: Convert model sang .dlc

```bash
python3 tools/model_deploy.py convert \
  --model /path/to/model.onnx \
  --output /path/to/output/model.dlc
```

- Neu model co nhieu input hoac converter khong tu doan duoc shape, them:
  `--source-model-input-shape <ten_input> <dims>` (vi du `'input' '1,224,224,3'`).
- Output la file `.dlc` - day la dinh dang trung gian, CHUA phai thu chay
  duoc tren device.

## Buoc 3: Build context binary (.bin) tu .dlc

```bash
python3 tools/model_deploy.py build-context \
  --dlc /path/to/model.dlc \
  --output /path/to/model.bin \
  --backend cpu
```

- Day la buoc "bake" model + backend cu the (cpu/gpu/htp) thanh 1 file
  nhi phan duy nhat (`.bin`). File nay moi la thu duoc dong goi cung
  firmware khi deploy len thiet bi that - luc do device chi can
  `contextCreateFromBinary()`, khong can parse lai `.dlc`.
- `--backend cpu` dung cho test tren host. Khi co device that va toolchain
  dung, doi thanh `--backend htp` de chay tren NPU.

## Buoc 4: Build runtime + chay thu qua API (tren host)

```bash
python3 tools/model_deploy.py run-api \
  --context /path/to/model.bin \
  --input-raw /path/to/input.raw \
  --backend cpu
```

- Lenh nay tu build `libai_runtime.a` + app vi du (`examples/main.c`) voi
  `BACKEND=qnn_api` (goi truc tiep QNN C API, khong qua CLI tool), roi
  chay thu voi 1 file input mau (`.raw`, float32 lien tuc).
- Output JSON co `label`, `scores`, `latency_ms`.

## Buoc 5: Chay ca chuoi bang 1 lenh (khuyen nghi dung hang ngay)

```bash
python3 tools/model_deploy.py deploy \
  --model /path/to/model.onnx \
  --input-raw /path/to/input.raw \
  --backend cpu \
  --reference /path/to/reference_output.npy \
  --tolerance 1e-5
```

- Gop ca Buoc 2-4 thanh 1 lenh. Neu co `--reference` (output tham chieu
  tu framework goc, vi du chay bang ONNX Runtime), tool tu so sanh va tra
  `validation_passed: true/false` - day la cach xac nhan QNN chay ra ket
  qua giong het framework goc, khong chi la "chay khong loi".

## Buoc 6: Dua sang thiet bi that (CHUA lam xong, ghi lai de biet con thieu gi)

Day la khoang cach con lai giua "chay tren host" va "chay tren embedded
device":

```text
1. Can dung toolchain cross-compile cua CHINH thiet bi dich (khong phai
   toolchain generic nhu aarch64-linux-gnu-gcc cua Ubuntu) - vi router
   Wi-Fi dua tren OpenWrt thuong dung musl libc, khac glibc cua toolchain
   generic. Toolchain dung phai lay tu SDK di kem firmware cua thiet bi.
2. Build lai model_deploy.py build-context / run-api voi --target tro
   dung vao thu muc bin/lib cua kien truc device trong QAIRT SDK (vi du
   mot thu muc kieu aarch64-oe-linux-gcc11.2, tuy SDK co ho tro dung
   target cho dong Networking Pro/IPQ hay khong - CAN KIEM TRA, chua xac
   nhan).
3. Neu dung backend HTP (NPU that): can them buoc quantization/calibration
   model (INT8) - HTP thuong chi tang toc ro ret voi model da quantize,
   khong phai float32 nhu dang test tren CPU host.
4. Copy context binary (.bin) + libai_runtime.a + app len device that
   (qua SCP/firmware image), chay thu.
```

## Tom tat file lien quan

```text
tools/model_deploy.py       CLI chinh: convert/build-context/run-api/deploy
tools/regression_test.py    Tu dong test lai nhieu model, assert khong hong
include/ai_runtime.h        API public: dl_init/dl_get_io_count/dl_inference_ex/dl_deinit
src/ai_runtime.c            Logic chung, khong phu thuoc QNN
src/backend_qnn_api.c       Backend that, goi QNN C API truc tiep (duong chinh)
examples/main.c             Vi du application dung API
```