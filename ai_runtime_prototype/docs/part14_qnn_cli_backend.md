# 14. Prototype end-to-end voi QNN backend

Sau khi API `dl_init()`, `dl_inference()` va `dl_deinit()` da on dinh, buoc tiep theo la thay mock backend bang backend goi QAIRT/QNN that.

Trong prototype nay, backend `qnn_cli` thuc hien cac viec sau ben trong `dl_inference()`:

```text
Nhan mang float input tu application
Ghi input thanh file raw
Tao input_list.txt dung format cua qnn-net-run
Goi qnn-net-run voi QNN CPU backend
Doc output class_probs.raw
Tinh argmax
Tra label ve application
```

Application khong can biet ben duoi co `qnn-net-run`, `libQnnCpu.so`, `libQnnModelDlc.so` hay file `.dlc`.

Cach build mock backend:

```bash
make clean
make BACKEND=mock run
```

Cach build QNN backend:

```bash
make clean
make BACKEND=qnn_cli run
```

Cac bien moi truong co the override:

```text
DL_QAIRT_ROOT=/home/congtuan/qairt_sdk/qairt/2.44.0.260225
DL_MODEL_PATH=/home/congtuan/model_test/traffic_qos_model.dlc
DL_WORK_DIR=/tmp/ai_runtime_qnn
DL_INPUT_NAME=input_seq
DL_OUTPUT_NAME=class_probs
DL_INPUT_RAW=/home/congtuan/model_test/input_seq.raw
```

Ket qua mong doi voi model da validate:

```text
label=2
scores=0.000443353,0.002215946,0.9973278,0.0000124046,0.0000004268
```

Day la ban prototype quan trong vi no chuyen luong chay tu command line thu cong sang API runtime ma application co the goi truc tiep.
