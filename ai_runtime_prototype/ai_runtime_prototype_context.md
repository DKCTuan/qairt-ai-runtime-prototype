# ai_runtime_prototype context

Generated source and notes bundle. Binary files and build output are excluded.

## `.gitignore`

```text
build/
__pycache__/
*.pyc
```

## `Makefile`

```makefile
CC ?= gcc
AR ?= ar
BACKEND ?= mock

CFLAGS ?= -O2 -Wall -Wextra -Iinclude -Isrc

BUILD_DIR := build/$(BACKEND)
LIB := $(BUILD_DIR)/libai_runtime.a
APP := $(BUILD_DIR)/traffic_app

DL_QAIRT_ROOT ?= /home/congtuan/qairt_sdk/qairt/2.44.0.260225

ifeq ($(BACKEND),qnn_cli)
BACKEND_SRC := src/backend_qnn_cli.c
else ifeq ($(BACKEND),qnn_api)
BACKEND_SRC := src/backend_qnn_api.c
CFLAGS += -I$(DL_QAIRT_ROOT)/include/QNN
LDLIBS += -ldl
else
BACKEND_SRC := src/backend_mock.c
endif

RUNTIME_OBJS := \
	$(BUILD_DIR)/ai_runtime.o \
	$(BUILD_DIR)/backend.o

.PHONY: all clean run

all: $(APP)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/ai_runtime.o: src/ai_runtime.c include/ai_runtime.h src/backend.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/backend.o: $(BACKEND_SRC) src/backend.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(RUNTIME_OBJS)
	$(AR) rcs $@ $^

$(APP): examples/main.c $(LIB)
	$(CC) $(CFLAGS) $< $(LIB) -o $@ $(LDLIBS)

run: $(APP)
	./$(APP)

clean:
	rm -rf build
```

## `README.md`

```markdown
# AI Runtime Prototype for Qualcomm QAIRT/QNN

This prototype proves the target application API:

```c
dl_init();
int label = dl_inference(input);
dl_deinit();
```

The application does not call QAIRT/QNN directly. It links `libai_runtime.a`; the runtime backend handles model execution.

## Build mock backend

```bash
make clean
make BACKEND=mock run
```

## Build QAIRT/QNN host backend

```bash
./scripts/run_host_qnn.sh
```

Default configuration:

```text
DL_QAIRT_ROOT=/home/congtuan/qairt_sdk/qairt/2.44.0.260225
DL_MODEL_PATH=/home/congtuan/model_test/traffic_qos_model.dlc
DL_INPUT_RAW=/home/congtuan/model_test/input_seq.raw
DL_WORK_DIR=/tmp/ai_runtime_qnn
DL_INPUT_NAME=input_seq
DL_OUTPUT_NAME=class_probs
```

Expected result:

```text
label=2
scores=0.000443353143,0.00221594586,0.997327805,1.24046474e-05,4.26769219e-07
```

Current backend flow:

```text
Application
  -> ai_runtime.h
  -> libai_runtime.a
  -> backend_qnn_cli.c
  -> qnn-net-run
  -> traffic_qos_model.dlc
  -> class_probs.raw
  -> argmax label
```

Next production step: replace the CLI backend with direct QNN C API calls for embedded deployment.
```

## `docs/part12_runtime_api.md`

```markdown
# 12. Xac dinh muc tieu runtime API tren thiet bi nhung Qualcomm

Muc tieu cua he thong khong chi la convert model bang command line, ma la tao ra mot lop runtime API de application co the goi inference mot cach don gian tren thiet bi nhung Qualcomm.

O tang application, developer khong can biet model ben duoi la `.onnx`, `.tflite`, `.dlc` hay context binary `.bin`. Application chi can goi cac ham API chung:

```c
int main(void)
{
    dl_init();

    float input[FEATURE_NUM];

    /* Prepare input feature here */

    int label = dl_inference(input);

    printf("%d\n", label);

    dl_deinit();

    return 0;
}
```

Kien truc muc tieu:

```text
Application
    -> AI Runtime API
    -> Qualcomm Backend Adapter
    -> QAIRT/QNN Runtime
    -> Model Artifact (.dlc / .bin)
    -> Qualcomm Hardware (CPU / GPU / HTP)
```

Trong giai doan hien tai, ta da kiem chung duoc phan nen tang: model `traffic_qos_model.dlc` co the chay bang `qnn-net-run` voi QNN CPU backend tren host va output khop voi ONNX reference.

Ket qua da kiem chung:

```text
qnn shape: (5,)
ref shape: (5,)
max abs diff: 1.3969839e-09
mean abs diff: 2.7956162e-10
qnn argmax: 2
ref argmax: 2
```

Dieu nay xac nhan rang model sau convert co the duoc dung lam backend artifact cho lop AI Runtime API.

# 13. Prototype API o tang application

Truoc khi tich hop truc tiep QNN API, ta tao mot prototype runtime don gian voi API on dinh:

```c
int dl_init(void);
int dl_inference(const float *input);
void dl_deinit(void);
```

Ban dau, backend la mock backend de kiem tra flow application. Sau do mock backend se duoc thay bang Qualcomm QNN backend.

Luong thuc thi:

```text
main()
    -> dl_init()
        -> backend_init()
    -> dl_inference(input)
        -> backend_execute(input, scores)
        -> argmax(scores)
        -> return label
    -> dl_deinit()
        -> backend_deinit()
```
```

## `docs/part14_qnn_cli_backend.md`

```markdown
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
```

## `docs/part15_end_to_end_flow.md`

```markdown
# 15. Flow da hoan thien trong prototype

Prototype hien tai da di duoc mot vong end-to-end tren host:

```text
Application C
    -> dl_init()
    -> dl_inference(input)
    -> AI Runtime
    -> QNN CLI backend
    -> qnn-net-run
    -> traffic_qos_model.dlc
    -> class_probs.raw
    -> argmax
    -> label
```

Application khong goi truc tiep `qnn-net-run`. Application chi include `ai_runtime.h` va link voi `libai_runtime.a`.

File chinh:

```text
include/ai_runtime.h        API public cho application
src/ai_runtime.c            Quan ly init, inference, deinit va argmax
src/backend.h               Interface backend noi bo
src/backend_mock.c          Backend gia lap de test API nhanh
src/backend_qnn_cli.c       Backend goi QAIRT/QNN that qua qnn-net-run
examples/main.c             Application vi du
Makefile                    Build static library va example app
scripts/run_host_qnn.sh     Chay flow QNN host bang mot lenh
```

Lenh chay:

```bash
cd /home/congtuan/qairt_sdk/ai_runtime_prototype
./scripts/run_host_qnn.sh
```

Ket qua:

```text
label=2
scores=0.000443353143,0.00221594586,0.997327805,1.24046474e-05,4.26769219e-07
```

Y nghia cua ket qua nay: application da goi API runtime thanh cong, runtime da chay model Qualcomm `.dlc` bang QAIRT/QNN, doc output va tra ve label cho application.

Day la cau truc gan voi muc tieu tren thiet bi nhung Qualcomm. Tuy nhien, `backend_qnn_cli.c` van dang dung command-line tool `qnn-net-run`. Khi dua sang production tren device, backend nay nen duoc thay bang `backend_qnn_api.c`, goi truc tiep QNN C API de tranh phu thuoc vao command-line tool.
```

## `docs/part16_model_deploy_cli.md`

```markdown
# 16. Them tool CLI quan ly convert/validate/run-host

Sau khi runtime API da chay duoc model `.dlc` thong qua QNN backend, buoc tiep theo la tao tool CLI de tu dong hoa cac buoc thu cong.

Tool moi nam tai:

```text
tools/model_deploy.py
```

Cac lenh ho tro:

```text
convert    Convert .onnx/.tflite/.dlc sang artifact .dlc
run-host   Chay .dlc tren host bang QNN CPU backend
validate   Chay .dlc tren host va so sanh voi reference output
```

Vi du validate model hien tai:

```bash
cd /home/congtuan/qairt_sdk/ai_runtime_prototype
./tools/model_deploy.py validate \
  --dlc /home/congtuan/model_test/traffic_qos_model.dlc \
  --input-raw /home/congtuan/model_test/input_seq.raw \
  --reference /home/congtuan/model_test/onnx_output_ref.npy
```

Y nghia cua buoc nay: pipeline khong con phu thuoc vao viec go tung command QNN bang tay. Tool co the nhan model/input/reference, tu tao input list, goi `qnn-net-run`, doc output raw, tinh `argmax`, va tra ket qua dang JSON.

Day la nen tang de mo rong thanh REST API hoac service deployment sau nay.

# 16.1 Chuyen converter mac dinh sang QAIRT

Tool da duoc chuyen sang huong QAIRT/QNN-first. Lenh `convert` mac dinh goi:

```text
qairt-converter
```

Thay vi dung SNPE converter lam duong chinh.

Vi du:

```bash
./tools/model_deploy.py convert \
  --model /home/congtuan/model_test/traffic_qos_model.onnx \
  --output /tmp/model_deploy_tool/qairt_cli_test/traffic_qos_model.dlc
```

Ket qua da kiem chung:

```text
converter: qairt
status: success
validation_passed: true
max_abs_diff: 1.3969838619232178e-09
qnn_argmax: 2
reference_argmax: 2
```

SNPE converter van duoc giu lai nhu compatibility fallback:

```bash
./tools/model_deploy.py convert --converter snpe \
  --model model.onnx \
  --output model.dlc
```

Cac loai model co the mo rong:

```text
ONNX       Da test thuc te voi qairt-converter
TFLite     Da co path qua qairt-converter, can file mau de verify
TensorFlow QAIRT converter co frontend/option lien quan TensorFlow
PyTorch    SDK co qnn-pytorch-converter, can thiet ke rieng vi artifact co the khac DLC flow
DLC        Copy/validate/run truc tiep
```

Huong production: QAIRT/QNN la duong chinh, SNPE chi la fallback.
```

## `docs/part17_rest_api.md`

```markdown
# 17. Them REST API prototype

Sau khi co CLI `tools/model_deploy.py`, he thong duoc mo rong them REST API server toi thieu:

```text
tools/model_deploy_server.py
```

Server nay dung Python standard library, chua can FastAPI. Muc tieu la chung minh client co the goi pipeline QAIRT/QNN thong qua HTTP API.

Chay server:

```bash
cd /home/congtuan/qairt_sdk/ai_runtime_prototype
./scripts/run_api_server.sh --host 127.0.0.1 --port 8088
```

Kiem tra health:

```bash
curl http://127.0.0.1:8088/health
```

Validate model qua API:

```bash
curl -s -X POST http://127.0.0.1:8088/validate \
  -H 'Content-Type: application/json' \
  -d '{
    "dlc": "/home/congtuan/model_test/traffic_qos_model.dlc",
    "input_raw": "/home/congtuan/model_test/input_seq.raw",
    "reference": "/home/congtuan/model_test/onnx_output_ref.npy"
  }'
```

Cac endpoint hien co:

```text
GET  /health
POST /convert
POST /run-host
POST /validate
```

Y nghia: tu giai doan nay, flow khong chi chay qua application C ma con co the duoc kich hoat boi mot app ben ngoai thong qua HTTP API.
```

## `examples/main.c`

```c
#include <stdio.h>
#include <stdlib.h>

#include "ai_runtime.h"

static int load_input(const char *path, float *input, int input_count)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return -1;
    }

    size_t read_count = fread(input, sizeof(float), (size_t)input_count, fp);
    fclose(fp);

    return read_count == (size_t)input_count ? 0 : -1;
}

int main(void)
{
    const char *input_path = getenv("DL_INPUT_RAW");
    if (input_path == NULL || input_path[0] == '\0') {
        input_path = "/home/congtuan/model_test/input_seq.raw";
    }

    float input[FEATURE_NUM] = {0};
    if (load_input(input_path, input, FEATURE_NUM) != 0) {
        fprintf(stderr, "failed to load input: %s\n", input_path);
        return 1;
    }

    if (dl_init() != 0) {
        fprintf(stderr, "dl_init failed\n");
        return 1;
    }

    dl_result_t result;
    if (dl_inference_ex(input, &result) != 0) {
        fprintf(stderr, "dl_inference failed\n");
        dl_deinit();
        return 1;
    }

    printf("label=%d\n", result.label);
    printf("scores=");
    for (int i = 0; i < result.score_count; ++i) {
        printf("%s%.9g", i == 0 ? "" : ",", result.scores[i]);
    }
    printf("\n");

    dl_deinit();
    return 0;
}
```

## `include/ai_runtime.h`

```c
#ifndef AI_RUNTIME_H
#define AI_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

#define FEATURE_NUM 240
#define CLASS_NUM 5

typedef struct {
    int label;
    float scores[CLASS_NUM];
    int score_count;
} dl_result_t;

int dl_init(void);
int dl_inference(const float *input);
int dl_inference_ex(const float *input, dl_result_t *result);
void dl_deinit(void);

#ifdef __cplusplus
}
#endif

#endif
```

## `scripts/build_context_binary.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Sinh QNN context binary (.bin) tu mot .dlc da convert/validate, dung mot
# lan tren host/CI. File .bin sinh ra la thu duoc dong goi cung firmware
# khi deploy len thiet bi nhung - luc do runtime chi con contextCreateFromBinary().
#
# Usage:
#   ./scripts/build_context_binary.sh [output.bin]
#
# Bien moi truong:
#   DL_QAIRT_ROOT   thu muc goc QAIRT SDK
#   DL_MODEL_PATH   duong dan .dlc dau vao

DL_QAIRT_ROOT="${DL_QAIRT_ROOT:-/home/congtuan/qairt_sdk/qairt/2.44.0.260225}"
DL_MODEL_PATH="${DL_MODEL_PATH:-/home/congtuan/model_test/traffic_qos_model.dlc}"
OUTPUT_BIN="${1:-${DL_MODEL_PATH%.dlc}.bin}"

GEN_BIN="$DL_QAIRT_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator"
BACKEND_LIB="$DL_QAIRT_ROOT/lib/x86_64-linux-clang/libQnnCpu.so"
MODEL_LIB="$DL_QAIRT_ROOT/lib/x86_64-linux-clang/libQnnModelDlc.so"
LIB_DIR="$DL_QAIRT_ROOT/lib/x86_64-linux-clang"

echo "model      : $DL_MODEL_PATH"
echo "output     : $OUTPUT_BIN"
echo "qairt_root : $DL_QAIRT_ROOT"

LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}" \
"$GEN_BIN" \
  --backend "$BACKEND_LIB" \
  --model "$MODEL_LIB" \
  --dlc_path "$DL_MODEL_PATH" \
  --binary_file "$(basename "$OUTPUT_BIN" .bin)" \
  --output_dir "$(dirname "$OUTPUT_BIN")"

echo "done: $OUTPUT_BIN"
```

## `scripts/run_api_server.sh`

```bash
#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
exec python3 tools/model_deploy_server.py "$@"
```

## `scripts/run_host_qnn.sh`

```bash
#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

export DL_QAIRT_ROOT="${DL_QAIRT_ROOT:-/home/congtuan/qairt_sdk/qairt/2.44.0.260225}"
export DL_MODEL_PATH="${DL_MODEL_PATH:-/home/congtuan/model_test/traffic_qos_model.dlc}"
export DL_INPUT_RAW="${DL_INPUT_RAW:-/home/congtuan/model_test/input_seq.raw}"
export DL_WORK_DIR="${DL_WORK_DIR:-/tmp/ai_runtime_qnn}"
export DL_INPUT_NAME="${DL_INPUT_NAME:-input_seq}"
export DL_OUTPUT_NAME="${DL_OUTPUT_NAME:-class_probs}"

make BACKEND=qnn_cli run
```

## `src/ai_runtime.c`

```c
#include "ai_runtime.h"

#include <stddef.h>

#include "backend.h"

static int g_initialized = 0;

static int argmax(const float *values, int count)
{
    int best_index = 0;
    float best_value = values[0];

    for (int i = 1; i < count; ++i) {
        if (values[i] > best_value) {
            best_value = values[i];
            best_index = i;
        }
    }

    return best_index;
}

int dl_init(void)
{
    if (g_initialized) {
        return 0;
    }

    if (backend_init() != 0) {
        return -1;
    }

    g_initialized = 1;
    return 0;
}

int dl_inference(const float *input)
{
    dl_result_t result;

    if (dl_inference_ex(input, &result) != 0) {
        return -1;
    }

    return result.label;
}

int dl_inference_ex(const float *input, dl_result_t *result)
{
    if (!g_initialized || input == NULL || result == NULL) {
        return -1;
    }

    result->score_count = CLASS_NUM;

    if (backend_execute(input, FEATURE_NUM, result->scores, CLASS_NUM) != 0) {
        return -1;
    }

    result->label = argmax(result->scores, CLASS_NUM);
    return 0;
}

void dl_deinit(void)
{
    if (!g_initialized) {
        return;
    }

    backend_deinit();
    g_initialized = 0;
}
```

## `src/backend.h`

```c
#ifndef BACKEND_H
#define BACKEND_H

int backend_init(void);
int backend_execute(const float *input, int input_count, float *output, int output_count);
void backend_deinit(void);

#endif
```

## `src/backend_mock.c`

```c
#include "backend.h"

#include <stddef.h>

static int g_backend_ready = 0;

int backend_init(void)
{
    g_backend_ready = 1;
    return 0;
}

int backend_execute(const float *input, int input_count, float *output, int output_count)
{
    if (!g_backend_ready || input == NULL || output == NULL || input_count <= 0 || output_count < 5) {
        return -1;
    }

    /* Mock output copied from the validated QNN CPU result. */
    output[0] = 4.4335314e-04f;
    output[1] = 2.2159459e-03f;
    output[2] = 9.9732780e-01f;
    output[3] = 1.2404647e-05f;
    output[4] = 4.2676922e-07f;

    return 0;
}

void backend_deinit(void)
{
    g_backend_ready = 0;
}
```

## `src/backend_qnn_api.c`

```c
/*
 * src/backend_qnn_api.c
 *
 * Backend QNN goi truc tiep QNN C API bang context binary da build san,
 * khong qua qnn-net-run CLI va khong parse .dlc luc runtime.
 *
 * Chuan bi truoc (mot lan, tren host/CI), xem scripts/build_context_binary.sh:
 *   qnn-context-binary-generator \
 *     --model $QAIRT_ROOT/lib/x86_64-linux-clang/libQnnModelDlc.so \
 *     --dlc_path traffic_qos_model.dlc \
 *     --backend  $QAIRT_ROOT/lib/x86_64-linux-clang/libQnnCpu.so \
 *     --binary_file traffic_qos_model.bin
 *
 * Bien moi truong:
 *   DL_QAIRT_ROOT       thu muc goc QAIRT SDK
 *   DL_QNN_BACKEND_LIB  duong dan libQnnCpu.so (mac dinh suy tu DL_QAIRT_ROOT)
 *   DL_MODEL_PATH       duong dan context binary .bin (KHONG phai .dlc)
 *   DL_GRAPH_NAME       ten graph trong model (mac dinh: traffic_qos_model)
 */

#include "backend.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnDevice.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"

#define DEFAULT_QAIRT_ROOT  "/home/congtuan/qairt_sdk/qairt/2.44.0.260225"
#define DEFAULT_MODEL_PATH  "/home/congtuan/model_test/traffic_qos_model.bin"
#define DEFAULT_GRAPH_NAME  "traffic_qos_model"

/* ---- state -------------------------------------------------------- */

static void *g_backend_lib_handle = NULL;
static void *g_system_lib_handle  = NULL;

static QNN_INTERFACE_VER_TYPE g_qnn = {0};
static QNN_SYSTEM_INTERFACE_VER_TYPE g_qnn_system = {0};

static Qnn_BackendHandle_t g_backend_handle = NULL;
static Qnn_DeviceHandle_t  g_device_handle  = NULL;
static Qnn_ContextHandle_t g_context_handle = NULL;
static Qnn_GraphHandle_t   g_graph_handle   = NULL;
static QnnSystemContext_Handle_t g_sys_context_handle = NULL;

/* Tensor lay tu chinh binary info cua context binary (co day du id, name,
 * rank, dimensions) - khong tu dung tu dau nhu ban truoc. */
static Qnn_Tensor_t g_input_tensor  = {0};
static Qnn_Tensor_t g_output_tensor = {0};

static int g_backend_ready = 0;

/* ---- helpers -------------------------------------------------------- */

static const char *env_or_default(const char *name, const char *fallback)
{
    const char *value = getenv(name);
    return (value != NULL && value[0] != '\0') ? value : fallback;
}

static int load_symbol(void *handle, const char *name, void **out_fn)
{
    void *fn = dlsym(handle, name);
    if (fn == NULL) {
        fprintf(stderr, "backend_qnn_api: missing symbol %s (%s)\n", name, dlerror());
        return -1;
    }
    *out_fn = fn;
    return 0;
}

static int load_backend_interface(const char *backend_lib_path)
{
    typedef Qnn_ErrorHandle_t (*GetProvidersFn)(const QnnInterface_t ***providers, uint32_t *num_providers);

    g_backend_lib_handle = dlopen(backend_lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (g_backend_lib_handle == NULL) {
        fprintf(stderr, "backend_qnn_api: dlopen(%s) failed: %s\n", backend_lib_path, dlerror());
        return -1;
    }

    void *fn = NULL;
    if (load_symbol(g_backend_lib_handle, "QnnInterface_getProviders", &fn) != 0) {
        return -1;
    }
    GetProvidersFn get_providers = (GetProvidersFn)fn;

    const QnnInterface_t **providers = NULL;
    uint32_t num_providers = 0;

    if (get_providers(&providers, &num_providers) != QNN_SUCCESS || num_providers == 0) {
        fprintf(stderr, "backend_qnn_api: QnnInterface_getProviders failed\n");
        return -1;
    }

    /* SDK co the tra ve nhieu provider (vi du nhieu API version). Trong
     * thuc te nen duyet providers[] va chon cai co
     * apiVersion.coreApiVersion khop voi header dang build cung, thay vi
     * luon lay phan tu dau tien. */
    g_qnn = providers[0]->QNN_INTERFACE_VER_NAME;
    return 0;
}

static int load_system_interface(const char *qairt_root)
{
    typedef Qnn_ErrorHandle_t (*GetSysProvidersFn)(const QnnSystemInterface_t ***providers, uint32_t *num_providers);

    char system_lib_path[2048];
    snprintf(system_lib_path, sizeof(system_lib_path),
             "%s/lib/x86_64-linux-clang/libQnnSystem.so", qairt_root);

    g_system_lib_handle = dlopen(system_lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (g_system_lib_handle == NULL) {
        fprintf(stderr, "backend_qnn_api: dlopen(%s) failed: %s\n", system_lib_path, dlerror());
        return -1;
    }

    void *fn = NULL;
    if (load_symbol(g_system_lib_handle, "QnnSystemInterface_getProviders", &fn) != 0) {
        return -1;
    }
    GetSysProvidersFn get_providers = (GetSysProvidersFn)fn;

    const QnnSystemInterface_t **providers = NULL;
    uint32_t num_providers = 0;

    if (get_providers(&providers, &num_providers) != QNN_SUCCESS || num_providers == 0) {
        fprintf(stderr, "backend_qnn_api: QnnSystemInterface_getProviders failed\n");
        return -1;
    }

    g_qnn_system = providers[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    return 0;
}

/* Doc metadata tensor that (id, name, rank, dimensions, dataType) tu chinh
 * context binary thay vi tu dung Qnn_Tensor_t rong. Day la nguyen nhan
 * QnnGraph_execute bao loi 6004 o ban truoc: tensor truyen vao execute
 * phai khop dung metadata da "dong bang" trong graph khi compile, khong
 * the tu tao tensor voi chi dataType + clientBuf. */
static int read_io_tensors_from_binary_info(const void *binary_buffer, uint64_t binary_size,
                                             const char *graph_name)
{
    if (g_qnn_system.systemContextCreate(&g_sys_context_handle) != QNN_SUCCESS) {
        fprintf(stderr, "backend_qnn_api: QnnSystemContext_create failed\n");
        return -1;
    }

    const QnnSystemContext_BinaryInfo_t *binary_info = NULL;
    Qnn_ContextBinarySize_t binary_info_size = 0;

    if (g_qnn_system.systemContextGetBinaryInfo(
            g_sys_context_handle,
            (void *)binary_buffer, binary_size,
            &binary_info, &binary_info_size) != QNN_SUCCESS || binary_info == NULL) {
        fprintf(stderr, "backend_qnn_api: QnnSystemContext_getBinaryInfo failed\n");
        return -1;
    }

    /* SDK 2.44.0.260225 tra ve version 3 cho model nay. GraphInfoV3_t va
     * BinaryInfoV3_t giu nguyen ten field nhu V1 (graphName, numGraphInputs,
     * graphInputs, numGraphOutputs, graphOutputs, numGraphs, graphs), chi
     * them vai field moi khong can dung o day (updateableTensors, blob...).
     * Nen dung chung logic cho V1 va V3, chi khac ten member trong union. */
    const QnnSystemContext_GraphInfo_t *graphs = NULL;
    uint32_t num_graphs = 0;

    if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        graphs = binary_info->contextBinaryInfoV1.graphs;
        num_graphs = binary_info->contextBinaryInfoV1.numGraphs;
    } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        graphs = binary_info->contextBinaryInfoV3.graphs;
        num_graphs = binary_info->contextBinaryInfoV3.numGraphs;
    } else {
        fprintf(stderr, "backend_qnn_api: unsupported binary info version %d\n",
                (int)binary_info->version);
        return -1;
    }

    const QnnSystemContext_GraphInfo_t *target_graph = NULL;
    for (uint32_t i = 0; i < num_graphs; ++i) {
        const char *this_graph_name = NULL;

        if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
            this_graph_name = graphs[i].graphInfoV1.graphName;
        } else if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
            this_graph_name = graphs[i].graphInfoV3.graphName;
        } else {
            continue;
        }

        if (this_graph_name != NULL && strcmp(this_graph_name, graph_name) == 0) {
            target_graph = &graphs[i];
            break;
        }
    }
    /* Neu khong tim thay dung ten (vi du DL_GRAPH_NAME khong khop), lay
     * graph dau tien de con chay duoc - nhung nen sua DL_GRAPH_NAME cho
     * dung ten that in ra tu log neu roi vao nhanh nay. */
    if (target_graph == NULL && num_graphs > 0) {
        const char *fallback_name = (graphs[0].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3)
            ? graphs[0].graphInfoV3.graphName
            : graphs[0].graphInfoV1.graphName;
        fprintf(stderr, "backend_qnn_api: graph '%s' not found, using graph[0] ('%s')\n",
                graph_name, fallback_name ? fallback_name : "?");
        target_graph = &graphs[0];
    }
    if (target_graph == NULL) {
        fprintf(stderr, "backend_qnn_api: no graph found in context binary\n");
        return -1;
    }

    uint32_t num_inputs = 0;
    uint32_t num_outputs = 0;
    Qnn_Tensor_t *graph_inputs = NULL;
    Qnn_Tensor_t *graph_outputs = NULL;

    if (target_graph->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        num_inputs = target_graph->graphInfoV3.numGraphInputs;
        num_outputs = target_graph->graphInfoV3.numGraphOutputs;
        graph_inputs = target_graph->graphInfoV3.graphInputs;
        graph_outputs = target_graph->graphInfoV3.graphOutputs;
    } else {
        num_inputs = target_graph->graphInfoV1.numGraphInputs;
        num_outputs = target_graph->graphInfoV1.numGraphOutputs;
        graph_inputs = target_graph->graphInfoV1.graphInputs;
        graph_outputs = target_graph->graphInfoV1.graphOutputs;
    }

    if (num_inputs < 1 || num_outputs < 1) {
        fprintf(stderr, "backend_qnn_api: graph has no input/output tensors\n");
        return -1;
    }

    /* Copy struct - dimensions/name ben trong van tro vao bo nho do
     * g_sys_context_handle so huu, nen phai giu handle nay song den luc
     * backend_deinit(), khong duoc free som. */
    g_input_tensor  = graph_inputs[0];
    g_output_tensor = graph_outputs[0];

    return 0;
}

static void *read_whole_file(const char *path, long *out_size)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "backend_qnn_api: cannot open %s\n", path);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (size <= 0) {
        fclose(fp);
        fprintf(stderr, "backend_qnn_api: invalid file size for %s\n", path);
        return NULL;
    }

    void *buffer = malloc((size_t)size);
    if (buffer == NULL || fread(buffer, 1, (size_t)size, fp) != (size_t)size) {
        fclose(fp);
        free(buffer);
        fprintf(stderr, "backend_qnn_api: failed to read %s\n", path);
        return NULL;
    }
    fclose(fp);

    *out_size = size;
    return buffer;
}

static int create_backend_device_context_from_binary(const void *buffer, long size)
{
    if (g_qnn.backendCreate(NULL, NULL, &g_backend_handle) != QNN_SUCCESS) {
        fprintf(stderr, "backend_qnn_api: QnnBackend_create failed\n");
        return -1;
    }

    /* CPU backend thuong KHONG ho tro Device API (khac GPU/HTP can quan ly
     * device/memory rieng). QnnDevice_create tra loi la hanh vi binh
     * thuong trong truong hop nay - khong coi la loi chi tu, chi log canh
     * bao va tiep tuc voi g_device_handle = NULL. Cac buoc sau (context
     * creation) chap nhan device handle NULL cho backend khong can device
     * quan ly rieng. */
    Qnn_ErrorHandle_t device_status = g_qnn.deviceCreate(NULL, NULL, &g_device_handle);
    if (device_status != QNN_SUCCESS) {
        fprintf(stderr, "backend_qnn_api: QnnDevice_create not supported by backend "
                         "(0x%lx), continuing without explicit device handle\n",
                (unsigned long)device_status);
        g_device_handle = NULL;
    }

    Qnn_ErrorHandle_t status = g_qnn.contextCreateFromBinary(
        g_backend_handle, g_device_handle, NULL,
        (void *)buffer, (uint64_t)size, &g_context_handle, NULL);

    if (status != QNN_SUCCESS) {
        fprintf(stderr, "backend_qnn_api: QnnContext_createFromBinary failed (0x%lx)\n",
                (unsigned long)status);
        return -1;
    }

    return 0;
}

static int retrieve_graph(const char *graph_name)
{
    if (g_qnn.graphRetrieve(g_context_handle, graph_name, &g_graph_handle) != QNN_SUCCESS) {
        fprintf(stderr, "backend_qnn_api: QnnGraph_retrieve('%s') failed\n", graph_name);
        return -1;
    }
    return 0;
}

/* ---- backend.h interface -------------------------------------------- */

int backend_init(void)
{
    const char *qairt_root = env_or_default("DL_QAIRT_ROOT", DEFAULT_QAIRT_ROOT);
    const char *model_path = env_or_default("DL_MODEL_PATH", DEFAULT_MODEL_PATH);
    const char *graph_name = env_or_default("DL_GRAPH_NAME", DEFAULT_GRAPH_NAME);

    char backend_lib_default[2048];
    snprintf(backend_lib_default, sizeof(backend_lib_default),
             "%s/lib/x86_64-linux-clang/libQnnCpu.so", qairt_root);
    const char *backend_lib = env_or_default("DL_QNN_BACKEND_LIB", backend_lib_default);

    if (load_backend_interface(backend_lib) != 0) {
        return -1;
    }

    if (load_system_interface(qairt_root) != 0) {
        return -1;
    }

    long binary_size = 0;
    void *binary_buffer = read_whole_file(model_path, &binary_size);
    if (binary_buffer == NULL) {
        return -1;
    }

    /* Doc metadata tensor that TRUOC (system context chi doc, khong giu
     * quyen so huu buffer), sau do dua buffer cho contextCreateFromBinary.
     * Buffer duoc giai phong ngay sau ca hai buoc, vi ca hai API deu copy
     * du lieu can thiet vao bo nho rieng cua chung. */
    int io_ok = read_io_tensors_from_binary_info(binary_buffer, (uint64_t)binary_size, graph_name) == 0;
    int ctx_ok = io_ok && create_backend_device_context_from_binary(binary_buffer, binary_size) == 0;

    free(binary_buffer);

    if (!io_ok || !ctx_ok) {
        return -1;
    }

    if (retrieve_graph(graph_name) != 0) {
        return -1;
    }

    g_backend_ready = 1;
    return 0;
}

int backend_execute(const float *input, int input_count, float *output, int output_count)
{
    if (!g_backend_ready || input == NULL || output == NULL) {
        return -1;
    }

    /* Tro tensor thang vao buffer cua caller trong RAM - khong ghi file
     * trung gian nhu backend_qnn_cli.c. */
    g_input_tensor.v1.clientBuf.data = (void *)input;
    g_input_tensor.v1.clientBuf.dataSize = (uint32_t)(input_count * sizeof(float));

    g_output_tensor.v1.clientBuf.data = (void *)output;
    g_output_tensor.v1.clientBuf.dataSize = (uint32_t)(output_count * sizeof(float));

    Qnn_Tensor_t inputs[1]  = { g_input_tensor };
    Qnn_Tensor_t outputs[1] = { g_output_tensor };

    Qnn_ErrorHandle_t status = g_qnn.graphExecute(
        g_graph_handle,
        inputs, 1,
        outputs, 1,
        NULL, NULL);

    if (status != QNN_SUCCESS) {
        fprintf(stderr, "backend_qnn_api: QnnGraph_execute failed (0x%lx)\n", (unsigned long)status);
        return -1;
    }

    return 0;
}

void backend_deinit(void)
{
    if (g_context_handle != NULL) {
        g_qnn.contextFree(g_context_handle, NULL);
        g_context_handle = NULL;
    }
    if (g_device_handle != NULL) {
        g_qnn.deviceFree(g_device_handle);
        g_device_handle = NULL;
    }
    if (g_backend_handle != NULL) {
        g_qnn.backendFree(g_backend_handle);
        g_backend_handle = NULL;
    }
    /* Giai phong sau cung vi g_input_tensor/g_output_tensor tro vao bo nho
     * do system context nay so huu. */
    if (g_sys_context_handle != NULL) {
        g_qnn_system.systemContextFree(g_sys_context_handle);
        g_sys_context_handle = NULL;
    }
    if (g_backend_lib_handle != NULL) {
        dlclose(g_backend_lib_handle);
        g_backend_lib_handle = NULL;
    }
    if (g_system_lib_handle != NULL) {
        dlclose(g_system_lib_handle);
        g_system_lib_handle = NULL;
    }

    memset(&g_input_tensor, 0, sizeof(g_input_tensor));
    memset(&g_output_tensor, 0, sizeof(g_output_tensor));

    g_backend_ready = 0;
}
```

## `src/backend_qnn_cli.c`

```c
#include "backend.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define DEFAULT_QAIRT_ROOT "/home/congtuan/qairt_sdk/qairt/2.44.0.260225"
#define DEFAULT_MODEL_PATH "/home/congtuan/model_test/traffic_qos_model.dlc"
#define DEFAULT_WORK_DIR "/tmp/ai_runtime_qnn"
#define DEFAULT_INPUT_NAME "input_seq"
#define DEFAULT_OUTPUT_NAME "class_probs"

static int g_backend_ready = 0;
static char g_qairt_root[1024];
static char g_model_path[1024];
static char g_work_dir[1024];
static char g_input_name[128];
static char g_output_name[128];

static const char *env_or_default(const char *name, const char *fallback)
{
    const char *value = getenv(name);
    return (value != NULL && value[0] != '\0') ? value : fallback;
}

static void copy_string(char *dst, size_t dst_size, const char *src)
{
    snprintf(dst, dst_size, "%s", src);
}

static int make_dir(const char *path)
{
    if (mkdir(path, 0755) == 0 || errno == EEXIST) {
        return 0;
    }
    return -1;
}

static int write_input_file(const char *path, const float *input, int input_count)
{
    FILE *fp = fopen(path, "wb");
    if (fp == NULL) {
        return -1;
    }

    size_t written = fwrite(input, sizeof(float), (size_t)input_count, fp);
    fclose(fp);

    return written == (size_t)input_count ? 0 : -1;
}

static int write_input_list(const char *path, const char *input_raw_path)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }

    fprintf(fp, "%s:=%s\n", g_input_name, input_raw_path);
    fclose(fp);
    return 0;
}

static int read_output_file(const char *path, float *output, int output_count)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return -1;
    }

    size_t read_count = fread(output, sizeof(float), (size_t)output_count, fp);
    fclose(fp);

    return read_count == (size_t)output_count ? 0 : -1;
}

static int run_qnn_net_run(const char *qnn_net_run,
                           const char *backend_lib,
                           const char *model_dlc_lib,
                           const char *input_list_path,
                           const char *output_dir,
                           const char *lib_dir,
                           const char *log_path)
{
    char ld_library_path[4096];
    const char *old_ld_library_path = getenv("LD_LIBRARY_PATH");

    if (old_ld_library_path != NULL && old_ld_library_path[0] != '\0') {
        snprintf(ld_library_path, sizeof(ld_library_path), "%s:%s", lib_dir, old_ld_library_path);
    } else {
        snprintf(ld_library_path, sizeof(ld_library_path), "%s", lib_dir);
    }

    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }

    if (pid == 0) {
        FILE *log_fp = fopen(log_path, "w");
        if (log_fp != NULL) {
            dup2(fileno(log_fp), STDOUT_FILENO);
            dup2(fileno(log_fp), STDERR_FILENO);
            fclose(log_fp);
        }

        setenv("LD_LIBRARY_PATH", ld_library_path, 1);

        execl(qnn_net_run,
              qnn_net_run,
              "--backend", backend_lib,
              "--model", model_dlc_lib,
              "--dlc_path", g_model_path,
              "--input_list", input_list_path,
              "--output_dir", output_dir,
              "--log_level", "error",
              (char *)NULL);
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }

    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : -1;
}

int backend_init(void)
{
    copy_string(g_qairt_root, sizeof(g_qairt_root), env_or_default("DL_QAIRT_ROOT", DEFAULT_QAIRT_ROOT));
    copy_string(g_model_path, sizeof(g_model_path), env_or_default("DL_MODEL_PATH", DEFAULT_MODEL_PATH));
    copy_string(g_work_dir, sizeof(g_work_dir), env_or_default("DL_WORK_DIR", DEFAULT_WORK_DIR));
    copy_string(g_input_name, sizeof(g_input_name), env_or_default("DL_INPUT_NAME", DEFAULT_INPUT_NAME));
    copy_string(g_output_name, sizeof(g_output_name), env_or_default("DL_OUTPUT_NAME", DEFAULT_OUTPUT_NAME));

    if (access(g_model_path, R_OK) != 0) {
        fprintf(stderr, "model is not readable: %s\n", g_model_path);
        return -1;
    }

    if (make_dir(g_work_dir) != 0) {
        fprintf(stderr, "failed to create work dir: %s\n", g_work_dir);
        return -1;
    }

    g_backend_ready = 1;
    return 0;
}

int backend_execute(const float *input, int input_count, float *output, int output_count)
{
    char input_raw_path[2048];
    char input_list_path[2048];
    char output_dir[2048];
    char output_raw_path[4096];
    char qnn_net_run[2048];
    char backend_lib[2048];
    char model_dlc_lib[2048];
    char lib_dir[2048];
    char log_path[2048];

    if (!g_backend_ready || input == NULL || output == NULL || input_count <= 0 || output_count <= 0) {
        return -1;
    }

    snprintf(input_raw_path, sizeof(input_raw_path), "%s/input_seq.raw", g_work_dir);
    snprintf(input_list_path, sizeof(input_list_path), "%s/input_list.txt", g_work_dir);
    snprintf(output_dir, sizeof(output_dir), "%s/output", g_work_dir);
    snprintf(output_raw_path, sizeof(output_raw_path), "%s/Result_0/%s.raw", output_dir, g_output_name);
    snprintf(qnn_net_run, sizeof(qnn_net_run), "%s/bin/x86_64-linux-clang/qnn-net-run", g_qairt_root);
    snprintf(backend_lib, sizeof(backend_lib), "%s/lib/x86_64-linux-clang/libQnnCpu.so", g_qairt_root);
    snprintf(model_dlc_lib, sizeof(model_dlc_lib), "%s/lib/x86_64-linux-clang/libQnnModelDlc.so", g_qairt_root);
    snprintf(lib_dir, sizeof(lib_dir), "%s/lib/x86_64-linux-clang", g_qairt_root);
    snprintf(log_path, sizeof(log_path), "%s/qnn_run.log", g_work_dir);

    if (write_input_file(input_raw_path, input, input_count) != 0) {
        fprintf(stderr, "failed to write input raw\n");
        return -1;
    }

    if (write_input_list(input_list_path, input_raw_path) != 0) {
        fprintf(stderr, "failed to write input list\n");
        return -1;
    }

    if (make_dir(output_dir) != 0) {
        fprintf(stderr, "failed to create output dir\n");
        return -1;
    }

    if (run_qnn_net_run(qnn_net_run, backend_lib, model_dlc_lib, input_list_path, output_dir, lib_dir, log_path) != 0) {
        fprintf(stderr, "qnn-net-run failed, see log: %s\n", log_path);
        return -1;
    }

    if (read_output_file(output_raw_path, output, output_count) != 0) {
        fprintf(stderr, "failed to read output raw: %s\n", output_raw_path);
        return -1;
    }

    return 0;
}

void backend_deinit(void)
{
    g_backend_ready = 0;
}
```

## `tools/model_deploy.py`

```python
#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

DEFAULT_QAIRT_ROOT = Path('/home/congtuan/qairt_sdk/qairt/2.44.0.260225')
DEFAULT_WORK_ROOT = Path('/tmp/model_deploy_tool')


def fail(message: str, code: int = 1):
    print(f'ERROR: {message}', file=sys.stderr)
    raise SystemExit(code)


def ensure_file(path: Path, label: str):
    if not path.is_file():
        fail(f'{label} not found: {path}')


def qairt_paths(qairt_root: Path):
    return {
        'bin': qairt_root / 'bin' / 'x86_64-linux-clang',
        'lib': qairt_root / 'lib' / 'x86_64-linux-clang',
        'qnn_net_run': qairt_root / 'bin' / 'x86_64-linux-clang' / 'qnn-net-run',
        'qnn_cpu': qairt_root / 'lib' / 'x86_64-linux-clang' / 'libQnnCpu.so',
        'qnn_model_dlc': qairt_root / 'lib' / 'x86_64-linux-clang' / 'libQnnModelDlc.so',
        'qairt_converter': qairt_root / 'bin' / 'x86_64-linux-clang' / 'qairt-converter',
        'qnn_onnx_converter': qairt_root / 'bin' / 'x86_64-linux-clang' / 'qnn-onnx-converter',
        'qnn_tflite_converter': qairt_root / 'bin' / 'x86_64-linux-clang' / 'qnn-tflite-converter',
        'snpe_onnx_to_dlc': qairt_root / 'bin' / 'x86_64-linux-clang' / 'snpe-onnx-to-dlc',
        'snpe_tflite_to_dlc': qairt_root / 'bin' / 'x86_64-linux-clang' / 'snpe-tflite-to-dlc',
    }


def run(cmd, env=None, cwd=None, log_path: Path | None = None):
    printable = ' '.join(str(x) for x in cmd)
    print(f'$ {printable}')
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w') as log:
            proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd, cwd=cwd, env=env, text=True)
    if proc.returncode != 0:
        if log_path is not None:
            fail(f'command failed, see log: {log_path}')
        fail('command failed')


def make_env(qairt_root: Path):
    paths = qairt_paths(qairt_root)
    env = os.environ.copy()
    env['PATH'] = f"{paths['bin']}:{env.get('PATH', '')}"
    env['LD_LIBRARY_PATH'] = f"{paths['lib']}:{env.get('LD_LIBRARY_PATH', '')}"
    python_dir = qairt_root / 'lib' / 'python'
    env['PYTHONPATH'] = f"{python_dir}:{env.get('PYTHONPATH', '')}"
    env['SNPE_ROOT'] = str(qairt_root)
    env['QNN_SDK_ROOT'] = str(qairt_root)
    env['QAIRT_ROOT'] = str(qairt_root)
    return env


def ensure_exists(path: Path, label: str):
    if not path.exists():
        fail(f'{label} not found: {path}')


def qairt_python(qairt_root: Path):
    python = qairt_root / 'qairt_env' / 'bin' / 'python'
    ensure_file(python, 'QAIRT Python environment')
    return python


def command_convert(args):
    qairt_root = Path(args.qairt_root)
    paths = qairt_paths(qairt_root)
    model = Path(args.model).resolve()
    output = Path(args.output).resolve()
    ensure_exists(model, 'input model')
    output.parent.mkdir(parents=True, exist_ok=True)

    ext = model.suffix.lower()
    env = make_env(qairt_root)
    python = qairt_python(qairt_root)
    converter = args.converter

    if ext == '.dlc':
        shutil.copy2(model, output)
        print(json.dumps({'status': 'success', 'dlc_path': str(output), 'mode': 'copy'}, indent=2))
        return

    if converter == 'auto':
        converter = 'qairt'

    if converter == 'qairt':
        ensure_file(paths['qairt_converter'], 'qairt-converter')
        cmd = [python, paths['qairt_converter'], '--input_network', model, '--output_path', output]
    elif converter == 'snpe':
        if ext == '.onnx':
            ensure_file(paths['snpe_onnx_to_dlc'], 'snpe-onnx-to-dlc')
            cmd = [python, paths['snpe_onnx_to_dlc'], '--input_network', model, '--output_path', output]
        elif ext == '.tflite':
            ensure_file(paths['snpe_tflite_to_dlc'], 'snpe-tflite-to-dlc')
            cmd = [python, paths['snpe_tflite_to_dlc'], '--input_network', model, '--output_path', output]
        else:
            fail(f'snpe converter only supports .onnx/.tflite in this prototype, got: {ext}')
    elif converter == 'qnn':
        if ext == '.onnx':
            ensure_file(paths['qnn_onnx_converter'], 'qnn-onnx-converter')
            cmd = [python, paths['qnn_onnx_converter'], '--input_network', model, '--output_path', output]
        elif ext == '.tflite':
            ensure_file(paths['qnn_tflite_converter'], 'qnn-tflite-converter')
            cmd = [python, paths['qnn_tflite_converter'], '--input_network', model, '--output_path', output]
        else:
            fail(f'qnn converter only supports .onnx/.tflite in this prototype, got: {ext}')
    else:
        fail(f'unsupported converter: {converter}')

    if args.source_model_input_shape:
        for item in args.source_model_input_shape:
            cmd.extend(['--source_model_input_shape', item[0], item[1]])

    if args.out_tensor_node:
        for name in args.out_tensor_node:
            cmd.extend(['--out_tensor_node', name])

    if args.extra_args:
        cmd.extend(args.extra_args)

    log_path = output.with_suffix('.convert.log')
    run(cmd, env=env, log_path=log_path)
    print(json.dumps({
        'status': 'success',
        'converter': converter,
        'dlc_path': str(output),
        'log_path': str(log_path),
    }, indent=2))
def write_input_list(input_name: str, input_raw: Path, input_list: Path):
    input_list.write_text(f'{input_name}:={input_raw}\n')


def read_float32(path: Path):
    data = path.read_bytes()
    if len(data) % 4 != 0:
        fail(f'raw output size is not float32-aligned: {path}')
    return list(struct.unpack('<' + 'f' * (len(data) // 4), data))


def command_run_host(args):
    qairt_root = Path(args.qairt_root)
    paths = qairt_paths(qairt_root)
    dlc = Path(args.dlc).resolve()
    input_raw = Path(args.input_raw).resolve()
    output_dir = Path(args.output_dir).resolve()
    work_dir = Path(args.work_dir).resolve()

    ensure_file(paths['qnn_net_run'], 'qnn-net-run')
    ensure_file(paths['qnn_cpu'], 'libQnnCpu.so')
    ensure_file(paths['qnn_model_dlc'], 'libQnnModelDlc.so')
    ensure_file(dlc, 'DLC model')
    ensure_file(input_raw, 'input raw')

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_list = work_dir / 'input_list.txt'
    write_input_list(args.input_name, input_raw, input_list)

    env = make_env(qairt_root)
    log_path = output_dir / 'qnn-net-run.log'
    cmd = [
        paths['qnn_net_run'],
        '--backend', paths['qnn_cpu'],
        '--model', paths['qnn_model_dlc'],
        '--dlc_path', dlc,
        '--input_list', input_list,
        '--output_dir', output_dir,
        '--log_level', args.log_level,
    ]
    run(cmd, env=env, log_path=log_path)

    output_raw = output_dir / 'Result_0' / f'{args.output_name}.raw'
    ensure_file(output_raw, 'QNN output')
    scores = read_float32(output_raw)
    label = max(range(len(scores)), key=lambda i: scores[i]) if scores else -1
    result = {'status': 'success', 'label': label, 'scores': scores, 'output_raw': str(output_raw), 'log_path': str(log_path)}
    print(json.dumps(result, indent=2))
    return result


def read_npy_float32(path: Path):
    try:
        import numpy as np
    except Exception as exc:
        fail(f'numpy is required to read .npy reference: {exc}')
    return [float(x) for x in np.load(path).reshape(-1).astype('float32')]


def command_validate(args):
    command_run_host(args)
    output_raw = Path(args.output_dir).resolve() / 'Result_0' / f'{args.output_name}.raw'
    qnn = read_float32(output_raw)
    ref_path = Path(args.reference).resolve()
    ensure_file(ref_path, 'reference output')

    if ref_path.suffix.lower() == '.npy':
        ref_values = read_npy_float32(ref_path)
    elif ref_path.suffix.lower() == '.raw':
        ref_values = read_float32(ref_path)
    else:
        fail('reference must be .npy or .raw')

    n = min(len(qnn), len(ref_values))
    diffs = [abs(qnn[i] - ref_values[i]) for i in range(n)]
    max_abs_diff = max(diffs) if diffs else float('inf')
    mean_abs_diff = sum(diffs) / len(diffs) if diffs else float('inf')
    qnn_argmax = max(range(len(qnn)), key=lambda i: qnn[i]) if qnn else -1
    ref_argmax = max(range(len(ref_values)), key=lambda i: ref_values[i]) if ref_values else -1
    passed = len(qnn) == len(ref_values) and max_abs_diff <= args.tolerance and qnn_argmax == ref_argmax

    result = {
        'status': 'success' if passed else 'failed',
        'validation_passed': passed,
        'qnn_shape': [len(qnn)],
        'reference_shape': [len(ref_values)],
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'qnn_argmax': qnn_argmax,
        'reference_argmax': ref_argmax,
        'tolerance': args.tolerance,
    }
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


def build_parser():
    parser = argparse.ArgumentParser(description='Model deploy helper for QAIRT/QNN prototype')
    parser.add_argument('--qairt-root', default=str(DEFAULT_QAIRT_ROOT))
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('convert', help='Convert source model to QAIRT/QNN deployable artifact')
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--converter', choices=['auto', 'qairt', 'qnn', 'snpe'], default='qairt')
    p.add_argument('--source-model-input-shape', nargs=2, action='append', metavar=('INPUT_NAME', 'INPUT_DIMS'))
    p.add_argument('--out-tensor-node', action='append')
    p.add_argument('extra_args', nargs=argparse.REMAINDER)
    p.set_defaults(func=command_convert)

    p = sub.add_parser('run-host', help='Run a DLC using QNN CPU backend on host')
    p.add_argument('--dlc', required=True)
    p.add_argument('--input-raw', required=True)
    p.add_argument('--input-name', default='input_seq')
    p.add_argument('--output-name', default='class_probs')
    p.add_argument('--output-dir', default=str(DEFAULT_WORK_ROOT / 'run_output'))
    p.add_argument('--work-dir', default=str(DEFAULT_WORK_ROOT / 'work'))
    p.add_argument('--log-level', default='error')
    p.set_defaults(func=command_run_host)

    p = sub.add_parser('validate', help='Run host inference and compare with reference output')
    p.add_argument('--dlc', required=True)
    p.add_argument('--input-raw', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--input-name', default='input_seq')
    p.add_argument('--output-name', default='class_probs')
    p.add_argument('--output-dir', default=str(DEFAULT_WORK_ROOT / 'validate_output'))
    p.add_argument('--work-dir', default=str(DEFAULT_WORK_ROOT / 'validate_work'))
    p.add_argument('--log-level', default='error')
    p.add_argument('--tolerance', type=float, default=1e-5)
    p.set_defaults(func=command_validate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
```

## `tools/model_deploy_server.py`

```python
#!/usr/bin/env python3
import argparse
import contextlib
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import model_deploy


def normalize_body(body):
    return body if isinstance(body, dict) else {}


def namespace_for_run(body):
    body = normalize_body(body)
    return argparse.Namespace(
        qairt_root=body.get('qairt_root', str(model_deploy.DEFAULT_QAIRT_ROOT)),
        dlc=body['dlc'],
        input_raw=body['input_raw'],
        input_name=body.get('input_name', 'input_seq'),
        output_name=body.get('output_name', 'class_probs'),
        output_dir=body.get('output_dir', str(model_deploy.DEFAULT_WORK_ROOT / 'api_run_output')),
        work_dir=body.get('work_dir', str(model_deploy.DEFAULT_WORK_ROOT / 'api_work')),
        log_level=body.get('log_level', 'error'),
    )


def namespace_for_validate(body):
    ns = namespace_for_run(body)
    ns.reference = body['reference']
    ns.tolerance = float(body.get('tolerance', 1e-5))
    return ns


def namespace_for_convert(body):
    body = normalize_body(body)
    return argparse.Namespace(
        qairt_root=body.get('qairt_root', str(model_deploy.DEFAULT_QAIRT_ROOT)),
        model=body['model'],
        output=body['output'],
        converter=body.get('converter', 'qairt'),
        source_model_input_shape=body.get('source_model_input_shape'),
        out_tensor_node=body.get('out_tensor_node'),
        extra_args=body.get('extra_args', []),
    )


def capture_json(func, args):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        func(args)
    text = stream.getvalue()
    json_objects = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        start = text.find('{', index)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            break
        json_objects.append(obj)
        index = start + end
    return json_objects[-1] if json_objects else {'status': 'success', 'log': text}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        data = json.dumps(payload, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get('Content-Length', '0'))
        raw = self.rfile.read(length) if length else b'{}'
        return json.loads(raw.decode('utf-8'))

    def do_GET(self):
        if self.path == '/health':
            self._send(200, {'status': 'ok'})
            return
        self._send(404, {'status': 'error', 'message': 'not found'})

    def do_POST(self):
        try:
            body = self._read_json()
            if self.path == '/run-host':
                result = capture_json(model_deploy.command_run_host, namespace_for_run(body))
                self._send(200, result)
            elif self.path == '/validate':
                result = capture_json(model_deploy.command_validate, namespace_for_validate(body))
                status = 200 if result.get('validation_passed') else 422
                self._send(status, result)
            elif self.path == '/convert':
                result = capture_json(model_deploy.command_convert, namespace_for_convert(body))
                self._send(200, result)
            else:
                self._send(404, {'status': 'error', 'message': 'not found'})
        except KeyError as exc:
            self._send(400, {'status': 'error', 'message': f'missing field: {exc.args[0]}'})
        except SystemExit as exc:
            self._send(500, {'status': 'error', 'message': f'command failed with code {exc.code}'})
        except Exception as exc:
            self._send(500, {'status': 'error', 'message': str(exc)})

    def log_message(self, fmt, *args):
        sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))


def main():
    parser = argparse.ArgumentParser(description='REST API server for model deploy prototype')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8088)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'listening on http://{args.host}:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
```

## `tools/package_project_context.py`

```python
#!/usr/bin/env python3
"""Package the prototype source and notes into one shareable context archive."""

from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "context_package"
ARCHIVE_PATH = OUTPUT_DIR / "ai_runtime_prototype_context.zip"
TEXT_PATH = OUTPUT_DIR / "ai_runtime_prototype_context.md"

INCLUDED_SUFFIXES = {
    ".c",
    ".h",
    ".md",
    ".py",
    ".sh",
    ".mk",
    ".txt",
}
INCLUDED_NAMES = {"Makefile", ".gitignore"}
EXCLUDED_PARTS = {"build", ".git", "__pycache__", "context_package"}


def source_files() -> list[Path]:
    files = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_PARTS for part in path.relative_to(PROJECT_ROOT).parts):
            continue
        if path.name in INCLUDED_NAMES or path.suffix.lower() in INCLUDED_SUFFIXES:
            files.append(path)
    return sorted(files)


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def build_text(files: list[Path]) -> str:
    sections = [
        "# ai_runtime_prototype context",
        "",
        "Generated source and notes bundle. Binary files and build output are excluded.",
        "",
    ]
    for path in files:
        name = relative_path(path)
        language = {
            ".c": "c",
            ".h": "c",
            ".md": "markdown",
            ".py": "python",
            ".sh": "bash",
            ".mk": "makefile",
            ".txt": "text",
        }.get(path.suffix.lower(), "text")
        if path.name == "Makefile":
            language = "makefile"
        sections.extend(
            [
                f"## `{name}`",
                "",
                f"```{language}",
                path.read_text(encoding="utf-8", errors="replace").rstrip(),
                "```",
                "",
            ]
        )
    return "\n".join(sections)


def main() -> None:
    files = source_files()
    text = build_text(files)
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEXT_PATH.write_text(text, encoding="utf-8")
    with zipfile.ZipFile(ARCHIVE_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(TEXT_PATH, TEXT_PATH.name)
        for path in files:
            archive.write(path, relative_path(path))
    print(f"Packaged {len(files)} files")
    print(f"Text bundle: {TEXT_PATH}")
    print(f"Zip archive: {ARCHIVE_PATH}")


if __name__ == "__main__":
    main()
```
