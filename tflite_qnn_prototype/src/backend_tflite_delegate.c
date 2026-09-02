/*
 * src/backend_tflite_delegate.c
 *
 * Track B: TFLite Interpreter + QNN External Delegate.
 * Artifact deploy la .tflite (KHONG convert sang .dlc/.bin nhu Track A).
 *
 * QUYET DINH KIEN TRUC: dung QNN "External Delegate Interface"
 * (tensorflow/lite/delegates/external/external_delegate.h), KHONG dung
 * "Qualcomm AI Engine Direct Delegate Interface" (QnnTFLiteDelegate.h).
 * Ly do:
 *   - QnnTFLiteDelegate.h doc lai la Android-primary (tai lieu Qualcomm:
 *     "Aarch64-Android is supported"), va dung interpreter->ModifyGraphWithDelegate()
 *     - API C++ cua tflite::Interpreter, khong phai C API thuan.
 *   - External Delegate Interface dung API C thuan (TfLiteInterpreterCreate,
 *     TfLiteInterpreterOptionsAddDelegate...), cung ho voi kien truc C cua
 *     ai_runtime.c/backend.h hien tai, va chinh Qualcomm dung co che nay cho
 *     benchmark_model tren Linux nhung (vi du OS "le", khong rieng Android)
 *     - phu hop thiet bi dich OpenWrt/musl libc cua du an nay hon.
 *
 * YEU CAU MOI TRUONG (thieu 1 trong 2 cai duoi thi khong build duoc):
 *   1. TFLite C API library + header (libtensorflowlite_c.so +
 *      tensorflow/lite/c/c_api.h + tensorflow/lite/delegates/external/external_delegate.h)
 *      - QAIRT SDK KHONG dong goi san cai nay, phai tu build tu TensorFlow
 *        source (Bazel) hoac tim ban prebuilt rieng.
 *   2. libQnnTFLiteDelegate.so tu QAIRT SDK (thuong nam trong
 *      lib/x86_64-linux-clang/ cho host test, hoac lib/aarch64-.../ cho target).
 *
 * Bien moi truong:
 *   DL_TFLITE_MODEL_PATH     duong dan .tflite
 *   DL_QNN_DELEGATE_LIB      duong dan libQnnTFLiteDelegate.so
 *   DL_QNN_BACKEND_LIB       duong dan libQnnCpu.so / libQnnHtp.so / libQnnGpu.so
 *   DL_QNN_BACKEND_TYPE      "cpu" | "gpu" | "htp" (mac dinh: cpu - de test host truoc)
 *   DL_QNN_SKEL_DIR          chi can cho htp tren thiet bi that (skel_library_dir)
 *   DL_DISABLE_QNN_DELEGATE  set non-empty/non-0 de bo qua delegate, chay TFLite CPU thuan
 *   DL_QNN_STRICT_DELEGATE   set non-empty/non-0 de fail ngay neu delegate khong init duoc
 */

#include "backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/external/external_delegate.h"

#define DEFAULT_TFLITE_MODEL     "/home/congtuan/model_test/tflite_output/traffic_qos_model_float32.tflite"
#define DEFAULT_QNN_DELEGATE_LIB "/home/congtuan/qairt_sdk/qairt/2.44.0.260225/lib/x86_64-linux-clang/libQnnTFLiteDelegate.so"
#define DEFAULT_QNN_BACKEND_LIB  "/home/congtuan/qairt_sdk/qairt/2.44.0.260225/lib/x86_64-linux-clang/libQnnCpu.so"
#define DEFAULT_QNN_BACKEND_TYPE "cpu"

/* ---- state ------------------------------------------------------------ */

static TfLiteModel *g_model                        = NULL;
static TfLiteInterpreterOptions *g_interp_options  = NULL;
static TfLiteInterpreter *g_interpreter            = NULL;
static TfLiteDelegate *g_delegate                  = NULL;

static int g_backend_ready = 0;
static int g_using_delegate = 0;

/* THEM MOI: ghi lai LY DO cu the neu roi ve CPU thuan, de khong con phai
 * doan qua log rai rac - in ra 1 banner CANH BAO ro rang o cuoi backend_init()
 * thay vi de nguoi dung tuong nham "chay duoc" = "chay tren HTP/QNN". */
static const char *g_fallback_reason = NULL;

/* ---- helpers ---------------------------------------------------------- */

static const char *env_or_default(const char *name, const char *fallback)
{
    const char *value = getenv(name);
    return (value != NULL && value[0] != '\0') ? value : fallback;
}

static int env_flag_enabled(const char *name)
{
    const char *value = getenv(name);
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static int readable_file_exists(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return 0;
    }
    fclose(fp);
    return 1;
}

/* Dung dinh dang key:value;key:value giong het benchmark_model va cac vi
 * du chinh thuc cua Qualcomm (vd:
 * "backend_type:htp;library_path:/usr/lib/libQnnHtp.so;skel_library_dir:/usr/lib/rfsa/adsp"),
 * roi parse tach tung cap de goi options.insert() cua external_delegate.h. */
static int apply_external_delegate_options(TfLiteExternalDelegateOptions *options,
                                           const char *backend_type,
                                           const char *backend_lib,
                                           const char *skel_dir)
{
    char combined[2048];
    if (skel_dir != NULL && skel_dir[0] != '\0') {
        snprintf(combined, sizeof(combined), "backend_type:%s;library_path:%s;skel_library_dir:%s",
                 backend_type, backend_lib, skel_dir);
    } else {
        snprintf(combined, sizeof(combined), "backend_type:%s;library_path:%s",
                 backend_type, backend_lib);
    }

    char *buf = strdup(combined);
    if (buf == NULL) {
        return -1;
    }

    char *saveptr = NULL;
    char *token = strtok_r(buf, ";", &saveptr);
    while (token != NULL) {
        char *colon = strchr(token, ':');
        if (colon != NULL) {
            *colon = '\0';
            const char *key = token;
            const char *value = colon + 1;
            options->insert(options, key, value);
        }
        token = strtok_r(NULL, ";", &saveptr);
    }

    free(buf);
    return 0;
}

static int create_plain_interpreter(void)
{
    if (g_interp_options != NULL) {
        TfLiteInterpreterOptionsDelete(g_interp_options);
        g_interp_options = NULL;
    }
    if (g_delegate != NULL) {
        TfLiteExternalDelegateDelete(g_delegate);
        g_delegate = NULL;
    }

    g_interp_options = TfLiteInterpreterOptionsCreate();
    if (g_interp_options == NULL) {
        fprintf(stderr, "backend_tflite_delegate: TfLiteInterpreterOptionsCreate failed\n");
        return -1;
    }

    g_interpreter = TfLiteInterpreterCreate(g_model, g_interp_options);
    if (g_interpreter == NULL) {
        fprintf(stderr, "backend_tflite_delegate: plain TfLiteInterpreterCreate failed\n");
        return -1;
    }

    g_using_delegate = 0;
    return 0;
}

/* ---- backend.h interface ---------------------------------------------- */

int backend_init(void)
{
    g_fallback_reason = NULL;
    const char *model_path   = env_or_default("DL_TFLITE_MODEL_PATH", DEFAULT_TFLITE_MODEL);
    const char *delegate_lib = env_or_default("DL_QNN_DELEGATE_LIB", DEFAULT_QNN_DELEGATE_LIB);
    const char *backend_lib  = env_or_default("DL_QNN_BACKEND_LIB", DEFAULT_QNN_BACKEND_LIB);
    const char *backend_type = env_or_default("DL_QNN_BACKEND_TYPE", DEFAULT_QNN_BACKEND_TYPE);
    const char *skel_dir     = getenv("DL_QNN_SKEL_DIR");
    const int delegate_disabled_by_env = env_flag_enabled("DL_DISABLE_QNN_DELEGATE");
    int use_delegate = !delegate_disabled_by_env;
    const int strict_delegate = env_flag_enabled("DL_QNN_STRICT_DELEGATE");

    fprintf(stderr, "backend_tflite_delegate: model=%s\n", model_path);
    if (!readable_file_exists(model_path)) {
        fprintf(stderr, "backend_tflite_delegate: model file is not readable: %s\n", model_path);
        return -1;
    }

    g_model = TfLiteModelCreateFromFile(model_path);
    if (g_model == NULL) {
        fprintf(stderr, "backend_tflite_delegate: cannot load model %s\n", model_path);
        return -1;
    }

    if (use_delegate) {
        fprintf(stderr, "backend_tflite_delegate: delegate=%s\n", delegate_lib);
        fprintf(stderr, "backend_tflite_delegate: backend_type=%s backend_lib=%s\n",
                backend_type, backend_lib);
        if (!readable_file_exists(delegate_lib)) {
            fprintf(stderr, "backend_tflite_delegate: delegate lib is not readable: %s\n",
                    delegate_lib);
            if (strict_delegate) {
                return -1;
            }
            use_delegate = 0;
            /* THEM MOI: ghi ly do fallback thay vi chi log roi im lang tiep tuc. */
            g_fallback_reason = "delegate lib khong doc duoc (DL_QNN_DELEGATE_LIB sai duong dan?)";
        }
        if (!readable_file_exists(backend_lib)) {
            fprintf(stderr, "backend_tflite_delegate: backend lib is not readable: %s\n",
                    backend_lib);
            if (strict_delegate) {
                return -1;
            }
            use_delegate = 0;
            g_fallback_reason = "backend lib (libQnnCpu/Htp/Gpu.so) khong doc duoc (DL_QNN_BACKEND_LIB sai duong dan?)";
        }
        if (!use_delegate) {
            fprintf(stderr, "backend_tflite_delegate: falling back to plain TFLite CPU before delegate creation\n");
        }
    }

    if (use_delegate) {
        TfLiteExternalDelegateOptions ext_options = TfLiteExternalDelegateOptionsDefault(delegate_lib);
        if (apply_external_delegate_options(&ext_options, backend_type, backend_lib, skel_dir) != 0) {
            fprintf(stderr, "backend_tflite_delegate: failed to build delegate options\n");
            return -1;
        }

        g_delegate = TfLiteExternalDelegateCreate(&ext_options);
        if (g_delegate == NULL) {
            fprintf(stderr, "backend_tflite_delegate: TfLiteExternalDelegateCreate failed "
                            "(delegate_lib=%s, backend_lib=%s) - falling back to plain TFLite CPU\n",
                    delegate_lib, backend_lib);
            if (strict_delegate) {
                return -1;
            }
            /* THEM MOI */
            g_fallback_reason = "TfLiteExternalDelegateCreate() tra ve NULL - kiem tra lai "
                                 "backend_type/library_path/skel_library_dir co dung SDK/target khong";
        }
    } else if (delegate_disabled_by_env) {
        fprintf(stderr, "backend_tflite_delegate: QNN delegate disabled by DL_DISABLE_QNN_DELEGATE\n");
        /* THEM MOI: day la fallback CO CHU DICH (nguoi dung tu tat), khac voi
         * cac truong hop fail o tren - van ghi nhan de banner cuoi khong bi
         * hieu nham la "loi", ma la "chu dong chay CPU". */
        g_fallback_reason = "DL_DISABLE_QNN_DELEGATE dang bat (chu dong tat delegate, khong phai loi)";
    }

    g_interp_options = TfLiteInterpreterOptionsCreate();
    if (g_interp_options == NULL) {
        fprintf(stderr, "backend_tflite_delegate: TfLiteInterpreterOptionsCreate failed\n");
        return -1;
    }
    if (g_delegate != NULL) {
        TfLiteInterpreterOptionsAddDelegate(g_interp_options, g_delegate);
        g_using_delegate = 1;
    }

    g_interpreter = TfLiteInterpreterCreate(g_model, g_interp_options);
    if (g_interpreter == NULL) {
        fprintf(stderr,
                "backend_tflite_delegate: TfLiteInterpreterCreate with delegate failed "
                "- retrying plain TFLite CPU\n");
        if (strict_delegate) {
            return -1;
        }
        g_fallback_reason = "TfLiteInterpreterCreate() that bai khi gan delegate - "
                             "co the model co op khong duoc delegate ho tro, hoac ABI mismatch";
        if (create_plain_interpreter() != 0) {
            return -1;
        }
    }

    if (TfLiteInterpreterAllocateTensors(g_interpreter) != kTfLiteOk) {
        fprintf(stderr, "backend_tflite_delegate: AllocateTensors failed\n");
        return -1;
    }

    const TfLiteTensor *in_tensor = TfLiteInterpreterGetInputTensor(g_interpreter, 0);
    const TfLiteTensor *out_tensor = TfLiteInterpreterGetOutputTensor(g_interpreter, 0);
    if (in_tensor == NULL || out_tensor == NULL) {
        fprintf(stderr, "backend_tflite_delegate: failed to read input/output tensor metadata\n");
        return -1;
    }

    fprintf(stderr, "backend_tflite_delegate: ready mode=%s input_bytes=%zu output_bytes=%zu\n",
            g_using_delegate ? "qnn_delegate" : "tflite_cpu",
            TfLiteTensorByteSize(in_tensor),
            TfLiteTensorByteSize(out_tensor));

    /* THEM MOI: banner canh bao KHONG THE BO QUA khi thuc te dang chay CPU
     * thuan, du ban dau co the da yeu cau delegate. Truoc day thong tin nay
     * chi nam lan trong 1 dong log "ready mode=..." - de bi luot qua khi
     * doc terminal dai. Muon tat banner nay (vi da CHU DONG chon CPU) thi
     * dat DL_DISABLE_QNN_DELEGATE=1, luc do g_fallback_reason se noi ro la
     * "chu dong", khong in banner ***. */
    if (!g_using_delegate && g_fallback_reason != NULL && !delegate_disabled_by_env) {
        fprintf(stderr,
                "backend_tflite_delegate: *** CANH BAO: DANG CHAY TFLITE CPU THUAN, "
                "KHONG PHAI QNN/HTP DELEGATE ***\n"
                "backend_tflite_delegate: ly do: %s\n"
                "backend_tflite_delegate: neu muon fail cung thay vi fallback am tham, "
                "chay lai voi DL_QNN_STRICT_DELEGATE=1\n",
                g_fallback_reason);
    }

    g_backend_ready = 1;
    return 0;
}

/* THEM MOI: doc dtype that cua tensor (kTfLiteUInt8/kTfLiteInt8/kTfLiteFloat32)
 * qua TfLiteTensorType(), va tham so quantization qua
 * TfLiteTensorQuantizationParams() - ca hai deu la TFLite C API co san,
 * khong can sua gi ben ngoai file nay. */
static dl_tensor_dtype_t map_tflite_type(TfLiteType t)
{
    switch (t) {
        case kTfLiteFloat32: return DL_DTYPE_FLOAT32;
        case kTfLiteUInt8: return DL_DTYPE_UINT8;
        case kTfLiteInt8:  return DL_DTYPE_INT8;
        case kTfLiteFloat16: return DL_DTYPE_FLOAT16;
        case kTfLiteUInt16: return DL_DTYPE_UINT16;
        case kTfLiteInt16: return DL_DTYPE_INT16;
        case kTfLiteUInt32: return DL_DTYPE_UINT32;
        case kTfLiteInt32: return DL_DTYPE_INT32;
        case kTfLiteInt64: return DL_DTYPE_INT64;
        case kTfLiteBool: return DL_DTYPE_BOOL8;
        default: return DL_DTYPE_UNKNOWN;
    }
}

int backend_get_io_count(int *input_count, int *output_count)
{
    if (!g_backend_ready || input_count == NULL || output_count == NULL ||
        TfLiteInterpreterGetInputTensorCount(g_interpreter) != 1 ||
        TfLiteInterpreterGetOutputTensorCount(g_interpreter) != 1) {
        return -1;
    }

    /* Gia dinh model 1 input / 1 output, giong quy uoc cua backend_qnn_api.c.
     * Neu model TFLite thuc te co nhieu input/output tensor, can mo rong
     * interface backend.h de tra ve mang thay vi 1 gia tri. */
    const TfLiteTensor *in_tensor = TfLiteInterpreterGetInputTensor(g_interpreter, 0);
    const TfLiteTensor *out_tensor = TfLiteInterpreterGetOutputTensor(g_interpreter, 0);
    if (in_tensor == NULL || out_tensor == NULL) {
        return -1;
    }

    /* SUA: truoc day luon chia cho sizeof(float), sai voi model uint8/int8
     * (1 byte/phan tu chu khong phai 4). Gio chia theo dung kich thuoc
     * kieu du lieu that cua tensor. Voi model float32 (truong hop cu),
     * ket qua giu nguyen y het truoc day. */
    TfLiteType in_type = TfLiteTensorType(in_tensor);
    TfLiteType out_type = TfLiteTensorType(out_tensor);
    if ((in_type != kTfLiteFloat32 && in_type != kTfLiteUInt8 && in_type != kTfLiteInt8) ||
        (out_type != kTfLiteFloat32 && out_type != kTfLiteUInt8 && out_type != kTfLiteInt8)) {
        fprintf(stderr, "backend_tflite_delegate: legacy API supports only float32/uint8/int8\n");
        return -1;
    }
    size_t in_elem_size = (in_type == kTfLiteUInt8 || in_type == kTfLiteInt8) ? 1 : sizeof(float);
    size_t out_elem_size = (out_type == kTfLiteUInt8 || out_type == kTfLiteInt8) ? 1 : sizeof(float);

    *input_count  = (int)(TfLiteTensorByteSize(in_tensor) / in_elem_size);
    *output_count = (int)(TfLiteTensorByteSize(out_tensor) / out_elem_size);
    return 0;
}

int backend_get_tensor_count(int *input_tensor_count, int *output_tensor_count)
{
    if (!g_backend_ready || input_tensor_count == NULL || output_tensor_count == NULL) return -1;
    *input_tensor_count = TfLiteInterpreterGetInputTensorCount(g_interpreter);
    *output_tensor_count = TfLiteInterpreterGetOutputTensorCount(g_interpreter);
    return (*input_tensor_count > 0 && *output_tensor_count > 0) ? 0 : -1;
}

int backend_get_tensor_info(int is_input, int index, dl_tensor_info_t *info)
{
    if (!g_backend_ready || index < 0 || info == NULL) return -1;
    int tensor_count = is_input ? TfLiteInterpreterGetInputTensorCount(g_interpreter)
                                : TfLiteInterpreterGetOutputTensorCount(g_interpreter);
    if (index >= tensor_count) return -1;
    const TfLiteTensor *tensor = is_input
        ? TfLiteInterpreterGetInputTensor(g_interpreter, index)
        : TfLiteInterpreterGetOutputTensor(g_interpreter, index);
    if (tensor == NULL) return -1;

    memset(info, 0, sizeof(*info));
    const char *name = TfLiteTensorName(tensor);
    snprintf(info->name, sizeof(info->name), "%s", name != NULL ? name : (is_input ? "input_0" : "output_0"));
    info->dtype = map_tflite_type(TfLiteTensorType(tensor));
    if (info->dtype == DL_DTYPE_UNKNOWN) return -1;
    info->rank = TfLiteTensorNumDims(tensor);
    if (info->rank < 0 || info->rank > DL_MAX_TENSOR_RANK) return -1;
    size_t count = 1;
    for (int i = 0; i < info->rank; ++i) {
        int dim = TfLiteTensorDim(tensor, i);
        if (dim <= 0) return -1;
        info->dimensions[i] = (uint32_t)dim;
        count *= (size_t)dim;
    }
    info->element_count = count;
    info->byte_size = TfLiteTensorByteSize(tensor);
    TfLiteQuantizationParams quant = TfLiteTensorQuantizationParams(tensor);
    info->scale = info->dtype == DL_DTYPE_FLOAT32 ? 1.0f : quant.scale;
    info->zero_point = info->dtype == DL_DTYPE_FLOAT32 ? 0 : quant.zero_point;
    info->quantized_axis = -1;
    return 0;
}

int backend_get_io_dtype(dl_tensor_dtype_t *input_dtype, float *input_scale, int *input_zero_point,
                          dl_tensor_dtype_t *output_dtype, float *output_scale, int *output_zero_point)
{
    if (!g_backend_ready) {
        return -1;
    }

    const TfLiteTensor *in_tensor = TfLiteInterpreterGetInputTensor(g_interpreter, 0);
    const TfLiteTensor *out_tensor = TfLiteInterpreterGetOutputTensor(g_interpreter, 0);
    if (in_tensor == NULL || out_tensor == NULL) {
        return -1;
    }

    TfLiteQuantizationParams in_q = TfLiteTensorQuantizationParams(in_tensor);
    TfLiteQuantizationParams out_q = TfLiteTensorQuantizationParams(out_tensor);

    if (input_dtype != NULL)      *input_dtype = map_tflite_type(TfLiteTensorType(in_tensor));
    if (input_scale != NULL)      *input_scale = in_q.scale;
    if (input_zero_point != NULL) *input_zero_point = in_q.zero_point;

    if (output_dtype != NULL)      *output_dtype = map_tflite_type(TfLiteTensorType(out_tensor));
    if (output_scale != NULL)      *output_scale = out_q.scale;
    if (output_zero_point != NULL) *output_zero_point = out_q.zero_point;

    return 0;
}

int backend_execute(const float *input, int input_count, float *output, int output_count)
{
    if (!g_backend_ready || input == NULL || output == NULL) {
        return -1;
    }

    TfLiteTensor *in_tensor = TfLiteInterpreterGetInputTensor(g_interpreter, 0);
    if (in_tensor == NULL) {
        return -1;
    }

    size_t input_bytes = (size_t)input_count * sizeof(float);
    size_t expected_input_bytes = TfLiteTensorByteSize(in_tensor);
    if (input_bytes != expected_input_bytes) {
        fprintf(stderr,
                "backend_tflite_delegate: input size mismatch: provided=%zu bytes expected=%zu bytes\n",
                input_bytes, expected_input_bytes);
        return -1;
    }

    if (TfLiteTensorCopyFromBuffer(in_tensor, input, input_bytes) != kTfLiteOk) {
        fprintf(stderr, "backend_tflite_delegate: TfLiteTensorCopyFromBuffer failed\n");
        return -1;
    }

    if (TfLiteInterpreterInvoke(g_interpreter) != kTfLiteOk) {
        fprintf(stderr, "backend_tflite_delegate: TfLiteInterpreterInvoke failed\n");
        return -1;
    }

    const TfLiteTensor *out_tensor = TfLiteInterpreterGetOutputTensor(g_interpreter, 0);
    if (out_tensor == NULL) {
        return -1;
    }

    size_t output_bytes = (size_t)output_count * sizeof(float);
    size_t expected_output_bytes = TfLiteTensorByteSize(out_tensor);
    if (output_bytes != expected_output_bytes) {
        fprintf(stderr,
                "backend_tflite_delegate: output size mismatch: provided=%zu bytes expected=%zu bytes\n",
                output_bytes, expected_output_bytes);
        return -1;
    }

    if (TfLiteTensorCopyToBuffer(out_tensor, output, output_bytes) != kTfLiteOk) {
        fprintf(stderr, "backend_tflite_delegate: TfLiteTensorCopyToBuffer failed\n");
        return -1;
    }

    return 0;
}

/* THEM MOI: giong het backend_execute() ve luong xu ly (copy input ->
 * Invoke -> copy output), chi khac la lam viec voi byte tho theo dung
 * dtype that cua tensor thay vi ep ve float. input_count/output_count la
 * SO PHAN TU - ham nay tu doi ra so byte dung theo TfLiteTensorByteSize()
 * cua tensor that, khong tu gia dinh 1 hay 4 byte/phan tu. */
int backend_execute_raw(const void *input, int input_count, void *output, int output_count)
{
    (void)input_count;
    (void)output_count;

    if (!g_backend_ready || input == NULL || output == NULL) {
        return -1;
    }

    TfLiteTensor *in_tensor = TfLiteInterpreterGetInputTensor(g_interpreter, 0);
    if (in_tensor == NULL) {
        return -1;
    }

    size_t input_bytes = TfLiteTensorByteSize(in_tensor);
    if (TfLiteTensorCopyFromBuffer(in_tensor, input, input_bytes) != kTfLiteOk) {
        fprintf(stderr, "backend_tflite_delegate: (raw) TfLiteTensorCopyFromBuffer failed\n");
        return -1;
    }

    if (TfLiteInterpreterInvoke(g_interpreter) != kTfLiteOk) {
        fprintf(stderr, "backend_tflite_delegate: (raw) TfLiteInterpreterInvoke failed\n");
        return -1;
    }

    const TfLiteTensor *out_tensor = TfLiteInterpreterGetOutputTensor(g_interpreter, 0);
    if (out_tensor == NULL) {
        return -1;
    }

    size_t output_bytes = TfLiteTensorByteSize(out_tensor);
    if (TfLiteTensorCopyToBuffer(out_tensor, output, output_bytes) != kTfLiteOk) {
        fprintf(stderr, "backend_tflite_delegate: (raw) TfLiteTensorCopyToBuffer failed\n");
        return -1;
    }

    return 0;
}

int backend_execute_tensors(const dl_tensor_t *inputs, int input_tensor_count,
                            dl_tensor_t *outputs, int output_tensor_count)
{
    if (!g_backend_ready || inputs == NULL || outputs == NULL ||
        input_tensor_count != TfLiteInterpreterGetInputTensorCount(g_interpreter) ||
        output_tensor_count != TfLiteInterpreterGetOutputTensorCount(g_interpreter)) return -1;

    for (int i = 0; i < input_tensor_count; ++i) {
        TfLiteTensor *tensor = TfLiteInterpreterGetInputTensor(g_interpreter, i);
        if (tensor == NULL || inputs[i].data == NULL ||
            inputs[i].byte_size != TfLiteTensorByteSize(tensor) ||
            TfLiteTensorCopyFromBuffer(tensor, inputs[i].data, inputs[i].byte_size) != kTfLiteOk) {
            fprintf(stderr, "backend_tflite_delegate: failed to copy input tensor %d\n", i);
            return -1;
        }
    }
    if (TfLiteInterpreterInvoke(g_interpreter) != kTfLiteOk) {
        fprintf(stderr, "backend_tflite_delegate: multi-tensor Invoke failed\n");
        return -1;
    }
    for (int i = 0; i < output_tensor_count; ++i) {
        const TfLiteTensor *tensor = TfLiteInterpreterGetOutputTensor(g_interpreter, i);
        if (tensor == NULL || outputs[i].data == NULL ||
            outputs[i].byte_size != TfLiteTensorByteSize(tensor) ||
            TfLiteTensorCopyToBuffer(tensor, outputs[i].data, outputs[i].byte_size) != kTfLiteOk) {
            fprintf(stderr, "backend_tflite_delegate: failed to copy output tensor %d\n", i);
            return -1;
        }
    }
    return 0;
}

void backend_deinit(void)
{
    if (g_interpreter != NULL) {
        TfLiteInterpreterDelete(g_interpreter);
        g_interpreter = NULL;
    }
    /* The interpreter may still refer to delegate-owned state, so destroy
     * it before destroying the delegate. */
    if (g_delegate != NULL) {
        TfLiteExternalDelegateDelete(g_delegate);
        g_delegate = NULL;
    }
    if (g_interp_options != NULL) {
        TfLiteInterpreterOptionsDelete(g_interp_options);
        g_interp_options = NULL;
    }
    if (g_model != NULL) {
        TfLiteModelDelete(g_model);
        g_model = NULL;
    }

    g_backend_ready = 0;
    g_using_delegate = 0;
    g_fallback_reason = NULL;
}
