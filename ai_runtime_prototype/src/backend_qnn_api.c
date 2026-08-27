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
#include <stdint.h>
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
                                             const char *graph_name,
                                             char *resolved_graph_name_out,
                                             size_t resolved_graph_name_out_size)
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

    /* BUG FIX: truoc day ham nay chi in log khi fallback sang graphs[0],
     * khong tra ten that ra ngoai - nen backend_init() van goi
     * retrieve_graph() bang ten CU (khong khop), lam QnnGraph_retrieve
     * fail du buoc doc tensor o day da dung dung graph. Gio tra ten that
     * (matched hoac fallback) qua resolved_graph_name_out de backend_init()
     * dung dung ten nay cho retrieve_graph(). */
    if (resolved_graph_name_out != NULL && resolved_graph_name_out_size > 0) {
        const char *actual_name = (target_graph->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3)
            ? target_graph->graphInfoV3.graphName
            : target_graph->graphInfoV1.graphName;
        snprintf(resolved_graph_name_out, resolved_graph_name_out_size, "%s",
                 actual_name != NULL ? actual_name : graph_name);
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

/* Tinh tong so phan tu (float) cua 1 tensor tu rank/dimensions da doc
 * duoc trong read_io_tensors_from_binary_info(). Day la ly do backend nay
 * co the tra ve shape THAT cua bat ky model nao (khac backend_qnn_cli/
 * backend_mock phai khai bao shape thu cong): metadata nay lay truc tiep
 * tu context binary, khong phai gia dinh truoc.
 *
 * LUU Y: field .v1.rank / .v1.dimensions duoc dung dung theo pattern da
 * co san trong file nay (vi du g_input_tensor.v1.clientBuf...). Neu SDK
 * version khac co layout struct khac, doi chieu lai voi QnnTensor.h that
 * trong QAIRT SDK cua ban - mon nay khong compile-test duoc trong moi
 * truong hien tai vi thieu QNN headers. */
static int tensor_element_count(const Qnn_Tensor_t *tensor)
{
    uint32_t rank = tensor->v1.rank;
    const uint32_t *dimensions = tensor->v1.dimensions;

    if (rank == 0 || dimensions == NULL) {
        return -1;
    }

    long count = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        count *= (long)dimensions[i];
    }

    if (count <= 0 || count > INT32_MAX) {
        return -1;
    }
    return (int)count;
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
    char resolved_graph_name[256] = {0};
    int io_ok = read_io_tensors_from_binary_info(binary_buffer, (uint64_t)binary_size, graph_name,
                                                  resolved_graph_name, sizeof(resolved_graph_name)) == 0;
    int ctx_ok = io_ok && create_backend_device_context_from_binary(binary_buffer, binary_size) == 0;

    free(binary_buffer);

    if (!io_ok || !ctx_ok) {
        return -1;
    }

    /* Dung ten graph THAT (co the la ten fallback graphs[0] neu DL_GRAPH_NAME
     * khong khop) - khong dung lai graph_name goc, vi do la nguyen nhan
     * QnnGraph_retrieve fail ngay ca khi doc tensor da thanh cong. */
    if (retrieve_graph(resolved_graph_name[0] != '\0' ? resolved_graph_name : graph_name) != 0) {
        return -1;
    }

    g_backend_ready = 1;
    return 0;
}

int backend_get_io_count(int *input_count, int *output_count)
{
    if (!g_backend_ready) {
        return -1;
    }

    int in_count = tensor_element_count(&g_input_tensor);
    int out_count = tensor_element_count(&g_output_tensor);

    if (in_count <= 0 || out_count <= 0) {
        fprintf(stderr, "backend_qnn_api: failed to compute tensor element count from binary metadata\n");
        return -1;
    }

    if (input_count != NULL) {
        *input_count = in_count;
    }
    if (output_count != NULL) {
        *output_count = out_count;
    }
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
/* THEM MOI (dong bo voi backend.h/tflite_qnn_prototype): backend nay chua
 * ho tro doc dtype dong (chi lam viec voi float32 tu truoc gio), nen tra
 * ve -1 - ai_runtime.c se tu hieu la "khong ho tro", giu nguyen mac dinh
 * DL_DTYPE_FLOAT32 va duong code cu (backend_execute()). KHONG doi hanh
 * vi hien tai cua backend nay. */
int backend_get_io_dtype(dl_tensor_dtype_t *input_dtype, float *input_scale, int *input_zero_point,
                          dl_tensor_dtype_t *output_dtype, float *output_scale, int *output_zero_point)
{
    (void)input_dtype; (void)input_scale; (void)input_zero_point;
    (void)output_dtype; (void)output_scale; (void)output_zero_point;
    return -1;
}

/* Backend nay khong dung duong "raw" (chi co model float32) - khong bao
 * gio duoc goi thuc te vi backend_get_io_dtype() da tra -1 o tren, nhung
 * van dinh nghia de link OK. */
int backend_execute_raw(const void *input, int input_count, void *output, int output_count)
{
    (void)input; (void)input_count; (void)output; (void)output_count;
    return -1;
}
