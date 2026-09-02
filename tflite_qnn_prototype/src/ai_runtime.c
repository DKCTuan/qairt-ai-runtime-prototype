#define _POSIX_C_SOURCE 200809L

#include "ai_runtime.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "backend.h"
#include <math.h>

static int g_initialized = 0;
static int g_input_count = 0;
static int g_output_count = 0;
static float *g_scores_buf = NULL;

/* ===== THEM MOI: tong quat hoa dtype (uint8/int8 quantized) =====
 * Phat hien tu vu test model paddy: model quantize dung UINT8 cho input/
 * output, khac model traffic (float32 thuan). API cong khai dl_inference*
 * o duoi KHONG DOI (van nhan/tra float*) - phan nay chi la noi bo, tu
 * quyet dinh co can quantize/dequantize hay khong dua vao
 * backend_get_io_dtype(). Neu backend khong ho tro ham do (mock, qnn_cli,
 * qnn_api hien tai), g_input_dtype/g_output_dtype giu nguyen FLOAT32 va
 * toan bo code chay y het ban truoc - khong doi hanh vi cu. */
static dl_tensor_dtype_t g_input_dtype  = DL_DTYPE_FLOAT32;
static dl_tensor_dtype_t g_output_dtype = DL_DTYPE_FLOAT32;
static float g_input_scale  = 1.0f;
static float g_output_scale = 1.0f;
static int   g_input_zero_point  = 0;
static int   g_output_zero_point = 0;
/* Buffer byte tho (1 byte/phan tu) dung khi dtype la UINT8/INT8. Chi cap
 * phat neu thuc su can (dtype != FLOAT32) - xem dl_init(). */
static void *g_raw_input_buf  = NULL;
static void *g_raw_output_buf = NULL;

struct dl_runtime {
    unsigned int magic;
    int input_tensor_count;
    int output_tensor_count;
    dl_tensor_info_t *input_infos;
    dl_tensor_info_t *output_infos;
};

#define DL_RUNTIME_MAGIC 0x444C5254u
static dl_runtime_t *g_runtime_owner = NULL;

static int monotonic_time_ms(double *time_ms)
{
    struct timespec timestamp;
    if (time_ms == NULL || clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) {
        return -1;
    }
    *time_ms = (double)timestamp.tv_sec * 1000.0
             + (double)timestamp.tv_nsec / 1000000.0;
    return 0;
}

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
    if (g_runtime_owner != NULL) return -1;
    if (g_initialized) {
        return 0;
    }

    if (backend_init() != 0) {
        backend_deinit();
        return -1;
    }

    if (backend_get_io_count(&g_input_count, &g_output_count) != 0) {
        fprintf(stderr, "ai_runtime: backend_get_io_count failed\n");
        backend_deinit();
        return -1;
    }

    if (g_input_count <= 0) {
        fprintf(stderr, "ai_runtime: invalid model input count: %d\n", g_input_count);
        backend_deinit();
        return -1;
    }

    if (g_output_count <= 0 || g_output_count > DL_MAX_OUTPUT) {
        fprintf(stderr,
                "ai_runtime: model output count %d does not fit DL_MAX_OUTPUT (%d); "
                "rebuild with a higher DL_MAX_OUTPUT if this model is expected\n",
                g_output_count, DL_MAX_OUTPUT);
        backend_deinit();
        return -1;
    }

    g_scores_buf = (float *)malloc(sizeof(float) * (size_t)g_output_count);
    if (g_scores_buf == NULL) {
        fprintf(stderr, "ai_runtime: failed to allocate output buffer (%d floats)\n", g_output_count);
        backend_deinit();
        return -1;
    }

    /* THEM MOI: hoi backend xem model that su dung dtype gi. Neu backend
     * khong dinh nghia backend_get_io_dtype() (mock/qnn_cli/qnn_api hien
     * tai) thi ham nay khong ton tai o link-time doi voi backend do - vi
     * vay chi backend_tflite_delegate.c (noi da them ham nay) moi kich
     * hoat duong nay; cac backend khac se... (xem LUU Y duoi). */
    dl_tensor_dtype_t in_dtype = DL_DTYPE_FLOAT32;
    dl_tensor_dtype_t out_dtype = DL_DTYPE_FLOAT32;
    float in_scale = 1.0f, out_scale = 1.0f;
    int in_zp = 0, out_zp = 0;

    if (backend_get_io_dtype(&in_dtype, &in_scale, &in_zp, &out_dtype, &out_scale, &out_zp) == 0) {
        g_input_dtype = in_dtype;
        g_output_dtype = out_dtype;
        g_input_scale = in_scale;
        g_output_scale = out_scale;
        g_input_zero_point = in_zp;
        g_output_zero_point = out_zp;
    }
    if ((g_input_dtype != DL_DTYPE_FLOAT32 && g_input_dtype != DL_DTYPE_UINT8 &&
         g_input_dtype != DL_DTYPE_INT8) ||
        (g_output_dtype != DL_DTYPE_FLOAT32 && g_output_dtype != DL_DTYPE_UINT8 &&
         g_output_dtype != DL_DTYPE_INT8)) {
        fprintf(stderr, "ai_runtime: legacy API supports only float32/uint8/int8; use dl_runtime_* raw tensors\n");
        free(g_scores_buf); g_scores_buf = NULL; backend_deinit(); return -1;
    }
    /* LUU Y QUAN TRONG: ban ai_runtime.c nay CHI duoc sua trong thu muc
     * tflite_qnn_prototype/ (noi Makefile chi ho tro BACKEND=tflite_delegate
     * - xem $(error) trong Makefile). File backend_tflite_delegate.c da
     * duoc them backend_get_io_dtype()/backend_execute_raw() o tren nen
     * link OK.
     * Ban ai_runtime.c ben ai_runtime_prototype/ (dung cho backend_mock,
     * backend_qnn_cli, backend_qnn_api) KHONG bi dong tren cung mot file -
     * no van la ban CU, chua goi backend_get_io_dtype(), nen KHONG bi loi
     * linker. Neu sau nay ban muon hop nhat 2 prototype lam mot, hoac dung
     * ai_runtime.c ban moi nay cho ca qnn_api/qnn_cli/mock, thi luc do moi
     * can them backend_get_io_dtype() tra ve -1 (mac dinh float32) vao
     * backend_mock.c/backend_qnn_cli.c/backend_qnn_api.c truoc. */

    if (g_input_dtype == DL_DTYPE_UINT8 || g_input_dtype == DL_DTYPE_INT8) {
        g_raw_input_buf = malloc((size_t)g_input_count);
        if (g_raw_input_buf == NULL) {
            fprintf(stderr, "ai_runtime: failed to allocate raw input buffer (%d bytes)\n", g_input_count);
            free(g_scores_buf);
            g_scores_buf = NULL;
            backend_deinit();
            return -1;
        }
    }
    if (g_output_dtype == DL_DTYPE_UINT8 || g_output_dtype == DL_DTYPE_INT8) {
        g_raw_output_buf = malloc((size_t)g_output_count);
        if (g_raw_output_buf == NULL) {
            fprintf(stderr, "ai_runtime: failed to allocate raw output buffer (%d bytes)\n", g_output_count);
            free(g_raw_input_buf);
            g_raw_input_buf = NULL;
            free(g_scores_buf);
            g_scores_buf = NULL;
            backend_deinit();
            return -1;
        }
    }

    fprintf(stderr, "ai_runtime: input_dtype=%s output_dtype=%s\n",
            g_input_dtype == DL_DTYPE_FLOAT32 ? "float32" : (g_input_dtype == DL_DTYPE_UINT8 ? "uint8" : "int8"),
            g_output_dtype == DL_DTYPE_FLOAT32 ? "float32" : (g_output_dtype == DL_DTYPE_UINT8 ? "uint8" : "int8"));

    g_initialized = 1;
    return 0;
}

int dl_get_io_count(int *input_count, int *output_count)
{
    if (!g_initialized) {
        return -1;
    }

    if (input_count != NULL) {
        *input_count = g_input_count;
    }
    if (output_count != NULL) {
        *output_count = g_output_count;
    }
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

/* THEM MOI: quantize 1 gia tri float "that" thanh so nguyen theo dung
 * cong thuc chuan cua TFLite: quantized = round(real/scale) + zero_point,
 * clamp ve dung range cua kieu (0..255 cho uint8, -128..127 cho int8).
 * Caller van dua float vao (vi du gia tri pixel 0..255 hoac feature that),
 * ham nay tu doi sang byte tho de dua cho backend_execute_raw(). */
static uint8_t quantize_to_uint8(float real_value, float scale, int zero_point)
{
    long q = lroundf(real_value / scale) + zero_point;
    if (q < 0) q = 0;
    if (q > 255) q = 255;
    return (uint8_t)q;
}

static int8_t quantize_to_int8(float real_value, float scale, int zero_point)
{
    long q = lroundf(real_value / scale) + zero_point;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    return (int8_t)q;
}

static float dequantize_uint8(uint8_t q, float scale, int zero_point)
{
    return scale * ((float)q - (float)zero_point);
}

static float dequantize_int8(int8_t q, float scale, int zero_point)
{
    return scale * ((float)q - (float)zero_point);
}

int dl_inference_ex(const float *input, dl_result_t *result)
{
    if (!g_initialized || input == NULL || result == NULL) {
        return -1;
    }

    /* CLOCK_MONOTONIC: khong bi anh huong boi chinh gio he thong (NTP sync,
     * user doi gio tay...), dung chuan de do khoang thoi gian. Chi bao
     * quanh dung phan backend_execute*() - khong tinh malloc o dl_init() hay
     * argmax()/quantize/dequantize ben duoi, vi muc dich la so sanh thoi
     * gian INFERENCE THAT giua cac backend (CPU/HTP), khong phai tong thoi
     * gian ca ham. */
    double start_ms = 0.0;
    double end_ms = 0.0;
    if (monotonic_time_ms(&start_ms) != 0) {
        fprintf(stderr, "ai_runtime: CLOCK_MONOTONIC start failed\n");
        return -1;
    }

    int exec_status;

    if (g_input_dtype == DL_DTYPE_FLOAT32 && g_output_dtype == DL_DTYPE_FLOAT32) {
        /* DUONG CU, GIU NGUYEN Y HET BAN TRUOC - khong doi hanh vi cho
         * model float32 (vi du traffic_qos_model). */
        exec_status = backend_execute(input, g_input_count, g_scores_buf, g_output_count);
    } else {
        /* DUONG MOI: it nhat 1 trong 2 dau la quantized -> quantize input
         * (neu can), goi backend_execute_raw(), roi dequantize output
         * (neu can) - nhung van tra ve g_scores_buf dang float cho caller,
         * API cong khai khong doi. */
        const void *raw_input_ptr;
        if (g_input_dtype == DL_DTYPE_UINT8) {
            uint8_t *buf = (uint8_t *)g_raw_input_buf;
            for (int i = 0; i < g_input_count; ++i) {
                buf[i] = quantize_to_uint8(input[i], g_input_scale, g_input_zero_point);
            }
            raw_input_ptr = g_raw_input_buf;
        } else if (g_input_dtype == DL_DTYPE_INT8) {
            int8_t *buf = (int8_t *)g_raw_input_buf;
            for (int i = 0; i < g_input_count; ++i) {
                buf[i] = quantize_to_int8(input[i], g_input_scale, g_input_zero_point);
            }
            raw_input_ptr = g_raw_input_buf;
        } else {
            /* Input van la float32 nhung output moi quantized - dung
             * thang input goc, khong can buffer rieng. */
            raw_input_ptr = input;
        }

        void *raw_output_ptr = (g_output_dtype != DL_DTYPE_FLOAT32) ? g_raw_output_buf : (void *)g_scores_buf;

        exec_status = backend_execute_raw(raw_input_ptr, g_input_count, raw_output_ptr, g_output_count);

        if (exec_status == 0 && g_output_dtype != DL_DTYPE_FLOAT32) {
            if (g_output_dtype == DL_DTYPE_UINT8) {
                const uint8_t *buf = (const uint8_t *)g_raw_output_buf;
                for (int i = 0; i < g_output_count; ++i) {
                    g_scores_buf[i] = dequantize_uint8(buf[i], g_output_scale, g_output_zero_point);
                }
            } else {
                const int8_t *buf = (const int8_t *)g_raw_output_buf;
                for (int i = 0; i < g_output_count; ++i) {
                    g_scores_buf[i] = dequantize_int8(buf[i], g_output_scale, g_output_zero_point);
                }
            }
        }
    }

    if (monotonic_time_ms(&end_ms) != 0) {
        fprintf(stderr, "ai_runtime: CLOCK_MONOTONIC end failed\n");
        return -1;
    }

    if (exec_status != 0) {
        return -1;
    }

    result->scores = g_scores_buf;
    result->score_count = g_output_count;
    result->label = argmax(g_scores_buf, g_output_count);
    result->latency_ms = end_ms - start_ms;
    return 0;
}

void dl_deinit(void)
{
    if (!g_initialized) {
        return;
    }

    backend_deinit();
    free(g_scores_buf);
    g_scores_buf = NULL;
    /* THEM MOI: don dep buffer raw uint8/int8 neu co cap phat o dl_init(). */
    free(g_raw_input_buf);
    g_raw_input_buf = NULL;
    free(g_raw_output_buf);
    g_raw_output_buf = NULL;
    g_input_dtype = DL_DTYPE_FLOAT32;
    g_output_dtype = DL_DTYPE_FLOAT32;
    g_input_count = 0;
    g_output_count = 0;
    g_initialized = 0;
}

const char *dl_status_string(dl_status_t status)
{
    switch (status) {
        case DL_OK: return "success";
        case DL_ERROR_INVALID_ARGUMENT: return "invalid argument";
        case DL_ERROR_NOT_INITIALIZED: return "runtime not initialized";
        case DL_ERROR_ALREADY_INITIALIZED: return "runtime already initialized";
        case DL_ERROR_OUT_OF_MEMORY: return "out of memory";
        case DL_ERROR_BACKEND_INIT: return "backend initialization failed";
        case DL_ERROR_BACKEND_EXECUTE: return "backend execution failed";
        case DL_ERROR_UNSUPPORTED: return "operation or tensor layout unsupported";
        case DL_ERROR_METADATA: return "invalid tensor metadata";
        case DL_ERROR_SIZE_MISMATCH: return "tensor buffer size mismatch";
        case DL_ERROR_CLOCK: return "monotonic clock failed";
        case DL_ERROR_BUSY: return "backend already owned by another runtime";
        default: return "unknown error";
    }
}

dl_status_t dl_runtime_create(dl_runtime_t **runtime)
{
    if (runtime == NULL) return DL_ERROR_INVALID_ARGUMENT;
    *runtime = NULL;
    if (g_runtime_owner != NULL || g_initialized) return DL_ERROR_BUSY;
    if (backend_init() != 0) { backend_deinit(); return DL_ERROR_BACKEND_INIT; }
    dl_runtime_t *created = (dl_runtime_t *)calloc(1, sizeof(*created));
    if (created == NULL) { backend_deinit(); return DL_ERROR_OUT_OF_MEMORY; }
    created->magic = DL_RUNTIME_MAGIC;
    if (backend_get_tensor_count(&created->input_tensor_count,
                                 &created->output_tensor_count) != 0 ||
        created->input_tensor_count <= 0 || created->output_tensor_count <= 0) {
        backend_deinit(); free(created); return DL_ERROR_METADATA;
    }
    created->input_infos = (dl_tensor_info_t *)calloc((size_t)created->input_tensor_count,
                                                       sizeof(*created->input_infos));
    created->output_infos = (dl_tensor_info_t *)calloc((size_t)created->output_tensor_count,
                                                        sizeof(*created->output_infos));
    if (created->input_infos == NULL || created->output_infos == NULL) {
        free(created->input_infos); free(created->output_infos);
        backend_deinit(); free(created); return DL_ERROR_OUT_OF_MEMORY;
    }
    for (int i = 0; i < created->input_tensor_count; ++i) {
        if (backend_get_tensor_info(1, i, &created->input_infos[i]) != 0) {
            free(created->input_infos); free(created->output_infos);
            backend_deinit(); free(created); return DL_ERROR_METADATA;
        }
    }
    for (int i = 0; i < created->output_tensor_count; ++i) {
        if (backend_get_tensor_info(0, i, &created->output_infos[i]) != 0) {
            free(created->input_infos); free(created->output_infos);
            backend_deinit(); free(created); return DL_ERROR_METADATA;
        }
    }
    g_runtime_owner = created;
    *runtime = created;
    return DL_OK;
}

void dl_runtime_destroy(dl_runtime_t *runtime)
{
    if (runtime == NULL || runtime->magic != DL_RUNTIME_MAGIC || runtime != g_runtime_owner) return;
    runtime->magic = 0;
    g_runtime_owner = NULL;
    backend_deinit();
    free(runtime->input_infos);
    free(runtime->output_infos);
    free(runtime);
}

static int valid_runtime(const dl_runtime_t *runtime)
{
    return runtime != NULL && runtime == g_runtime_owner && runtime->magic == DL_RUNTIME_MAGIC;
}

dl_status_t dl_runtime_get_tensor_count(const dl_runtime_t *runtime, int *inputs, int *outputs)
{
    if (!valid_runtime(runtime)) return DL_ERROR_NOT_INITIALIZED;
    if (inputs == NULL || outputs == NULL) return DL_ERROR_INVALID_ARGUMENT;
    *inputs = runtime->input_tensor_count;
    *outputs = runtime->output_tensor_count;
    return DL_OK;
}

dl_status_t dl_runtime_get_input_info(const dl_runtime_t *runtime, int index, dl_tensor_info_t *info)
{
    if (!valid_runtime(runtime)) return DL_ERROR_NOT_INITIALIZED;
    if (index < 0 || index >= runtime->input_tensor_count || info == NULL) return DL_ERROR_INVALID_ARGUMENT;
    *info = runtime->input_infos[index]; return DL_OK;
}

dl_status_t dl_runtime_get_output_info(const dl_runtime_t *runtime, int index, dl_tensor_info_t *info)
{
    if (!valid_runtime(runtime)) return DL_ERROR_NOT_INITIALIZED;
    if (index < 0 || index >= runtime->output_tensor_count || info == NULL) return DL_ERROR_INVALID_ARGUMENT;
    *info = runtime->output_infos[index]; return DL_OK;
}

dl_status_t dl_runtime_execute(dl_runtime_t *runtime,
                               const dl_tensor_t *inputs, int input_count,
                               dl_tensor_t *outputs, int output_count,
                               double *latency_ms)
{
    if (!valid_runtime(runtime)) return DL_ERROR_NOT_INITIALIZED;
    if (inputs == NULL || outputs == NULL || input_count != runtime->input_tensor_count ||
        output_count != runtime->output_tensor_count) return DL_ERROR_INVALID_ARGUMENT;
    for (int i = 0; i < input_count; ++i) {
        if (inputs[i].data == NULL) return DL_ERROR_INVALID_ARGUMENT;
        if (inputs[i].byte_size != runtime->input_infos[i].byte_size) return DL_ERROR_SIZE_MISMATCH;
    }
    for (int i = 0; i < output_count; ++i) {
        if (outputs[i].data == NULL) return DL_ERROR_INVALID_ARGUMENT;
        if (outputs[i].byte_size != runtime->output_infos[i].byte_size) return DL_ERROR_SIZE_MISMATCH;
    }
    double start_ms = 0.0, end_ms = 0.0;
    if (monotonic_time_ms(&start_ms) != 0) return DL_ERROR_CLOCK;
    int status = backend_execute_tensors(inputs, input_count, outputs, output_count);
    if (monotonic_time_ms(&end_ms) != 0) return DL_ERROR_CLOCK;
    if (status != 0) return DL_ERROR_BACKEND_EXECUTE;
    if (latency_ms != NULL) *latency_ms = end_ms - start_ms;
    return DL_OK;
}

static int compare_double(const void *left, const void *right)
{
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

static double percentile(const double *sorted, int count, double fraction)
{
    int index = (int)(fraction * (double)(count - 1) + 0.5);
    return sorted[index];
}

dl_status_t dl_runtime_benchmark(dl_runtime_t *runtime,
                                 const dl_tensor_t *inputs, int input_count,
                                 dl_tensor_t *outputs, int output_count,
                                 const dl_benchmark_config_t *config,
                                 dl_benchmark_result_t *result)
{
    if (!valid_runtime(runtime)) return DL_ERROR_NOT_INITIALIZED;
    if (config == NULL || result == NULL || config->warmup_runs < 0 || config->measured_runs <= 0)
        return DL_ERROR_INVALID_ARGUMENT;
    for (int i = 0; i < config->warmup_runs; ++i) {
        dl_status_t status = dl_runtime_execute(runtime, inputs, input_count, outputs, output_count, NULL);
        if (status != DL_OK) return status;
    }
    double *samples = (double *)malloc(sizeof(*samples) * (size_t)config->measured_runs);
    if (samples == NULL) return DL_ERROR_OUT_OF_MEMORY;
    double sum = 0.0;
    for (int i = 0; i < config->measured_runs; ++i) {
        dl_status_t status = dl_runtime_execute(runtime, inputs, input_count, outputs, output_count, &samples[i]);
        if (status != DL_OK) { free(samples); return status; }
        sum += samples[i];
    }
    qsort(samples, (size_t)config->measured_runs, sizeof(*samples), compare_double);
    result->completed_runs = config->measured_runs;
    result->min_ms = samples[0]; result->max_ms = samples[config->measured_runs - 1];
    result->mean_ms = sum / (double)config->measured_runs;
    result->p50_ms = percentile(samples, config->measured_runs, 0.50);
    result->p90_ms = percentile(samples, config->measured_runs, 0.90);
    result->p95_ms = percentile(samples, config->measured_runs, 0.95);
    result->p99_ms = percentile(samples, config->measured_runs, 0.99);
    free(samples);
    return DL_OK;
}
