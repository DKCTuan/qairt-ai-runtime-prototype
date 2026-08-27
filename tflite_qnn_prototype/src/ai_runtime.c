#include "ai_runtime.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

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

    clock_gettime(CLOCK_MONOTONIC, &t_end);

    if (exec_status != 0) {
        return -1;
    }

    double elapsed_ms = (double)(t_end.tv_sec - t_start.tv_sec) * 1000.0
                       + (double)(t_end.tv_nsec - t_start.tv_nsec) / 1e6;

    result->scores = g_scores_buf;
    result->score_count = g_output_count;
    result->label = argmax(g_scores_buf, g_output_count);
    result->latency_ms = elapsed_ms;
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
