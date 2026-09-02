#ifndef AI_RUNTIME_H
#define AI_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Gioi han tren chi de bat loi metadata rac (vi du doc nham rank/dimensions
 * sinh ra so am hoac so khong tuong), KHONG con la kich thuoc mang tinh
 * tren stack nhu ban truoc. scores duoc ai_runtime malloc dong theo dung
 * output_count that cua model (xem dl_result_t.scores ben duoi), nen
 * classifier 5 so hay detector nhu YOLOv5 (~151.200 so/output) deu chay
 * duoc ma khong ton bo nho co dinh qua muc hay tran stack.
 * Co the override bang cach define truoc khi include header nay. */
#ifndef DL_MAX_OUTPUT
#define DL_MAX_OUTPUT 2000000
#endif

#define DL_MAX_TENSOR_RANK 8
#define DL_MAX_TENSOR_NAME 128

typedef enum {
    DL_OK = 0,
    DL_ERROR_INVALID_ARGUMENT = -1,
    DL_ERROR_NOT_INITIALIZED = -2,
    DL_ERROR_ALREADY_INITIALIZED = -3,
    DL_ERROR_OUT_OF_MEMORY = -4,
    DL_ERROR_BACKEND_INIT = -5,
    DL_ERROR_BACKEND_EXECUTE = -6,
    DL_ERROR_UNSUPPORTED = -7,
    DL_ERROR_METADATA = -8,
    DL_ERROR_SIZE_MISMATCH = -9,
    DL_ERROR_CLOCK = -10,
    DL_ERROR_BUSY = -11
} dl_status_t;

typedef enum {
    DL_DTYPE_FLOAT32 = 0,
    DL_DTYPE_UINT8 = 1,
    DL_DTYPE_INT8 = 2,
    DL_DTYPE_FLOAT16 = 3,
    DL_DTYPE_UINT16 = 4,
    DL_DTYPE_INT16 = 5,
    DL_DTYPE_UINT32 = 6,
    DL_DTYPE_INT32 = 7,
    DL_DTYPE_UINT64 = 8,
    DL_DTYPE_INT64 = 9,
    DL_DTYPE_FLOAT64 = 10,
    DL_DTYPE_BOOL8 = 11,
    DL_DTYPE_UNKNOWN = 255
} dl_tensor_dtype_t;

typedef struct {
    char name[DL_MAX_TENSOR_NAME];
    dl_tensor_dtype_t dtype;
    int rank;
    uint32_t dimensions[DL_MAX_TENSOR_RANK];
    size_t element_count;
    size_t byte_size;
    float scale;
    int zero_point;
    int quantized_axis;
} dl_tensor_info_t;

typedef struct {
    void *data;
    size_t byte_size;
} dl_tensor_t;

typedef struct {
    int warmup_runs;
    int measured_runs;
} dl_benchmark_config_t;

typedef struct {
    int completed_runs;
    double min_ms;
    double max_ms;
    double mean_ms;
    double p50_ms;
    double p90_ms;
    double p95_ms;
    double p99_ms;
} dl_benchmark_result_t;

typedef struct dl_runtime dl_runtime_t;

/* Instance API: metadata and execution support any positive number of input
 * and output tensors exposed by the selected backend. The current backends
 * remain singleton-owned, so only one dl_runtime_t may exist at a time. */
const char *dl_status_string(dl_status_t status);
dl_status_t dl_runtime_create(dl_runtime_t **runtime);
void dl_runtime_destroy(dl_runtime_t *runtime);
dl_status_t dl_runtime_get_tensor_count(const dl_runtime_t *runtime,
                                        int *input_tensor_count,
                                        int *output_tensor_count);
dl_status_t dl_runtime_get_input_info(const dl_runtime_t *runtime, int index,
                                      dl_tensor_info_t *info);
dl_status_t dl_runtime_get_output_info(const dl_runtime_t *runtime, int index,
                                       dl_tensor_info_t *info);
dl_status_t dl_runtime_execute(dl_runtime_t *runtime,
                               const dl_tensor_t *inputs, int input_tensor_count,
                               dl_tensor_t *outputs, int output_tensor_count,
                               double *latency_ms);
dl_status_t dl_runtime_benchmark(dl_runtime_t *runtime,
                                 const dl_tensor_t *inputs, int input_tensor_count,
                                 dl_tensor_t *outputs, int output_tensor_count,
                                 const dl_benchmark_config_t *config,
                                 dl_benchmark_result_t *result);

typedef struct {
    int label;
    /* Con tro toi buffer scores do ai_runtime cap phat NOI BO (malloc 1 lan
     * trong dl_init(), kich thuoc = output_count that cua model). Con tro
     * nay CHI hop le den lan dl_inference_ex() ke tiep hoac den khi
     * dl_deinit() duoc goi - ung dung KHONG duoc tu free() con tro nay,
     * va khong duoc giu lai sau dl_deinit(). Neu can giu ket qua lau hon,
     * ung dung tu copy sang buffer rieng cua minh. */
    const float *scores;
    int score_count;
    /* Thoi gian CHI RIENG backend_execute() (khong tinh malloc/argmax/parse
     * o lop tren), tinh bang millisecond, do bang CLOCK_MONOTONIC. Muc dich
     * la so sanh CPU vs HTP sau nay - neu can do ca overhead cua dl_inference_ex
     * (memcpy/argmax), do rieng o phia caller (ngoai ham nay). */
    double latency_ms;
} dl_result_t;

/* Legacy single-model float API. New applications should prefer the
 * instance-based dl_runtime_* API above. */
int dl_init(void);

/* Goi sau dl_init() thanh cong, truoc khi cap phat buffer input va goi
 * dl_inference_ex(). Kich thuoc input/output duoc backend xac dinh dong
 * theo model that (vi du: doc tu metadata cua context binary .bin trong
 * backend_qnn_api), khong con la macro co dinh nhu ban truoc. */
int dl_get_io_count(int *input_count, int *output_count);

int dl_inference(const float *input);
int dl_inference_ex(const float *input, dl_result_t *result);
void dl_deinit(void);

#ifdef __cplusplus
}
#endif

#endif
