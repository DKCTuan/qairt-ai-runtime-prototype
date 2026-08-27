#ifndef BACKEND_H
#define BACKEND_H

int backend_init(void);

/* Tra ve so luong phan tu (float) cua input/output tensor cua model dang
 * duoc backend nay quan ly. Phai goi SAU backend_init() thanh cong.
 * Backend nao biet shape that cua model (vi du qnn_api doc tu context
 * binary metadata) thi tra ve gia tri chinh xac; backend nao khong the
 * tu suy ra shape (mock, qnn_cli) thi tra ve gia tri cau hinh san/qua
 * env var - xem comment trong tung file backend_*.c.
 * Return 0 neu thanh cong, -1 neu that bai. */
int backend_get_io_count(int *input_count, int *output_count);

int backend_execute(const float *input, int input_count, float *output, int output_count);
void backend_deinit(void);

/* ===== THEM MOI: tong quat hoa dtype (khong xoa API cu o tren) =====
 *
 * Phat hien duoc tu vu test model paddy (uint8, quantized): API
 * backend_execute(const float*) o tren gia dinh NGAM DINH moi model deu
 * float32. Nhieu model TFLite that (dac biet model quantize de chay
 * nhanh tren NPU/HTP) dung UINT8/INT8 cho input/output, kem theo tham so
 * quantization (scale, zero_point) de doi ve gia tri thuc.
 *
 * Cac ham duoi day la MO RONG, khong bat buoc: backend nao chua ho tro
 * (mock, qnn_cli, qnn_api hien tai) co the KHONG dinh nghia chung - phia
 * ai_runtime.c se goi backend_get_io_dtype() truoc, neu backend khong co
 * ham nay (hoac tra ve -1) thi mac dinh coi la DL_DTYPE_FLOAT32, dung
 * nguyen duong code cu (backend_execute) - hanh vi cu KHONG doi. */

typedef enum {
    DL_DTYPE_FLOAT32 = 0,
    DL_DTYPE_UINT8   = 1,
    DL_DTYPE_INT8    = 2
} dl_tensor_dtype_t;

/* Tra ve dtype that + tham so quantization (chi co y nghia neu dtype la
 * UINT8/INT8) cua input va output tensor. Model khong quantize (float32
 * thuan) thi scale=1.0f, zero_point=0 (khong dung toi).
 * Return 0 neu backend nay biet tra loi cau hoi nay, -1 neu backend
 * khong ho tro (ai_runtime.c se tu dong fallback ve gia dinh float32). */
int backend_get_io_dtype(dl_tensor_dtype_t *input_dtype, float *input_scale, int *input_zero_point,
                          dl_tensor_dtype_t *output_dtype, float *output_scale, int *output_zero_point);

/* Phien ban "raw" cua backend_execute: input/output la byte tho dung dtype
 * that cua tensor (vi du uint8 cho model quantize), KHONG phai lam tron ve
 * float. input_count/output_count van la SO PHAN TU (khong phai so byte).
 * Chi can dinh nghia neu backend_get_io_dtype() co the tra ve dtype khac
 * FLOAT32; neu model la float32 thuan thi cu dung backend_execute() cu la
 * du, khong can ham nay. Return 0 neu thanh cong. */
int backend_execute_raw(const void *input, int input_count, void *output, int output_count);

#endif
