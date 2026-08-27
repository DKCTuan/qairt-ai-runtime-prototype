#ifndef AI_RUNTIME_H
#define AI_RUNTIME_H

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
