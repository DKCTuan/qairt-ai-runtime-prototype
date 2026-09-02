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

/* qnn_cli backend chay qnn-net-run tren 1 file .dlc va KHONG tu doc duoc
 * shape input/output cua model (khac backend_qnn_api, backend nay doc
 * duoc shape that tu metadata cua context binary). Vi vay shape phai
 * duoc khai bao qua DL_INPUT_COUNT/DL_OUTPUT_COUNT; mac dinh fallback ve
 * shape cua traffic_qos_model (240/5) de tuong thich nguoc voi cac lenh
 * da validate truoc do. Neu doi model, PHAI set 2 bien env nay. */
#define DEFAULT_INPUT_COUNT 240
#define DEFAULT_OUTPUT_COUNT 5

/* Target triplet trong SDK (vi du "x86_64-linux-clang" tren host, hoac
 * "aarch64-android"/"aarch64-oe-linux-gcc11.2" khi cross-compile cho
 * device nhung that). Ten thu vien backend cung parametrize duoc de
 * chon cpu/gpu/htp ma khong phai sua code. */
#define DEFAULT_QAIRT_TARGET "x86_64-linux-clang"
#define DEFAULT_BACKEND_LIB_NAME "libQnnCpu.so"

static int g_backend_ready = 0;
static char g_qairt_root[1024];
static char g_model_path[1024];
static char g_work_dir[1024];
static char g_input_name[128];
static char g_output_name[128];
static char g_qairt_target[128];
static char g_backend_lib_name[128];
static int g_input_count = 0;
static int g_output_count = 0;

static const char *env_or_default(const char *name, const char *fallback)
{
    const char *value = getenv(name);
    return (value != NULL && value[0] != '\0') ? value : fallback;
}

static int env_or_int(const char *name, int fallback)
{
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    int parsed = atoi(value);
    return parsed > 0 ? parsed : fallback;
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
    g_input_count = env_or_int("DL_INPUT_COUNT", DEFAULT_INPUT_COUNT);
    g_output_count = env_or_int("DL_OUTPUT_COUNT", DEFAULT_OUTPUT_COUNT);
    copy_string(g_qairt_target, sizeof(g_qairt_target), env_or_default("DL_QAIRT_TARGET", DEFAULT_QAIRT_TARGET));
    copy_string(g_backend_lib_name, sizeof(g_backend_lib_name), env_or_default("DL_QNN_BACKEND_LIB_NAME", DEFAULT_BACKEND_LIB_NAME));

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

int backend_get_io_count(int *input_count, int *output_count)
{
    if (!g_backend_ready) {
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

int backend_get_tensor_count(int *input_tensor_count, int *output_tensor_count)
{
    if (!g_backend_ready || input_tensor_count == NULL || output_tensor_count == NULL) return -1;
    *input_tensor_count = 1;
    *output_tensor_count = 1;
    return 0;
}

int backend_get_tensor_info(int is_input, int index, dl_tensor_info_t *info)
{
    if (!g_backend_ready || index != 0 || info == NULL) return -1;
    int count = is_input ? g_input_count : g_output_count;
    memset(info, 0, sizeof(*info));
    snprintf(info->name, sizeof(info->name), "%s", is_input ? g_input_name : g_output_name);
    info->dtype = DL_DTYPE_FLOAT32;
    info->rank = 1;
    info->dimensions[0] = (uint32_t)count;
    info->element_count = (size_t)count;
    info->byte_size = (size_t)count * sizeof(float);
    info->scale = 1.0f;
    info->quantized_axis = -1;
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
    snprintf(qnn_net_run, sizeof(qnn_net_run), "%s/bin/%s/qnn-net-run", g_qairt_root, g_qairt_target);
    snprintf(backend_lib, sizeof(backend_lib), "%s/lib/%s/%s", g_qairt_root, g_qairt_target, g_backend_lib_name);
    snprintf(model_dlc_lib, sizeof(model_dlc_lib), "%s/lib/%s/libQnnModelDlc.so", g_qairt_root, g_qairt_target);
    snprintf(lib_dir, sizeof(lib_dir), "%s/lib/%s", g_qairt_root, g_qairt_target);
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
    return backend_execute((const float *)input, input_count,
                           (float *)output, output_count);
}

int backend_execute_tensors(const dl_tensor_t *inputs, int input_tensor_count,
                            dl_tensor_t *outputs, int output_tensor_count)
{
    if (inputs == NULL || outputs == NULL || input_tensor_count != 1 || output_tensor_count != 1 ||
        inputs[0].byte_size != (size_t)g_input_count * sizeof(float) ||
        outputs[0].byte_size != (size_t)g_output_count * sizeof(float)) return -1;
    return backend_execute((const float *)inputs[0].data, g_input_count,
                           (float *)outputs[0].data, g_output_count);
}
