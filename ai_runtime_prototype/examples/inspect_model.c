#include <stdio.h>

#include "ai_runtime.h"

static const char *dtype_name(dl_tensor_dtype_t dtype)
{
    switch (dtype) {
        case DL_DTYPE_FLOAT32: return "float32";
        case DL_DTYPE_UINT8: return "uint8";
        case DL_DTYPE_INT8: return "int8";
        case DL_DTYPE_FLOAT16: return "float16";
        case DL_DTYPE_UINT16: return "uint16";
        case DL_DTYPE_INT16: return "int16";
        case DL_DTYPE_UINT32: return "uint32";
        case DL_DTYPE_INT32: return "int32";
        case DL_DTYPE_UINT64: return "uint64";
        case DL_DTYPE_INT64: return "int64";
        case DL_DTYPE_FLOAT64: return "float64";
        case DL_DTYPE_BOOL8: return "bool8";
        default: return "unknown";
    }
}

static void print_json_string(const char *value)
{
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)value; *p != '\0'; ++p) {
        if (*p == '"' || *p == '\\') { putchar('\\'); putchar(*p); }
        else if (*p >= 0x20) putchar(*p);
    }
    putchar('"');
}

static int print_tensors(dl_runtime_t *runtime, int is_input, int count)
{
    for (int i = 0; i < count; ++i) {
        dl_tensor_info_t info;
        dl_status_t status = is_input ? dl_runtime_get_input_info(runtime, i, &info)
                                      : dl_runtime_get_output_info(runtime, i, &info);
        if (status != DL_OK) return -1;
        if (i != 0) putchar(',');
        printf("{\"index\":%d,\"name\":", i);
        print_json_string(info.name);
        printf(",\"dtype\":\"%s\",\"shape\":[", dtype_name(info.dtype));
        for (int d = 0; d < info.rank; ++d) printf("%s%u", d == 0 ? "" : ",", info.dimensions[d]);
        printf("],\"element_count\":%zu,\"byte_size\":%zu,"
               "\"scale\":%.9g,\"zero_point\":%d,\"quantized_axis\":%d}",
               info.element_count, info.byte_size, info.scale, info.zero_point, info.quantized_axis);
    }
    return 0;
}

int main(void)
{
    dl_runtime_t *runtime = NULL;
    dl_status_t status = dl_runtime_create(&runtime);
    if (status != DL_OK) {
        fprintf(stderr, "inspect_model: %s\n", dl_status_string(status));
        return 1;
    }
    int inputs = 0, outputs = 0;
    status = dl_runtime_get_tensor_count(runtime, &inputs, &outputs);
    if (status != DL_OK) {
        fprintf(stderr, "inspect_model: %s\n", dl_status_string(status));
        dl_runtime_destroy(runtime);
        return 1;
    }
    printf("{\"input_count\":%d,\"output_count\":%d,\"inputs\":[", inputs, outputs);
    if (print_tensors(runtime, 1, inputs) != 0) { dl_runtime_destroy(runtime); return 1; }
    printf("],\"outputs\":[");
    if (print_tensors(runtime, 0, outputs) != 0) { dl_runtime_destroy(runtime); return 1; }
    printf("]}\n");
    dl_runtime_destroy(runtime);
    return 0;
}
