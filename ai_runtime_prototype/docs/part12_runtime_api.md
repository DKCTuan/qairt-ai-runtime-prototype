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
