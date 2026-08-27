# 19. Tong ket trang thai va phat hien quan trong ve kien truc

## Phat hien quan trong: spec goc (Muc tieu + Thiet ke module) da la TFLite tu dau

Doc lai phan "Muc tieu" va "Thiet ke module" o dau tai lieu (viet TRUOC
khi bat dau setup SDK o Phan 1):

```text
Input: khong phu thuoc framework, chi coi dau vao la .tflite

Module Runtime: Load model - Allocate Tensor Arena - Create Interpreter
                - Allocate Tensor - Ready

Packaging:
    app/
        libai_runtime.a
        traffic_model.tflite    <- artifact deploy la CHINH file .tflite
```

"Allocate Tensor Arena", "Create Interpreter" la thuat ngu rieng cua
TFLite Interpreter API, khong phai QNN. Artifact deploy trong spec goc la
`.tflite` truc tiep, khong phai `.dlc`/`.bin`.

**Ket luan:** kien truc dang build tu Phan 12-18 (QNN native - convert
thang sang `.dlc`/`.bin`, bo qua TFLite Runtime) la **di lech khoi spec
goc**. Spec goc tu dau da la kien truc TFLite Runtime + QNN Delegate
(giu TFLite Interpreter, QNN chi tang toc mot phan op qua delegate).
Day khong phai loi - phan da build van chay dung va co gia tri (verify
duoc toolchain QAIRT, converter, QNN API hoat dong) - nhung can quyet
dinh ro: tiep tuc huong QNN native hien tai, hay chuyen sang dung TFLite
Delegate cho khop spec goc. Xem muc "Cau hoi dang cho" ben duoi.

## Tom tat da lam duoc (Phan 1-18)

```text
Phan 1-11   Setup toan bo moi truong: WSL2, Python 3.10 + venv, system
            dependencies (va cac lib bi Ubuntu 24.04 go: libncurses5,
            libllvm14...), build + chay thanh cong qnn-sample-app mau
            cua SDK - xac nhan toolchain hoat dong dung.

Phan 12     Verify qairt-converter voi model TFLite mau, fix 2 loi
            (libLLVM-14, thieu package python `tflite`).

Phan 13     Convert traffic_qos_model.onnx -> .dlc, chay bang qnn-net-run
            (CPU backend), so sanh voi ONNX reference: max_abs_diff =
            1.4e-09, argmax khop (=2). Xac nhan buoc convert dung.

Phan 14     Prototype API dau tien (dl_init/dl_inference/dl_deinit) voi
            mock backend, sau do thay bang backend_qnn_cli.c (goi CLI
            qnn-net-run ben trong).

Phan 15-16  Chuyen tu backend CLI (qnn-net-run) sang backend_qnn_api.c
            (goi truc tiep QNN C API, khong qua CLI). Sua bug graph-name
            fallback. tools/model_deploy.py mo rong them build-context/
            run-api/deploy - lenh "mot cua" tu model goc den ket qua.

Phan 16.2   tools/regression_test.py - tu dong chay deploy + assert
            validation_passed cho nhieu model, tranh lap lai bug chi lo
            ra khi test voi 1 model duy nhat.

Phan 17     REST API (model_deploy_server.py) - GHI NHAN: chua theo kip
            CLI, moi chi co /convert /run-host /validate (van qua
            qnn-net-run), CHUA co /deploy hay /run-api (QNN native).

Phan 18     Do latency (CLOCK_MONOTONIC, rieng backend_execute()). Verify
            cross-compile thu sang aarch64 bang toolchain generic Ubuntu
            (gcc-aarch64-linux-gnu) - build ra dung ELF ARM aarch64.
```

## Da xac nhan ve phan cung dich

Thiet bi dich la dong **Qualcomm Networking Pro** (Wi-Fi 7/8, vi du A7
Elite) - CO Hexagon NPU rieng cho AI (40 TOPS), khac voi dong IPQ co ban
(khong co NPU). Chay tren nen **OpenWrt** (Linux nhung, thuong dung musl
libc), KHONG phai Android, KHONG phai Yocto Kirkstone nhu tai lieu
sample_app.html cua dong compute (QCM6490) mo ta.

He qua: toolchain cross-compile generic (`gcc-aarch64-linux-gnu`, dung
glibc) build ra dung kien truc ARM64 nhung CO THE khong chay duoc tren
firmware that (can toolchain rieng di kem SDK/firmware cua thiet bi,
khac libc). Chua co toolchain that de xac nhan.

## Cau hoi dang cho (chan tien do neu khong co cau tra loi)

```text
1. TFLite Runtime + QNN Delegate (dung spec goc) hay QNN native (dang
   build)? -> Da gui cau hoi cho anh, dang cho tra loi. Voi phat hien o
   tren (spec goc la TFLite), kha nang cao can chuyen huong.
2. Toolchain/SDK cross-compile cho dung firmware OpenWrt cua thiet bi -
   chua co, can hoi rieng (khong nhat thiet phai xin muon thiet bi vat
   ly truoc).
```

## Viec lam duoc ngay, KHONG can cho cau tra loi tren

```text
- Sua path model YOLO that trong tools/regression_test.py (dang la
  placeholder) va chay lai de xac nhan bo test pass ca 2 model.
- Neu quyet dinh la QNN native (khong doi): tiep tuc hoan thien
  Nhom A con lai, don dep REST API cho khop CLI (them /deploy).
- Neu quyet dinh la TFLite Delegate: bat dau doc
  examples/QNN/TFLiteDelegate trong QAIRT SDK (da thay trong Phan setup
  SDK, co san san example + Models/) truoc khi viet lai backend.
```