# AI Runtime Prototype for Qualcomm QAIRT/QNN

This prototype proves the target application API:

```c
dl_init();
int label = dl_inference(input);
dl_deinit();
```

The application does not call QAIRT/QNN directly. It links `libai_runtime.a`; the runtime backend handles model execution.

## Build mock backend

```bash
make clean
make BACKEND=mock run
```

## Build QAIRT/QNN host backend

```bash
./scripts/run_host_qnn.sh
```

Default configuration:

```text
DL_QAIRT_ROOT=/home/congtuan/qairt_sdk/qairt/2.44.0.260225
DL_MODEL_PATH=/home/congtuan/model_test/traffic_qos_model.dlc
DL_INPUT_RAW=/home/congtuan/model_test/input_seq.raw
DL_WORK_DIR=/tmp/ai_runtime_qnn
DL_INPUT_NAME=input_seq
DL_OUTPUT_NAME=class_probs
```

Expected result:

```text
label=2
scores=0.000443353143,0.00221594586,0.997327805,1.24046474e-05,4.26769219e-07
```

Current backend flow:

```text
Application
  -> ai_runtime.h
  -> libai_runtime.a
  -> backend_qnn_cli.c
  -> qnn-net-run
  -> traffic_qos_model.dlc
  -> class_probs.raw
  -> argmax label
```

Next production step: replace the CLI backend with direct QNN C API calls for embedded deployment.

