CC ?= gcc
AR ?= ar
BACKEND ?= tflite_delegate

CFLAGS ?= -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L -Iinclude -Isrc

ARCH ?= x86_64

BUILD_DIR := build/$(BACKEND)-$(ARCH)
LIB := $(BUILD_DIR)/libai_runtime.a
APP := $(BUILD_DIR)/traffic_app
INSPECT_APP := $(BUILD_DIR)/inspect_model
RUN_MODEL_APP := $(BUILD_DIR)/run_model
COMMON_EXAMPLE_DIR ?= ../ai_runtime_prototype/examples

# TFLITE_ROOT: thu muc chua include/ (tensorflow/lite/c/c_api.h,
# tensorflow/lite/delegates/external/external_delegate.h) va lib-$(ARCH)/
# (libtensorflowlite_c.so, libexternal_delegate.so). Duoc tu build tu
# TensorFlow source (v2.15.0, dung version voi QNN_TFLITE_DELEGATE_ReleaseNotes.txt)
# bang Bazel - xem docs/part19_tflite_delegate_setup.md.
# Include header dung chung cho moi ARCH (khong phu thuoc kien truc bien dich).
TFLITE_ROOT ?= third_party
TFLITE_LIB_DIR := $(TFLITE_ROOT)/lib-$(ARCH)

DL_QAIRT_ROOT ?= /home/congtuan/qairt_sdk/qairt/2.44.0.260225

ifeq ($(BACKEND),tflite_delegate)
BACKEND_SRC := src/backend_tflite_delegate.c
CFLAGS += -I$(TFLITE_ROOT)/include
LDFLAGS += -L$(TFLITE_LIB_DIR) -Wl,-rpath,'$$ORIGIN' -Wl,-rpath,$(abspath $(TFLITE_LIB_DIR))
LDLIBS += -ltensorflowlite_c -lexternal_delegate -ldl -lm
else
$(error unsupported BACKEND for this prototype: $(BACKEND) - only tflite_delegate is implemented here)
endif

RUNTIME_OBJS := \
	$(BUILD_DIR)/ai_runtime.o \
	$(BUILD_DIR)/backend.o

.PHONY: all clean run inspect run-model

all: $(APP)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/ai_runtime.o: src/ai_runtime.c include/ai_runtime.h src/backend.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/backend.o: $(BACKEND_SRC) src/backend.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(RUNTIME_OBJS)
	$(AR) rcs $@ $^

$(APP): examples/main.c $(LIB)
	$(CC) $(CFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LDLIBS)

run: $(APP)
	LD_LIBRARY_PATH=$(TFLITE_LIB_DIR):$(DL_QAIRT_ROOT)/lib/x86_64-linux-clang:$$LD_LIBRARY_PATH $(APP)

$(INSPECT_APP): $(COMMON_EXAMPLE_DIR)/inspect_model.c $(LIB)
	$(CC) $(CFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LDLIBS)

inspect: $(INSPECT_APP)

$(RUN_MODEL_APP): $(COMMON_EXAMPLE_DIR)/run_model.c $(LIB)
	$(CC) $(CFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LDLIBS)

run-model: $(RUN_MODEL_APP)

clean:
	rm -rf build
