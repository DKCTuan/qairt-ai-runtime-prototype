CC ?= gcc
AR ?= ar
PYTHON ?= python3
BACKEND ?= mock

CFLAGS ?= -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L -Iinclude -Isrc

BUILD_DIR := build/$(BACKEND)
LIB := $(BUILD_DIR)/libai_runtime.a
APP := $(BUILD_DIR)/traffic_app
TEST_APP := $(BUILD_DIR)/runtime_api_test
TEST_MULTI_APP := $(BUILD_DIR)/runtime_multi_io_test
INSPECT_APP := $(BUILD_DIR)/inspect_model
RUN_MODEL_APP := $(BUILD_DIR)/run_model

DL_QAIRT_ROOT ?= /home/congtuan/qairt_sdk/qairt/2.44.0.260225

LDLIBS += -lm

ifeq ($(BACKEND),qnn_cli)
BACKEND_SRC := src/backend_qnn_cli.c
else ifeq ($(BACKEND),qnn_api)
BACKEND_SRC := src/backend_qnn_api.c
CFLAGS += -I$(DL_QAIRT_ROOT)/include/QNN
LDLIBS += -ldl 
else
BACKEND_SRC := src/backend_mock.c
endif

RUNTIME_OBJS := \
	$(BUILD_DIR)/ai_runtime.o \
	$(BUILD_DIR)/backend.o

.PHONY: all clean run test inspect run-model

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
	$(CC) $(CFLAGS) $< $(LIB) -o $@ $(LDLIBS)

run: $(APP)
	$(APP)

$(TEST_APP): tests/runtime_api_test.c $(LIB)
	$(CC) $(CFLAGS) $< $(LIB) -o $@ $(LDLIBS)

$(TEST_MULTI_APP): tests/runtime_multi_io_test.c $(LIB)
	$(CC) $(CFLAGS) $< $(LIB) -o $@ $(LDLIBS)

test: $(TEST_APP) $(TEST_MULTI_APP)
	$(TEST_APP)
	DL_MOCK_MULTI_IO=1 $(TEST_MULTI_APP)
	$(PYTHON) -m unittest tests/test_model_deploy.py

$(INSPECT_APP): examples/inspect_model.c $(LIB)
	$(CC) $(CFLAGS) $< $(LIB) -o $@ $(LDLIBS)

inspect: $(INSPECT_APP)

$(RUN_MODEL_APP): examples/run_model.c $(LIB)
	$(CC) $(CFLAGS) $< $(LIB) -o $@ $(LDLIBS)

run-model: $(RUN_MODEL_APP)

clean:
	rm -rf build
