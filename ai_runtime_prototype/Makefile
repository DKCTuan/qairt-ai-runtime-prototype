CC ?= gcc
AR ?= ar
BACKEND ?= mock

CFLAGS ?= -O2 -Wall -Wextra -Iinclude -Isrc

BUILD_DIR := build/$(BACKEND)
LIB := $(BUILD_DIR)/libai_runtime.a
APP := $(BUILD_DIR)/traffic_app

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

.PHONY: all clean run

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
	./$(APP)

clean:
	rm -rf build
